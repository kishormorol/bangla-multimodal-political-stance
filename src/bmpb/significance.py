"""Pairwise significance tests between CV model predictions.

Two tests per pair:
* **McNemar's test** — are the two models' error patterns significantly
  different? Only uses items both models predicted on.
* **Paired bootstrap** — is the difference in macro-F1 significant?
  Resamples the shared items 10 000 times and checks whether 0 falls outside
  the 95% percentile interval of the F1 difference.

The module works entirely from the pooled out-of-fold `predictions.csv` files
that every CV run writes.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from bmpb.paths import EXPERIMENTS, TABLES
from bmpb.utils.logging import get_logger

log = get_logger(__name__)


def _latest_predictions(runs_dir: Path) -> dict[str, pd.DataFrame]:
    """Return the latest predictions.csv for each model name, keyed by model."""
    candidates: dict[str, Path] = {}
    for run in sorted(runs_dir.glob("cv-*/")):
        preds = run / "predictions.csv"
        cv_json = run / "cv.json"
        if not preds.exists():
            continue
        if cv_json.exists():
            name = json.loads(cv_json.read_text()).get("name", run.name)
        else:
            # Fallback: extract model name from directory
            name = run.name.split("-", 1)[1].rsplit("-", 1)[0]
        candidates[name] = preds  # sorted order → last wins (newest)

    out = {}
    for name, path in candidates.items():
        df = pd.read_csv(path)
        out[name] = df
    return out


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray):
    """McNemar's test (mid-p corrected) on paired predictions.

    Returns (chi2, p_value, contingency dict).
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # b: A wrong, B right; c: A right, B wrong
    b = int(np.sum(~correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))

    # Mid-p McNemar (continuity-corrected chi-square for small samples)
    if b + c == 0:
        return 0.0, 1.0, {"b": b, "c": c}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) >= 25 else None

    from scipy.stats import binomtest
    p_value = binomtest(b, b + c, 0.5).pvalue

    if chi2 is None:
        chi2 = float("nan")

    return chi2, p_value, {"b": b, "c": c}


def _fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Macro-F1 without sklearn overhead — ~100x faster in tight loops."""
    total = 0.0
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        if tp == 0:
            total += 0.0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            total += 2 * prec * rec / (prec + rec)
    return total / n_classes


def paired_bootstrap_f1(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    rounds: int = 5_000,
    seed: int = 42,
) -> dict:
    """Bootstrap test for the difference in macro-F1 between two models.

    Returns dict with delta, ci_low, ci_high, p_value (two-sided).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_classes = max(3, int(y_true.max()) + 1)

    f1_a = _fast_macro_f1(y_true, pred_a, n_classes)
    f1_b = _fast_macro_f1(y_true, pred_b, n_classes)
    observed_delta = f1_a - f1_b

    deltas = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        deltas[i] = _fast_macro_f1(yt, pred_a[idx], n_classes) - _fast_macro_f1(yt, pred_b[idx], n_classes)

    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    # Two-sided p: proportion of bootstrap samples where sign flips
    if observed_delta >= 0:
        p_value = float(np.mean(deltas <= 0)) * 2
    else:
        p_value = float(np.mean(deltas >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "f1_a": round(f1_a, 4),
        "f1_b": round(f1_b, 4),
        "delta": round(observed_delta, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "p_value": round(p_value, 4),
        "significant_005": p_value < 0.05,
    }


def run_significance(
    runs_dir: Path = EXPERIMENTS, out_dir: Path = TABLES
) -> pd.DataFrame:
    """Run pairwise significance tests on all CV model pairs.

    Models are grouped by population (n=198 text vs n=86 multimodal) since
    only models evaluated on the same items can be compared.
    """
    all_preds = _latest_predictions(runs_dir)
    if len(all_preds) < 2:
        raise RuntimeError(f"Need >= 2 models, found {len(all_preds)}")

    # Group by population size
    groups: dict[int, dict[str, pd.DataFrame]] = {}
    for name, df in all_preds.items():
        n = len(df)
        groups.setdefault(n, {})[name] = df

    rows = []
    for pop_size, models in sorted(groups.items()):
        if len(models) < 2:
            continue
        log.info("Testing %d models on population n=%d", len(models), pop_size)

        # Align all predictions by item_id
        names = sorted(models.keys())
        for name_a, name_b in combinations(names, 2):
            df_a = models[name_a].set_index("item_id")
            df_b = models[name_b].set_index("item_id")
            shared = df_a.index.intersection(df_b.index)
            if len(shared) < 10:
                continue

            y_true = df_a.loc[shared, "label"].values
            pred_a = df_a.loc[shared, "predicted"].values
            pred_b = df_b.loc[shared, "predicted"].values

            # McNemar
            chi2, mcn_p, cont = mcnemar_test(y_true, pred_a, pred_b)

            # Paired bootstrap
            boot = paired_bootstrap_f1(y_true, pred_a, pred_b)

            rows.append({
                "model_a": name_a,
                "model_b": name_b,
                "n": len(shared),
                "f1_a": boot["f1_a"],
                "f1_b": boot["f1_b"],
                "delta": boot["delta"],
                "boot_ci_low": boot["ci_low"],
                "boot_ci_high": boot["ci_high"],
                "boot_p": boot["p_value"],
                "boot_sig": boot["significant_005"],
                "mcnemar_b": cont["b"],
                "mcnemar_c": cont["c"],
                "mcnemar_p": round(mcn_p, 4),
                "mcnemar_sig": mcn_p < 0.05,
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        log.warning("No pairs to test")
        return frame

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    frame.to_csv(out_dir / "significance.csv", index=False)

    # Build a readable markdown table
    lines = [
        "# Pairwise Significance Tests",
        "",
        "Two tests per pair: paired bootstrap on macro-F1 difference (10k rounds)",
        "and McNemar's exact test on error disagreement. Only models evaluated on",
        "the same items are compared. Significance at p < 0.05.",
        "",
        "| Model A | Model B | n | F1(A) | F1(B) | Delta | 95% CI | Boot p | Sig | McNemar p | Sig |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for r in frame.itertuples():
        ci = f"{r.boot_ci_low:.3f} to {r.boot_ci_high:.3f}"
        lines.append(
            f"| {r.model_a} | {r.model_b} | {r.n} | "
            f"{r.f1_a:.3f} | {r.f1_b:.3f} | {r.delta:+.3f} | {ci} | "
            f"{r.boot_p:.3f} | {'*' if r.boot_sig else ''} | "
            f"{r.mcnemar_p:.3f} | {'*' if r.mcnemar_sig else ''} |"
        )

    # Summary
    n_sig_boot = int(frame["boot_sig"].sum())
    n_sig_mcn = int(frame["mcnemar_sig"].sum())
    n_total = len(frame)
    lines.extend([
        "",
        f"**{n_sig_boot}/{n_total}** pairs significant by bootstrap, "
        f"**{n_sig_mcn}/{n_total}** by McNemar (p < 0.05).",
    ])

    md_path = out_dir / "significance.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s (%d pairs)", md_path, n_total)
    return frame
