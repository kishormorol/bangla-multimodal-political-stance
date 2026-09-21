"""Scoring.

Macro-F1 is the headline number: the corpus is imbalanced at the article level
(103 govt_critique / 53 neutral / 42 govt_leaning before balancing), so accuracy
rewards a model for following the majority class. Every table also carries the
majority-class floor and a bootstrap interval, because with a 30–47 row test set
a 2-point difference between two models is not a result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from bmpb.data.ingest import LABEL_NAMES


@dataclass
class Scores:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    support: dict[str, int]
    confusion: list[list[int]]
    n: int
    macro_f1_ci95: tuple[float, float] | None = None
    majority_baseline_accuracy: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        ci = ""
        if self.macro_f1_ci95:
            ci = f" [95% CI {self.macro_f1_ci95[0]:.3f}–{self.macro_f1_ci95[1]:.3f}]"
        return f"n={self.n}  acc={self.accuracy:.3f}  macro-F1={self.macro_f1:.3f}{ci}"


def bootstrap_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, rounds: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap interval for macro-F1, resampling test items."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    samples = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        samples[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def score(
    y_true, y_pred, *, labels: tuple[str, ...] = LABEL_NAMES, bootstrap: bool = True, seed: int = 42
) -> Scores:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    label_ids = list(range(len(labels)))

    _, _, per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    counts = np.bincount(y_true, minlength=len(labels))
    majority = float(counts.max() / len(y_true)) if len(y_true) else float("nan")

    notes = []
    if len(y_true) < 100:
        notes.append(
            f"test set is {len(y_true)} items; differences smaller than "
            "the CI width are not meaningful"
        )
    if len(set(y_pred.tolist())) == 1:
        notes.append("model predicted a single class for every item (degenerate)")

    return Scores(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        per_class_f1={name: float(v) for name, v in zip(labels, per_class, strict=True)},
        support={name: int(v) for name, v in zip(labels, support, strict=True)},
        confusion=confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
        n=int(len(y_true)),
        macro_f1_ci95=bootstrap_macro_f1(y_true, y_pred, seed=seed) if bootstrap else None,
        majority_baseline_accuracy=majority,
        notes=notes,
    )


def report(y_true, y_pred, labels: tuple[str, ...] = LABEL_NAMES) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(len(labels))), target_names=list(labels), zero_division=0
    )


def annotator_agreement(frame, columns: list[str]) -> dict[str, float]:
    """Pairwise Cohen's kappa between annotator columns.

    The raw sheet has three annotators with gaps (Annotator 1 is absent for
    ~40 rows), so each pair is scored on the rows both of them labelled.
    """
    out: dict[str, float] = {}
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            if left not in frame or right not in frame:
                continue
            both = frame[[left, right]].dropna()
            if len(both) < 2:
                continue
            out[f"{left}|{right}"] = float(
                cohen_kappa_score(both[left].astype(str), both[right].astype(str))
            )
    return out
