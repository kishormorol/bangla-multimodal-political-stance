"""Early-stopping counts for the matched run with the corrected BLIP.

The published matched run's BLIP never saw the image (see HANDOFF.md,
2026-11-20). Its corrected replacement is the seed-42 BLIP run made with the
fix (commit e5cc1a6), which shares the matched run's seed, items and folds.
This counts early stops exactly as summarize_training_log.py does, over the
published Kaggle log with its paired_headlines/blip section replaced by the
fixed run's section.

    python scripts/early_stopping_blipfix.py \
        experiments/kaggle-bundle/bmpb-matched-sweep.log \
        experiments/seeds149/rerun2.log

The second log must contain "HEAD now e5cc1a6"; the first
"=== paired_headlines / blip " section after it is the seed-42 run.
Writes reports/submission/early_stopping.json.
"""

import collections
import json
import re
import sys
from pathlib import Path

HEADER = re.compile(r"^=== \w+ / \w+", re.MULTILINE)


def kaggle_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    try:
        return "".join(record.get("data", "") for record in json.loads(raw))
    except (json.JSONDecodeError, AttributeError):
        return raw


def section(text: str, start: int) -> str:
    following = HEADER.search(text, start + 1)
    end = following.start() if following else len(text)
    # a sweep can also end with a non-header line such as "blip seed 42 exit"
    stop = re.search(r"^blip seed \d+ exit", text[start:end], re.MULTILINE)
    return text[start : start + stop.start()] if stop else text[start:end]


def main():
    published = kaggle_text(Path(sys.argv[1]))
    rerun = Path(sys.argv[2]).read_text(encoding="utf-8-sig")

    old = re.search(r"^=== paired_headlines / blip\b(?!_)", published, re.MULTILINE)
    old_section = section(published, old.start())
    fixed_from = rerun.index("HEAD now e5cc1a6")
    new = re.compile(r"^=== paired_headlines / blip \(", re.MULTILINE).search(rerun, fixed_from)
    new_section = section(rerun, new.start())

    text = published.replace(old_section, new_section, 1)
    fold_runs = len(re.findall(r"device: cuda", text))
    best = [int(epoch) for epoch in re.findall(r"best was epoch (\d+)", text)]
    summary = {
        "neural_fold_runs": fold_runs,
        "early_stopped": len(best),
        "selected_epoch_when_early_stopped": dict(sorted(collections.Counter(best).items())),
        "early_stopped_selecting_epoch_1_or_2": sum(epoch <= 2 for epoch in best),
        "blip_section": {
            "published_fold_runs": len(re.findall(r"device: cuda", old_section)),
            "published_early_stops": len(re.findall(r"best was epoch", old_section)),
            "fixed_fold_runs": len(re.findall(r"device: cuda", new_section)),
            "fixed_early_stops": len(re.findall(r"best was epoch", new_section)),
        },
    }
    destination = Path("reports/submission/early_stopping.json")
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
