"""Mirror the shared Google Drive folder into data/raw/.

The Drive folder is the upstream source of truth for the dataset: the CSVs, the
original `images/`, and the `processed_images/` produced by the preprocessing
pass. Nothing here is committed to git — run this once after cloning.

    python -m bmpb.data.download            # full mirror
    python -m bmpb.data.download --no-images   # CSVs only (fast, ~1 MB)

Requires the `fetch` extra:  pip install -e ".[fetch]"
"""

from __future__ import annotations

import argparse
import sys

from bmpb.config import DataConfig
from bmpb.paths import RAW, ensure_dirs
from bmpb.utils.logging import get_logger

log = get_logger(__name__)


def download(folder_id: str, images: bool = True) -> None:
    try:
        import gdown
    except ImportError:  # pragma: no cover - dependency guard
        sys.exit('gdown is not installed. Run: pip install -e ".[fetch]"')

    ensure_dirs()
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    log.info("Mirroring %s into %s", url, RAW)

    # gdown resumes: files already present are skipped, so re-running is cheap.
    gdown.download_folder(url=url, output=str(RAW), quiet=False, use_cookies=False)

    if not images:
        log.info(
            "--no-images was passed; the images/ folders may still have been "
            "fetched by gdown's folder walk. Delete them if you only want CSVs."
        )

    csvs = sorted(p.name for p in RAW.glob("*.csv"))
    log.info("%d CSV files in data/raw/", len(csvs))
    for directory in ("images", "processed_images"):
        count = len(list((RAW / directory).glob("*"))) if (RAW / directory).exists() else 0
        log.info("%s/: %d files", directory, count)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="data.yaml", help="data config to read the folder id from"
    )
    parser.add_argument("--no-images", action="store_true", help="skip the image folders")
    args = parser.parse_args(argv)

    cfg = DataConfig.load(args.config)
    download(cfg.drive_folder_id, images=not args.no_images)


if __name__ == "__main__":
    main()
