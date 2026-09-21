"""Mirror the shared Google Drive folder into data/raw/.

The Drive folder is the upstream source of truth: the 25 CSVs, the original
`images/`, and the `processed_images/` the vision models consumed. Nothing here
is committed to git — run this once after cloning.

    bmpb data                  # CSVs + both image folders
    bmpb data --no-images      # CSVs only (~1 MB, enough for `make ingest`)

Why this is not a one-line `gdown --folder` call
------------------------------------------------
The folder contains a Google Doc ("Code"). Docs have no binary download
endpoint, so gdown's recursive walk fails when it reaches that entry and takes
the whole listing down with it — including the image folders it had not visited
yet. The error message is the misleading "Cannot retrieve the public link … or
have had many accesses", which looks like rate limiting and is not.

So each folder is fetched separately, by id, and a failure in one is reported
without aborting the others. Google *does* also rate-limit bulk downloads, and
that failure looks identical; in that case the download resumes, so re-running
picks up only what is missing.
"""

from __future__ import annotations

import argparse
import sys

from bmpb.config import DataConfig
from bmpb.paths import RAW, ensure_dirs
from bmpb.utils.logging import get_logger

log = get_logger(__name__)

FOLDER_URL = "https://drive.google.com/drive/folders/{}"


def _gdown():
    try:
        import gdown
    except ImportError:  # pragma: no cover - dependency guard
        sys.exit('gdown is not installed. Run: pip install -e ".[fetch]"')
    return gdown


def download_folder(folder_id: str, destination, label: str) -> bool:
    """Fetch one Drive folder. Returns False on failure instead of raising."""
    gdown = _gdown()
    destination.mkdir(parents=True, exist_ok=True)
    log.info("fetching %s -> %s", label, destination)
    try:
        gdown.download_folder(
            url=FOLDER_URL.format(folder_id),
            output=str(destination),
            quiet=False,
            use_cookies=False,
        )
        return True
    except Exception as error:  # noqa: BLE001 - one folder failing must not stop the rest
        log.warning("%s did not complete: %s", label, type(error).__name__)
        log.warning(
            "  this is usually Google throttling a bulk download. The transfer resumes, "
            "so re-run `bmpb data` in a few minutes to pick up the rest."
        )
        return False


def download(cfg: DataConfig, images: bool = True) -> dict[str, int]:
    ensure_dirs()

    # The top-level walk is what trips over the Google Doc, so when the image
    # folders are addressed by their own ids we never ask gdown to recurse.
    have_subfolder_ids = bool(cfg.drive_image_folder_id and cfg.drive_processed_image_folder_id)

    if have_subfolder_ids:
        download_folder(cfg.drive_folder_id, RAW, "CSVs (top-level folder)")
        if images:
            download_folder(cfg.drive_image_folder_id, RAW / cfg.image_dir, "images/")
            download_folder(
                cfg.drive_processed_image_folder_id,
                RAW / cfg.processed_image_dir,
                "processed_images/",
            )
    else:
        log.warning(
            "drive_image_folder_id / drive_processed_image_folder_id are not set in "
            "data.yaml; falling back to a recursive walk, which fails on the Google Doc "
            "in this folder"
        )
        download_folder(cfg.drive_folder_id, RAW, "whole folder")

    counts = {
        "csv": len(list(RAW.glob("*.csv"))),
        cfg.image_dir: (
            len([p for p in (RAW / cfg.image_dir).glob("*") if p.is_file()])
            if (RAW / cfg.image_dir).exists()
            else 0
        ),
        cfg.processed_image_dir: (
            len([p for p in (RAW / cfg.processed_image_dir).glob("*") if p.is_file()])
            if (RAW / cfg.processed_image_dir).exists()
            else 0
        ),
    }
    log.info("data/raw now holds: %s", counts)
    if counts["csv"] < 25:
        log.warning("expected 25 CSVs; re-run `bmpb data` to fetch the rest")
    if images and counts[cfg.processed_image_dir] < 149:
        log.warning(
            "expected 149 processed_images (the multimodal models need them); "
            "re-run `bmpb data` to fetch the rest"
        )
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data.yaml")
    parser.add_argument("--no-images", action="store_true", help="fetch the CSVs only")
    args = parser.parse_args(argv)
    download(DataConfig.load(args.config), images=not args.no_images)


if __name__ == "__main__":
    main()
