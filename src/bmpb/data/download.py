"""Mirror the shared Google Drive folder into data/raw/.

The Drive folder is the upstream source of truth: the 25 CSVs, the original
`images/`, and the `processed_images/` the vision models consumed. Nothing here
is committed to git — run this once after cloning.

    bmpb data                  # CSVs + both image folders
    bmpb data --no-images      # CSVs only (~1 MB, enough for `make ingest`)

Why this is not a one-line `gdown --folder` call
------------------------------------------------
The folder contains a Google Doc ("Code"). A recursive download fails when it
tries to download that document. The top-level listing therefore uses
`skip_download` and downloads only its direct CSV files; image folders are
fetched separately by their IDs. A failure in one folder is reported without
aborting the others. Google can rate-limit bulk downloads, so re-running resumes
incomplete transfers and skips completed files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
            resume=True,
            timeout=30,
        )
        return True
    except Exception as error:  # noqa: BLE001 - one folder failing must not stop the rest
        log.warning("%s did not complete: %s", label, type(error).__name__)
        log.warning(
            "  this is usually Google throttling a bulk download. The transfer resumes, "
            "so re-run `bmpb data` in a few minutes to pick up the rest."
        )
        return False


def download_csv_files(folder_id: str, destination, label: str) -> bool:
    """List a Drive folder and download only its top-level CSV files."""
    gdown = _gdown()
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    log.info("fetching %s -> %s", label, destination)
    try:
        files = gdown.download_folder(
            url=FOLDER_URL.format(folder_id),
            output=str(destination),
            quiet=False,
            use_cookies=False,
            skip_download=True,
            timeout=30,
        )
        failed = False
        for file in files:
            relative_path = Path(file.path)
            target = (destination / relative_path).resolve()
            local_path = Path(file.local_path).resolve()
            if not target.is_relative_to(destination) or not local_path.is_relative_to(destination):
                log.warning("%s returned a path outside %s", label, destination)
                return False
            if relative_path.suffix.lower() != ".csv" or len(relative_path.parts) != 1:
                continue
            try:
                downloaded = gdown.download(
                    id=file.id,
                    output=str(target),
                    quiet=False,
                    use_cookies=False,
                    resume=True,
                    timeout=30,
                )
                if downloaded is None:
                    failed = True
                    log.warning("%s file %s did not complete", label, relative_path)
            except Exception as error:  # noqa: BLE001 - continue with other CSV files
                failed = True
                log.warning(
                    "%s file %s did not complete: %s", label, relative_path, type(error).__name__
                )
        return not failed
    except Exception as error:  # noqa: BLE001 - one folder failing must not stop the rest
        log.warning("%s did not complete: %s", label, type(error).__name__)
        log.warning(
            "  this is usually Google throttling a bulk download. The transfer resumes, "
            "so re-run `bmpb data` in a few minutes to pick up the rest."
        )
        return False


def download(cfg: DataConfig, images: bool = True) -> dict[str, int]:
    ensure_dirs()

    # List the root without downloading; only top-level CSV entries are files
    # we need from it. Image subfolders are fetched separately by their IDs.
    have_subfolder_ids = bool(cfg.drive_image_folder_id and cfg.drive_processed_image_folder_id)

    if have_subfolder_ids:
        download_csv_files(cfg.drive_folder_id, RAW, "CSVs (top-level folder)")
        if images:
            download_folder(cfg.drive_image_folder_id, RAW / cfg.image_dir, "images/")
            download_folder(
                cfg.drive_processed_image_folder_id,
                RAW / cfg.processed_image_dir,
                "processed_images/",
            )
    elif images:
        log.warning(
            "drive_image_folder_id / drive_processed_image_folder_id are not set in "
            "data.yaml; falling back to a recursive walk, which fails on the Google Doc "
            "in this folder"
        )
        download_folder(cfg.drive_folder_id, RAW, "whole folder")
    else:
        download_csv_files(cfg.drive_folder_id, RAW, "CSVs (top-level folder)")

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
