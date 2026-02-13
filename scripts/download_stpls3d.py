"""
Download STPLS3D dataset for training.

Downloads the 4 training regions (OCCC, RA, USC, WMSC) from the STPLS3D benchmark.
Files are saved to data/stpls3d/raw/.

Usage:
    python scripts/download_stpls3d.py
    python scripts/download_stpls3d.py --output-dir data/stpls3d/raw

Note:
    STPLS3D data is hosted on Google Drive. This script uses gdown for downloading.
    Install gdown: pip install gdown
"""

import argparse
import os
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None

# STPLS3D Google Drive file IDs for the 4 training regions
# Source: https://github.com/meidachen/STPLS3D
# TODO: Verify these IDs are current. Check the STPLS3D GitHub for updates.
STPLS3D_FILES = {
    "OCCC_points.las": None,  # TODO: Add Google Drive file ID
    "RA_points.las": None,    # TODO: Add Google Drive file ID
    "USC_points.las": None,   # TODO: Add Google Drive file ID
    "WMSC_points.las": None,  # TODO: Add Google Drive file ID
}

DEFAULT_OUTPUT_DIR = Path("data/stpls3d/raw")


def download_from_gdrive(file_id, output_path):
    """Download a file from Google Drive using gdown."""
    if gdown is None:
        print("ERROR: gdown is not installed. Install it: pip install gdown")
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(output_path), quiet=False)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download STPLS3D dataset")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"STPLS3D download directory: {output_dir}")
    print()

    has_missing_ids = False
    for filename, file_id in STPLS3D_FILES.items():
        dest = output_dir / filename
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  OK: {filename} ({size_mb:.1f} MB)")
            continue

        if file_id is None:
            print(f"  MISSING: {filename}")
            print(f"    Google Drive file ID not configured.")
            has_missing_ids = True
            continue

        print(f"  Downloading {filename}...")
        download_from_gdrive(file_id, dest)

    if has_missing_ids:
        print()
        print("Some files could not be downloaded automatically.")
        print("To download manually:")
        print("  1. Visit https://github.com/meidachen/STPLS3D")
        print("  2. Download the WMSC, USC, RA, and OCCC LAS files")
        print(f"  3. Place them in: {output_dir}")

    print()
    print("After downloading, prepare training data with:")
    print(f"  python training/prepare_data.py --raw-dir {output_dir}")


if __name__ == "__main__":
    main()
