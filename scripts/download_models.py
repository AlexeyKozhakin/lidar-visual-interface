"""
Download pretrained model weights.

Downloads ResNet34 encoder weights and trained segmentation checkpoints
to the models/ directory.

Usage:
    python scripts/download_models.py
"""

import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

# ResNet34 ImageNet pretrained weights (PyTorch official)
RESNET34_URL = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
RESNET34_FILE = "resnet34-333f7ec4.pth"

# TODO: Host trained checkpoints on a public URL (e.g., GitHub Releases, HuggingFace)
# For now, these files must be provided manually.
MODEL_FILES = {
    RESNET34_FILE: RESNET34_URL,
    # "model_epoch_25_binary.pth": None,        # Not yet publicly hosted
    # "model_epoch_31_multiclass.pth": None,     # Not yet publicly hosted
}


def download_file(url, dest_path):
    """Download a file from URL to destination path."""
    if dest_path.exists():
        print(f"  Already exists: {dest_path.name}")
        return

    print(f"  Downloading {dest_path.name}...")
    urllib.request.urlretrieve(url, str(dest_path))
    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {dest_path.name} ({size_mb:.1f} MB)")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {MODELS_DIR}")

    for filename, url in MODEL_FILES.items():
        dest = MODELS_DIR / filename
        if url is None:
            if not dest.exists():
                print(f"  MISSING: {filename} (no public URL, provide manually)")
            else:
                print(f"  OK: {filename}")
        else:
            download_file(url, dest)

    # Check for trained checkpoints
    for name in ["model_epoch_25_binary.pth", "model_epoch_31_multiclass.pth"]:
        p = MODELS_DIR / name
        if p.exists():
            print(f"  OK: {name}")
        else:
            print(f"  MISSING: {name} — copy from training output or download manually")

    print("\nDone.")


if __name__ == "__main__":
    main()
