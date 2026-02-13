"""
Training script for U-Net segmentation models.

Trains a U-Net with ResNet34 encoder on STPLS3D feature images using Dice Loss.
Collects all metrics needed for paper: mIoU, best model, confusion matrices,
loss/mIoU curves, timing, and hardware info.

Usage:
    python training/train.py
    python training/train.py --epochs 50 --lr 0.001 --device cuda
"""

import argparse
import csv
import json
import logging
import os
import platform
import random
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from musac_las_classifier.constants import CLASS_COLORS, CLASS_NAMES
from training.config_training import TrainingConfig
from training.dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_hardware_info(device: str) -> dict:
    """Collect hardware information for paper reporting.

    Returns:
        Dict with CPU, GPU, CUDA, RAM, and library version info.
    """
    info = {
        "cpu_model": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device_used": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_model": (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else None),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        try:
            info["ram_gb"] = round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1
            )
        except (ValueError, AttributeError):
            info["ram_gb"] = None
    return info


def evaluate_model(model, data_loader, criterion, device, num_classes):
    """Evaluate model on a data loader.

    Returns:
        Tuple of (average_loss, confusion_matrix).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())

    preds_flat = np.concatenate([p.flatten() for p in all_preds])
    labels_flat = np.concatenate([l.flatten() for l in all_labels])
    conf_matrix = confusion_matrix(labels_flat, preds_flat, labels=range(num_classes))
    return total_loss / len(data_loader), conf_matrix


def compute_metrics(conf_matrix, class_idx):
    """Compute precision, recall, IoU, and F1 for a single class."""
    TP = conf_matrix[class_idx, class_idx]
    FP = conf_matrix[:, class_idx].sum() - TP
    FN = conf_matrix[class_idx, :].sum() - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, iou, f1


def compute_miou(conf_matrix, num_classes):
    """Compute mean IoU from a confusion matrix."""
    ious = []
    for cls in range(num_classes):
        _, _, iou, _ = compute_metrics(conf_matrix, cls)
        ious.append(iou)
    return float(np.mean(ious))


def save_loss_curves(epoch_summary_csv, output_path):
    """Plot train/val loss curves from the epoch summary CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(epoch_summary_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=1.5)
    ax.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Loss curves saved to %s", output_path)


def save_miou_curves(epoch_summary_csv, output_path):
    """Plot train/val mIoU curves from the epoch summary CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(epoch_summary_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_miou"], label="Train mIoU", linewidth=1.5)
    ax.plot(df["epoch"], df["val_miou"], label="Val mIoU", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title("Training and Validation mIoU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("mIoU curves saved to %s", output_path)


def save_confusion_matrix_image(conf_matrix, output_path, class_names=None, title=""):
    """Save confusion matrix as a heatmap image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = CLASS_NAMES

    num_classes = conf_matrix.shape[0]
    labels = [class_names.get(i, str(i)) for i in range(num_classes)]

    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    conf_normalized = conf_matrix / row_sums * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(conf_normalized, cmap="Blues", aspect="auto")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix (% per true class)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_model(model, train_loader, val_loader, config: TrainingConfig):
    """Train the segmentation model with full metric collection.

    Args:
        model: The U-Net model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training configuration.
    """
    device = config.resolve_device()
    model.to(device)
    criterion = smp.losses.DiceLoss("multiclass")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.confusion_matrix_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)
    num_classes = config.num_classes

    # Save hardware info
    hw_info = collect_hardware_info(device)
    hw_path = os.path.join(config.checkpoint_dir, "hardware_info.json")
    with open(hw_path, "w") as f:
        json.dump(hw_info, f, indent=2)
    logger.info("Hardware info saved to %s", hw_path)

    # Save training config
    config.save_json(config.config_json_path)
    logger.info("Training config saved to %s", config.config_json_path)

    # Per-class CSV metrics log (existing format)
    with open(config.metrics_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "dataset", "class", "precision", "recall", "iou", "f1"])

    # Epoch summary CSV (new)
    with open(config.epoch_summary_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_miou", "val_miou",
                          "epoch_time_sec"])

    # Best model tracking
    best_val_miou = -1.0
    best_epoch = -1

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Save checkpoint
        checkpoint_path = os.path.join(
            config.checkpoint_dir, f"model_epoch_{epoch + 1}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)

        # Evaluate
        val_loss, val_conf = evaluate_model(model, val_loader, criterion, device, num_classes)
        train_eval_loss, train_conf = evaluate_model(
            model, train_loader, criterion, device, num_classes
        )

        # Compute mIoU
        train_miou = compute_miou(train_conf, num_classes)
        val_miou = compute_miou(val_conf, num_classes)

        # Best model selection
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), config.best_model_path)
            logger.info("New best model at epoch %d (val mIoU: %.4f)",
                        best_epoch, best_val_miou)

        epoch_time = time.time() - epoch_start

        # Log per-class metrics
        with open(config.metrics_csv, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for cls in range(num_classes):
                train_metrics = compute_metrics(train_conf, cls)
                val_metrics = compute_metrics(val_conf, cls)
                writer.writerow([epoch + 1, "Train", cls, *train_metrics])
                writer.writerow([epoch + 1, "Val", cls, *val_metrics])

        # Log epoch summary
        avg_train_loss = train_loss / len(train_loader)
        with open(config.epoch_summary_csv, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, avg_train_loss, val_loss,
                              train_miou, val_miou, round(epoch_time, 2)])

        logger.info(
            "Epoch %d/%d — Train Loss: %.4f, Val Loss: %.4f, "
            "Train mIoU: %.4f, Val mIoU: %.4f, Time: %.1fs",
            epoch + 1, config.num_epochs, avg_train_loss, val_loss,
            train_miou, val_miou, epoch_time,
        )

    # Save final confusion matrices
    save_confusion_matrix_image(
        val_conf,
        os.path.join(config.confusion_matrix_dir, "confusion_matrix_val_final.png"),
        title=f"Validation Confusion Matrix (Epoch {config.num_epochs})",
    )
    save_confusion_matrix_image(
        train_conf,
        os.path.join(config.confusion_matrix_dir, "confusion_matrix_train_final.png"),
        title=f"Training Confusion Matrix (Epoch {config.num_epochs})",
    )
    np.save(os.path.join(config.confusion_matrix_dir, "val_conf_final.npy"), val_conf)
    np.save(os.path.join(config.confusion_matrix_dir, "train_conf_final.npy"), train_conf)

    # Save loss and mIoU curves
    save_loss_curves(
        config.epoch_summary_csv,
        os.path.join(config.plots_dir, "loss_curves.png"),
    )
    save_miou_curves(
        config.epoch_summary_csv,
        os.path.join(config.plots_dir, "miou_curves.png"),
    )

    logger.info("Training complete. Best model: epoch %d (val mIoU: %.4f)",
                best_epoch, best_val_miou)
    logger.info("Checkpoints saved to %s", config.checkpoint_dir)


def main():
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model")
    parser.add_argument("--features-dir", default=None, help="Feature images directory")
    parser.add_argument("--masks-dir", default=None, help="Class mask images directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_ratio=args.train_ratio,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        num_classes=args.num_classes,
        seed=args.seed,
        metrics_csv=os.path.join(args.checkpoint_dir, "metrics_log.csv"),
        epoch_summary_csv=os.path.join(args.checkpoint_dir, "epoch_summary.csv"),
        best_model_path=os.path.join(args.checkpoint_dir, "best_model.pth"),
        config_json_path=os.path.join(args.checkpoint_dir, "training_config.json"),
        confusion_matrix_dir=os.path.join(args.checkpoint_dir, "confusion_matrices"),
        plots_dir=os.path.join(args.checkpoint_dir, "plots"),
    )

    set_seed(config.seed)

    features_dir = args.features_dir or config.img_features_dir
    masks_dir = args.masks_dir or config.img_class_dir

    logger.info("Loading data: features=%s, masks=%s", features_dir, masks_dir)
    train_loader, val_loader = get_dataloaders(
        features_dir, masks_dir,
        train_ratio=config.train_ratio,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    model = smp.Unet(
        encoder_name=config.encoder_name,
        in_channels=config.in_channels,
        classes=config.num_classes,
    )

    train_model(model, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
