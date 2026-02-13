"""
Evaluation script for trained segmentation models.

Loads a checkpoint and evaluates on the validation set,
printing per-class metrics and overall mIoU.

Usage:
    python training/evaluate.py --checkpoint checkpoints/model_epoch_31.pth
    python training/evaluate.py --checkpoint checkpoints/best_model.pth --output-dir eval_results
"""

import argparse
import csv
import json
import logging
import os
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch

from musac_las_classifier.constants import CLASS_COLORS, CLASS_NAMES
from training.config_training import TrainingConfig
from training.dataset import get_dataloaders
from training.train import (
    collect_hardware_info,
    compute_metrics,
    compute_miou,
    evaluate_model,
    save_confusion_matrix_image,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--masks-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for same val split)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save evaluation results")
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        device=args.device,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    device = config.resolve_device()

    features_dir = args.features_dir or config.img_features_dir
    masks_dir = args.masks_dir or config.img_class_dir

    _, val_loader = get_dataloaders(
        features_dir, masks_dir,
        train_ratio=0.8,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=args.num_classes,
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=False)
    )
    model.to(device)

    criterion = smp.losses.DiceLoss("multiclass")

    start_time = time.time()
    val_loss, conf_matrix = evaluate_model(
        model, val_loader, criterion, device, args.num_classes
    )
    eval_time = time.time() - start_time

    logger.info("Validation Loss: %.4f", val_loss)
    print(f"\n{'Class':>3} {'Name':<20} {'Prec':>8} {'Recall':>8} {'IoU':>8} {'F1':>8}")
    print("-" * 60)

    ious = []
    all_metrics = []
    for cls in range(args.num_classes):
        prec, rec, iou, f1 = compute_metrics(conf_matrix, cls)
        name = CLASS_NAMES.get(cls, f"Class {cls}")
        print(f"{cls:>3} {name:<20} {prec:>8.4f} {rec:>8.4f} {iou:>8.4f} {f1:>8.4f}")
        ious.append(iou)
        all_metrics.append((cls, name, prec, rec, iou, f1))

    miou = np.mean(ious)
    print(f"\nmIoU: {miou:.4f}")
    print(f"Evaluation time: {eval_time:.1f}s")

    # Save outputs if --output-dir is specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Confusion matrix image
        save_confusion_matrix_image(
            conf_matrix,
            os.path.join(args.output_dir, "confusion_matrix_eval.png"),
            title="Evaluation Confusion Matrix",
        )

        # Raw confusion matrix
        np.save(os.path.join(args.output_dir, "confusion_matrix_eval.npy"), conf_matrix)

        # Per-class metrics CSV
        csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "name", "precision", "recall", "iou", "f1"])
            for cls, name, prec, rec, iou, f1 in all_metrics:
                writer.writerow([cls, name, prec, rec, iou, f1])
            writer.writerow(["all", "mIoU", "", "", miou, ""])

        # Summary JSON
        summary = {
            "checkpoint": args.checkpoint,
            "val_loss": float(val_loss),
            "miou": float(miou),
            "eval_time_sec": round(eval_time, 2),
            "num_classes": args.num_classes,
            "num_val_batches": len(val_loader),
            "hardware": collect_hardware_info(device),
        }
        with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Evaluation results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
