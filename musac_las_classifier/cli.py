"""
Command-line interface for musac_las_classifier.

Entry points:
    musac-classify  — Run multiclass classification on LAS files
    musac-polygons  — Extract building polygons from LAS files
"""

import argparse
import logging
import sys

from musac_las_classifier.config import PipelineConfig
from musac_las_classifier.pipeline import LasClassificationPipeline


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def classify():
    """CLI: Run multiclass or binary classification on LAS files."""
    parser = argparse.ArgumentParser(
        prog="musac-classify",
        description="Classify LAS point clouds using 3D-to-2D segmentation pipeline.",
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing input LAS file(s).",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path for the output classified LAS file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint (.pth).",
    )
    parser.add_argument(
        "--encoder-weights",
        required=True,
        help="Path to ResNet34 encoder weights (.pth).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=20,
        help="Number of segmentation classes (default: 20).",
    )
    parser.add_argument(
        "--workdir",
        default="workdir",
        help="Working directory for intermediate files (default: workdir).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Torch device (default: auto).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=250,
        help="Tile size in meters (default: 250).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.num_classes == 2:
        config = PipelineConfig.for_binary(
            checkpoint_path=args.checkpoint,
            encoder_weights_path=args.encoder_weights,
            device=args.device,
            tile_size=args.tile_size,
        )
    else:
        config = PipelineConfig.for_multiclass(
            checkpoint_path=args.checkpoint,
            encoder_weights_path=args.encoder_weights,
            device=args.device,
            tile_size=args.tile_size,
        )
        config.num_classes = args.num_classes

    config.validate()

    pipeline = LasClassificationPipeline(config, workdir=args.workdir)
    pipeline.load_las(args.input_dir)
    pipeline.run_classification(args.output)

    print(f"Classification complete: {args.output}")


def polygons():
    """CLI: Extract building polygons from LAS files."""
    parser = argparse.ArgumentParser(
        prog="musac-polygons",
        description="Extract building footprint polygons from LAS point clouds.",
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing input LAS file(s).",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Directory for output Shapefiles.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the binary building model checkpoint (.pth).",
    )
    parser.add_argument(
        "--encoder-weights",
        required=True,
        help="Path to ResNet34 encoder weights (.pth).",
    )
    parser.add_argument(
        "--workdir",
        default="workdir",
        help="Working directory for intermediate files (default: workdir).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum polygon area in pixels (default: 500).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Torch device (default: auto).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    config = PipelineConfig.for_binary(
        checkpoint_path=args.checkpoint,
        encoder_weights_path=args.encoder_weights,
        device=args.device,
        min_polygon_area=args.min_area,
    )
    config.validate()

    pipeline = LasClassificationPipeline(config, workdir=args.workdir)
    pipeline.load_las(args.input_dir)
    pipeline.run_polygon_extraction(args.output)

    print(f"Polygon extraction complete: {args.output}")


if __name__ == "__main__":
    classify()
