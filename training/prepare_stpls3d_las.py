"""
Prepare STPLS3D PLY data for training by converting to stretched LAS tiles.

Key behavior:
- Converts PLY -> LAS with class and RGB preserved.
- Normalizes XY translation so min(x)=0 and min(y)=0.
- Stretches XY to nearest multiple of tile size.
- Tiles scene into fixed-size blocks.
- Downsamples only when a tile exceeds target points per tile.
- Produces CSV and JSON reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

try:
    from training.utils.ply_to_las import read_and_prepare_scene, tile_scene, write_las
except ModuleNotFoundError:
    from utils.ply_to_las import read_and_prepare_scene, tile_scene, write_las


LOGGER = logging.getLogger("prepare_stpls3d_las")


@dataclass
class FileResult:
    file_name: str
    input_ply: str
    output_scene_las: str
    points_in: int
    points_after_stretch: int
    points_after_tiling_total: int
    min_x_before_shift: float
    min_y_before_shift: float
    size_x_before_stretch: float
    size_y_before_stretch: float
    target_x: float
    target_y: float
    sx: float
    sy: float
    expected_tiles: int
    actual_tiles: int
    empty_tiles_count: int
    low_density_tiles_count: int
    low_density_tiles_ratio: float
    downsampled_tiles_count: int
    min_points_in_tile: int
    max_points_in_tile: int
    mean_points_in_tile: float
    has_low_density_tiles: bool
    target_points_per_tile: int
    status: str
    error: str


def _configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _process_single_file(
    ply_path: Path,
    scenes_dir: Path,
    tiles_dir: Path,
    tile_size: int,
    target_points_per_tile: int,
    seed: int,
    overwrite: bool,
) -> Dict[str, Any]:
    scene = read_and_prepare_scene(ply_path, tile_size=tile_size)

    scene_las_path = scenes_dir / f"{scene.stem}.las"
    if overwrite or not scene_las_path.exists():
        write_las(
            scene_las_path,
            scene.x,
            scene.y,
            scene.z,
            scene.classification,
            scene.red,
            scene.green,
            scene.blue,
        )

    tile_stats = tile_scene(
        scene=scene,
        tile_size=tile_size,
        target_points_per_tile=target_points_per_tile,
        seed=seed,
        tiles_output_dir=tiles_dir,
    )

    expected_tiles = int(tile_stats["expected_tiles"])
    low_density_count = int(tile_stats["low_density_tiles"])
    low_density_ratio = (
        float(low_density_count) / float(expected_tiles) if expected_tiles else 0.0
    )

    file_result = FileResult(
        file_name=scene.stem,
        input_ply=str(ply_path),
        output_scene_las=str(scene_las_path),
        points_in=int(scene.x.size),
        points_after_stretch=int(scene.x.size),
        points_after_tiling_total=int(tile_stats["total_points_after"]),
        min_x_before_shift=float(scene.min_x_before_shift),
        min_y_before_shift=float(scene.min_y_before_shift),
        size_x_before_stretch=float(scene.size_x_before_stretch),
        size_y_before_stretch=float(scene.size_y_before_stretch),
        target_x=float(scene.target_x),
        target_y=float(scene.target_y),
        sx=float(scene.sx),
        sy=float(scene.sy),
        expected_tiles=expected_tiles,
        actual_tiles=int(tile_stats["actual_tiles"]),
        empty_tiles_count=int(tile_stats["empty_tiles"]),
        low_density_tiles_count=low_density_count,
        low_density_tiles_ratio=low_density_ratio,
        downsampled_tiles_count=int(tile_stats["downsampled_tiles"]),
        min_points_in_tile=int(tile_stats["min_points_in_tile"]),
        max_points_in_tile=int(tile_stats["max_points_in_tile"]),
        mean_points_in_tile=float(tile_stats["mean_points_in_tile"]),
        has_low_density_tiles=low_density_count > 0,
        target_points_per_tile=int(target_points_per_tile),
        status="ok",
        error="",
    )

    return {
        "file": asdict(file_result),
        "tiles": tile_stats["per_tile_records"],
        "post_checks": {
            "min_x_after_shift_and_stretch": float(scene.x.min()),
            "min_y_after_shift_and_stretch": float(scene.y.min()),
        },
    }


def _build_summary(
    file_rows: List[Dict[str, Any]],
    tile_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    files_ok = [r for r in file_rows if r.get("status") == "ok"]
    files_failed = [r for r in file_rows if r.get("status") != "ok"]
    total_expected = int(sum(float(r.get("expected_tiles", 0)) for r in file_rows))
    total_actual = int(sum(float(r.get("actual_tiles", 0)) for r in file_rows))
    low_density_tiles = int(sum(float(r.get("low_density_tiles_count", 0)) for r in file_rows))
    downsampled_tiles = int(sum(float(r.get("downsampled_tiles_count", 0)) for r in file_rows))
    points_in_total = int(sum(float(r.get("points_in", 0)) for r in file_rows))
    points_after_total = int(sum(float(r.get("points_after_tiling_total", 0)) for r in file_rows))

    return {
        "config": {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "tile_size": args.tile_size,
            "target_points_per_tile": args.target_points_per_tile,
            "seed": args.seed,
            "workers": args.workers,
            "overwrite": args.overwrite,
        },
        "counts": {
            "files_total": len(file_rows),
            "files_ok": len(files_ok),
            "files_failed": len(files_failed),
            "tiles_records_total": len(tile_rows),
            "tiles_expected_total": total_expected,
            "tiles_written_total": total_actual,
        },
        "density": {
            "low_density_tiles_total": low_density_tiles,
            "downsampled_tiles_total": downsampled_tiles,
            "low_density_ratio_global": (
                float(low_density_tiles) / float(total_expected) if total_expected else 0.0
            ),
        },
        "points": {
            "points_in_total": points_in_total,
            "points_after_tiling_total": points_after_total,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert STPLS3D PLY to stretched LAS tiles with reports.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/stpls3d/raw_ply"),
        help="Directory with source .ply files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/stpls3d_prepared"),
        help="Base output directory for LAS and reports.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=250,
        help="Tile side length in meters.",
    )
    parser.add_argument(
        "--target-points-per-tile",
        type=int,
        default=2_000_000,
        help="Target density threshold. Tiles above this are downsampled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for deterministic downsampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for file-level parallelism.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene LAS files.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first file error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scenes_dir = args.output_dir / "scenes_las"
    tiles_dir = args.output_dir / "tiles_las"
    reports_dir = args.output_dir / "reports"
    logs_dir = args.output_dir / "logs"

    _configure_logging(logs_dir / "prepare.log")
    LOGGER.info("Starting STPLS3D LAS preparation")
    LOGGER.info("Input dir:  %s", args.input_dir)
    LOGGER.info("Output dir: %s", args.output_dir)

    if not args.input_dir.exists():
        LOGGER.error("Input dir does not exist: %s", args.input_dir)
        return 1

    ply_files = sorted(args.input_dir.glob("*.ply"))
    if not ply_files:
        LOGGER.error("No .ply files found in %s", args.input_dir)
        return 1

    scenes_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    file_rows: List[Dict[str, Any]] = []
    tile_rows: List[Dict[str, Any]] = []

    def handle_result(ply_path: Path, payload: Dict[str, Any]) -> None:
        file_rows.append(payload["file"])
        tile_rows.extend(payload["tiles"])
        post = payload["post_checks"]
        min_x = post["min_x_after_shift_and_stretch"]
        min_y = post["min_y_after_shift_and_stretch"]
        if not math.isclose(min_x, 0.0, abs_tol=1e-6):
            LOGGER.warning("%s: min(x) after shift/stretch is %s (expected 0)", ply_path.name, min_x)
        if not math.isclose(min_y, 0.0, abs_tol=1e-6):
            LOGGER.warning("%s: min(y) after shift/stretch is %s (expected 0)", ply_path.name, min_y)

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _process_single_file,
                    ply_path,
                    scenes_dir,
                    tiles_dir,
                    args.tile_size,
                    args.target_points_per_tile,
                    args.seed,
                    args.overwrite,
                ): ply_path
                for ply_path in ply_files
            }
            for future in as_completed(futures):
                ply_path = futures[future]
                try:
                    payload = future.result()
                    handle_result(ply_path, payload)
                    LOGGER.info("Processed %s", ply_path.name)
                except Exception as exc:
                    LOGGER.exception("Failed %s", ply_path.name)
                    file_rows.append(
                        asdict(
                            FileResult(
                                file_name=ply_path.stem,
                                input_ply=str(ply_path),
                                output_scene_las="",
                                points_in=0,
                                points_after_stretch=0,
                                points_after_tiling_total=0,
                                min_x_before_shift=0.0,
                                min_y_before_shift=0.0,
                                size_x_before_stretch=0.0,
                                size_y_before_stretch=0.0,
                                target_x=0.0,
                                target_y=0.0,
                                sx=0.0,
                                sy=0.0,
                                expected_tiles=0,
                                actual_tiles=0,
                                empty_tiles_count=0,
                                low_density_tiles_count=0,
                                low_density_tiles_ratio=0.0,
                                downsampled_tiles_count=0,
                                min_points_in_tile=0,
                                max_points_in_tile=0,
                                mean_points_in_tile=0.0,
                                has_low_density_tiles=False,
                                target_points_per_tile=args.target_points_per_tile,
                                status="error",
                                error=str(exc),
                            )
                        )
                    )
                    if args.fail_fast:
                        raise
    else:
        for ply_path in ply_files:
            try:
                payload = _process_single_file(
                    ply_path=ply_path,
                    scenes_dir=scenes_dir,
                    tiles_dir=tiles_dir,
                    tile_size=args.tile_size,
                    target_points_per_tile=args.target_points_per_tile,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
                handle_result(ply_path, payload)
                LOGGER.info("Processed %s", ply_path.name)
            except Exception as exc:
                LOGGER.exception("Failed %s", ply_path.name)
                file_rows.append(
                    asdict(
                        FileResult(
                            file_name=ply_path.stem,
                            input_ply=str(ply_path),
                            output_scene_las="",
                            points_in=0,
                            points_after_stretch=0,
                            points_after_tiling_total=0,
                            min_x_before_shift=0.0,
                            min_y_before_shift=0.0,
                            size_x_before_stretch=0.0,
                            size_y_before_stretch=0.0,
                            target_x=0.0,
                            target_y=0.0,
                            sx=0.0,
                            sy=0.0,
                            expected_tiles=0,
                            actual_tiles=0,
                            empty_tiles_count=0,
                            low_density_tiles_count=0,
                            low_density_tiles_ratio=0.0,
                            downsampled_tiles_count=0,
                            min_points_in_tile=0,
                            max_points_in_tile=0,
                            mean_points_in_tile=0.0,
                            has_low_density_tiles=False,
                            target_points_per_tile=args.target_points_per_tile,
                            status="error",
                            error=str(exc),
                        )
                    )
                )
                if args.fail_fast:
                    raise

    summary = _build_summary(file_rows=file_rows, tile_rows=tile_rows, args=args)

    _write_csv(file_rows, reports_dir / "per_file_report.csv")
    _write_csv(tile_rows, reports_dir / "per_tile_report.csv")

    with (reports_dir / "per_file_report.json").open("w", encoding="utf-8") as f:
        json.dump(file_rows, f, indent=2)
    with (reports_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Done. Files processed: %d", len(file_rows))
    LOGGER.info("Reports: %s", reports_dir)

    has_errors = any(r.get("status") != "ok" for r in file_rows)
    return 2 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
