"""Decompose 3D RAW volume annotations into individual 2D slice images and standard single-image JSON files."""

import argparse
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from anylabeling.views.labeling.utils.raw_decompose import (
    clean_shape_to_standard_2d,
    decompose_raw_directory,
    decompose_single_raw,
    imwrite_unicode,
)

__all__ = [
    "clean_shape_to_standard_2d",
    "decompose_raw_directory",
    "decompose_single_raw",
    "imwrite_unicode",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decompose 3D RAW volume annotations into individual 2D images and standard 2D JSONs."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=r"D:\raw_test",
        help="Input folder containing .raw and .json files",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Output folder for decomposed slices (default: sibling folder {input_dir}_slices)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="jpg",
        choices=["jpg", "png", "bmp"],
        help="Image format for exported slices: jpg (default), png, or bmp",
    )
    parser.add_argument(
        "--all_slices",
        action="store_true",
        help="Export all slices instead of only annotated slices",
    )

    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else None

    decompose_raw_directory(
        input_dir=in_dir,
        output_dir=out_dir,
        only_annotated=not args.all_slices,
        img_format=args.format,
    )
