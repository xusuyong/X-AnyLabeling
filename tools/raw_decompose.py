"""Decompose 3D RAW volume annotations into individual 2D slice PNG images and standard single-image JSON files."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from anylabeling.views.labeling.utils.raw_reader import parse_raw_file_info, RawVolume


def clean_shape_to_standard_2d(shape: dict) -> dict:
    """Clean a shape dict to standard 2D AnyLabeling/Labelme format without 3D specific metadata."""
    cleaned = {
        "label": shape.get("label", ""),
        "points": shape.get("points", []),
        "group_id": shape.get("group_id", None),
        "description": shape.get("description", None),
        "difficult": shape.get("difficult", False),
        "shape_type": shape.get("shape_type", "rectangle"),
        "flags": shape.get("flags", {}),
        "attributes": shape.get("attributes", {}),
        "kie_linking": shape.get("kie_linking", []),
    }
    # Preserve score if available
    if "score" in shape:
        cleaned["score"] = shape["score"]

    # If other_data exists, copy user attributes without 3D slice/z fields
    other_data = shape.get("other_data", {})
    if isinstance(other_data, dict):
        custom_other = {
            k: v
            for k, v in other_data.items()
            if k not in ("slice_index", "z", "raw_current_z", "raw_info")
        }
        if custom_other:
            cleaned["other_data"] = custom_other

    return cleaned


def decompose_single_raw(
    raw_path: Path,
    json_path: Path,
    output_dir: Path,
    only_annotated: bool = True,
    img_format: str = "jpg",
) -> int:
    """Decompose one (.raw, .json) pair into individual 2D images and standard 2D JSONs.

    Returns the number of exported slices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = img_format.lower().lstrip(".")
    if fmt not in ("jpg", "jpeg", "png", "bmp"):
        fmt = "jpg"
    ext = ".jpg" if fmt in ("jpg", "jpeg") else f".{fmt}"

    # 1. Parse RAW metadata
    raw_info = parse_raw_file_info(str(raw_path))
    if not raw_info:
        print(f"[!] Cannot parse RAW metadata for {raw_path.name}")
        return 0

    # 2. Parse JSON annotations
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
    except Exception as e:
        print(f"[!] Failed to read {json_path.name}: {e}")
        return 0

    shapes = label_data.get("shapes", [])
    version = label_data.get("version", "4.0.6")
    flags = label_data.get("flags", {})

    # Group shapes by slice_index
    slice_shapes_map: Dict[int, List[dict]] = {}
    for sh in shapes:
        s_idx = sh.get("slice_index")
        if s_idx is None:
            s_idx = sh.get("other_data", {}).get("slice_index", 0)
        s_idx = int(s_idx)

        if s_idx not in slice_shapes_map:
            slice_shapes_map[s_idx] = []
        slice_shapes_map[s_idx].append(sh)

    # Determine which slices to export
    if only_annotated:
        target_slices = sorted(slice_shapes_map.keys())
    else:
        target_slices = list(range(raw_info.depth))

    if not target_slices:
        print(f"[-] No annotated slices found in {json_path.name}")
        return 0

    # 3. Read slices from volume and export
    vol = RawVolume(raw_info, use_memmap=True)
    exported_count = 0

    # Compression parameters
    write_params = []
    if ext in (".jpg", ".jpeg"):
        write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif ext == ".png":
        write_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    try:
        for s_idx in target_slices:
            if s_idx < 0 or s_idx >= raw_info.depth:
                continue

            slice_arr = vol.get_axial_slice(s_idx)
            base_stem = f"{raw_path.stem}_slice{s_idx:04d}"
            img_filename = f"{base_stem}{ext}"
            json_filename = f"{base_stem}.json"

            img_path = output_dir / img_filename
            out_json_path = output_dir / json_filename

            # Save 2D image
            if len(slice_arr.shape) == 2:
                cv2.imwrite(str(img_path), slice_arr, write_params)
            elif len(slice_arr.shape) == 3 and slice_arr.shape[2] == 3:
                bgr = cv2.cvtColor(slice_arr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_path), bgr, write_params)

            # Build standard 2D JSON
            raw_slice_shapes = slice_shapes_map.get(s_idx, [])
            clean_shapes = [clean_shape_to_standard_2d(s) for s in raw_slice_shapes]

            single_json_data = {
                "version": version,
                "flags": flags,
                "shapes": clean_shapes,
                "imagePath": img_filename,
                "imageData": None,
                "imageHeight": raw_info.height,
                "imageWidth": raw_info.width,
            }

            with open(out_json_path, "w", encoding="utf-8") as f_out:
                json.dump(single_json_data, f_out, indent=2, ensure_ascii=False)

            exported_count += 1

    finally:
        vol.close()

    print(f"[OK] {raw_path.name}: Exported {exported_count} slice(s) to {output_dir}")
    return exported_count


def decompose_raw_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    only_annotated: bool = True,
    img_format: str = "jpg",
) -> Tuple[int, Path]:
    """Decompose all (.raw, .json) pairs in input_dir into output_dir.

    If output_dir is not provided, defaults to:
    input_dir.parent / f"{input_dir.name}_slices" (sibling directory at the same level!).
    """
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_slices"

    # Match .json files with corresponding .raw files
    json_files = list(input_dir.glob("*.json"))
    pairs = []

    for jf in json_files:
        stem = jf.stem
        # Candidates for raw file
        raw_candidates = [
            jf.with_suffix(".raw"),
            input_dir / f"{stem}.raw",
        ]
        matched_raw = next((r for r in raw_candidates if r.exists()), None)
        if not matched_raw:
            # Match by prefix (e.g. pad5-3)
            for rf in input_dir.glob("*.raw"):
                if rf.stem == stem or rf.stem.startswith(stem[:20]):
                    matched_raw = rf
                    break

        if matched_raw and matched_raw.exists():
            pairs.append((matched_raw, jf))

    if not pairs:
        print(f"[!] No matching (.raw, .json) files found in: {input_dir}")
        return 0, output_dir

    print(f"[*] Found {len(pairs)} raw volume(s) in {input_dir}")
    print(f"[*] Output sibling directory: {output_dir}")
    print(f"[*] Image format: {img_format.upper()}\n")

    total_slices = 0
    for raw_path, json_path in pairs:
        cnt = decompose_single_raw(
            raw_path,
            json_path,
            output_dir,
            only_annotated=only_annotated,
            img_format=img_format,
        )
        total_slices += cnt

    print(f"\n[DONE] Successfully decomposed {total_slices} total slice(s) into: {output_dir}")
    return total_slices, output_dir


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
