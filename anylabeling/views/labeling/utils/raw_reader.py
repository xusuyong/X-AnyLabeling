"""Raw volume file reader, dimension parser, and slice generator.

Reference: ImageJ Raw.java (ij/plugin/Raw.java)
Supports automatic dimension detection from filename, parent directories, sibling files,
CT square slice factorization, and manual configuration override.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6 import QtGui


@dataclass
class RawFileInfo:
    file_path: str
    file_name: str
    file_length: int
    width: int
    height: int
    depth: int
    data_type: str  # 'uint8', 'uint16', 'float32', 'rgb8'
    offset: int = 0
    little_endian: bool = True
    pixel_width: float = 1.0
    pixel_height: float = 1.0
    pixel_depth: float = 1.0
    unit: str = "px"
    start_z: Optional[int] = None
    end_z: Optional[int] = None

    @property
    def bytes_per_voxel(self) -> int:
        if self.data_type == "uint8":
            return 1
        elif self.data_type == "uint16":
            return 2
        elif self.data_type == "float32":
            return 4
        elif self.data_type == "rgb8":
            return 3
        return 1

    @property
    def total_voxels(self) -> int:
        return self.width * self.height * self.depth

    @property
    def numpy_dtype(self) -> np.dtype:
        order = "<" if self.little_endian else ">"
        if self.data_type == "uint8":
            return np.dtype(np.uint8)
        elif self.data_type == "uint16":
            return np.dtype(f"{order}u2")
        elif self.data_type == "float32":
            return np.dtype(f"{order}f4")
        elif self.data_type == "rgb8":
            return np.dtype(np.uint8)
        return np.dtype(np.uint8)

    @property
    def slice_thickness(self) -> float:
        return self.pixel_depth

    def get_slice_z_coord(self, slice_idx: int) -> Optional[int]:
        """Get the physical Z index from filename if available, e.g. Z118-214."""
        if self.start_z is not None:
            return self.start_z + slice_idx
        return None


def match_3d_pattern(text: str, file_length: int) -> Optional[Tuple[int, int, int, str, int]]:
    """Match 3D dimension pattern (e.g. 300x300x97, 512-512-100) from text and test against file size."""
    if not text or not text.strip():
        return None

    # Pattern: (?<!\d)(\d{2,5})[-_xX](\d{2,5})[-_xX](\d{1,5})(?!\d)
    pattern = re.compile(r"(?<!\d)(\d{2,5})[-_xX](\d{2,5})[-_xX](\d{1,5})(?!\d)")
    for match in pattern.finditer(text):
        try:
            w = int(match.group(1))
            h = int(match.group(2))
            d = int(match.group(3))
            if w <= 0 or h <= 0 or d <= 0:
                continue
            voxels = w * h * d
            if file_length == voxels:
                return (w, h, d, "uint8", 0)
            elif file_length == voxels * 2:
                return (w, h, d, "uint16", 0)
            elif file_length == voxels * 4:
                return (w, h, d, "float32", 0)
            elif file_length == voxels * 3:
                return (w, h, d, "rgb8", 0)
            elif file_length > voxels and (file_length - voxels) <= 65536:
                return (w, h, d, "uint8", file_length - voxels)
            elif file_length > voxels * 2 and (file_length - voxels * 2) <= 65536:
                return (w, h, d, "uint16", file_length - voxels * 2)
        except (ValueError, OverflowError):
            continue
    return None


def ct_square_slice_factorization(file_length: int) -> Optional[Tuple[int, int, int, str, int]]:
    """CT Square Slice Factorization (W == H from 500 to 4096) from Raw.java."""
    candidates = []
    for w in range(500, 4097):
        w2 = w * w
        if file_length % w2 == 0:
            d = file_length // w2
            if 10 <= d <= 5000:
                candidates.append((w, w, d, "uint8", 0))
        if file_length % (w2 * 2) == 0:
            d = file_length // (w2 * 2)
            if 10 <= d <= 5000:
                candidates.append((w, w, d, "uint16", 0))

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        for c in candidates:
            if c[2] >= c[0] * 0.1 and c[2] <= c[0] * 1.5:
                return c
        return candidates[0]
    return None


def match_2d_pattern(text: str, file_length: int) -> Optional[Tuple[int, int, int, str, int]]:
    """Match 2D dimension pattern (e.g. 512x512, 1024-1024) from text and test against file size."""
    if not text or not text.strip():
        return None

    pattern = re.compile(r"(?<!\d)(\d{2,5})[-_xX](\d{2,5})(?!\d)")
    for match in pattern.finditer(text):
        try:
            w = int(match.group(1))
            h = int(match.group(2))
            if w <= 0 or h <= 0:
                continue
            pixels = w * h
            if file_length == pixels:
                return (w, h, 1, "uint8", 0)
            elif file_length == pixels * 2:
                return (w, h, 1, "uint16", 0)
            elif file_length == pixels * 4:
                return (w, h, 1, "float32", 0)
            elif file_length == pixels * 3:
                return (w, h, 1, "rgb8", 0)
            elif file_length % pixels == 0:
                d = file_length // pixels
                if 1 < d <= 65536:
                    return (w, h, d, "uint8", 0)
            elif file_length % (pixels * 2) == 0:
                d = file_length // (pixels * 2)
                if 1 < d <= 65536:
                    return (w, h, d, "uint16", 0)
        except (ValueError, OverflowError):
            continue
    return None


def parse_raw_file_info(file_path: str | Path) -> Optional[RawFileInfo]:
    """Automatically parse raw dimensions, bit depth, offset, and voxel size.

    Implements the exact logic of ImageJ's Raw.java:
    1. Try 3D dimensions in file name
    2. Try parent directories (up to 4 levels)
    3. Try sibling files in the same directory
    4. CT Square Slice Factorization (W == H)
    5. Try 2D dimensions in file name
    6. Extract voxel size and unit
    7. Check byte order
    8. Extract Z coordinate range if available
    """
    path = Path(file_path)
    if not path.is_file():
        return None

    file_length = path.stat().st_size
    if file_length <= 0:
        return None

    name = path.name
    cand = None

    # 1. Try 3D dimensions in file name
    cand = match_3d_pattern(name, file_length)

    # 2. Try parent directories (up to 4 levels)
    if cand is None:
        p = path.parent
        depth = 0
        while p and depth < 4 and p != p.parent:
            cand = match_3d_pattern(p.name, file_length)
            if cand is not None:
                break
            p = p.parent
            depth += 1

    # 3. Try sibling files in the same directory
    if cand is None:
        try:
            parent_dir = path.parent
            if parent_dir.is_dir():
                for sib in parent_dir.iterdir():
                    if sib.is_file() and sib.name != name:
                        cand = match_3d_pattern(sib.name, file_length)
                        if cand is not None:
                            break
        except OSError:
            pass

    # 4. CT Square Slice Factorization (W == H)
    if cand is None:
        cand = ct_square_slice_factorization(file_length)

    # 5. Try 2D dimensions
    if cand is None:
        cand = match_2d_pattern(name, file_length)

    if cand is None:
        return None

    w, h, d, data_type, offset = cand

    # Byte order
    name_lower = name.lower()
    little_endian = not ("be.raw" in name_lower or "big_endian" in name_lower)

    # Extract voxel size, e.g. -5um, _5um, -0.5um, -6mm
    unit = "px"
    v_size = 1.0
    p_unit = re.compile(
        r"[-_](\d+(?:\.\d+)?)\s*(um|µm|nm|mm|cm|m)\b", re.IGNORECASE
    )
    m_unit = p_unit.search(name)
    if not m_unit:
        m_unit = p_unit.search(str(path.resolve()))
    if m_unit:
        try:
            v_size = float(m_unit.group(1))
            u = m_unit.group(2).lower()
            if u in ("um", "µm"):
                unit = "µm"
            else:
                unit = u
        except ValueError:
            pass

    # Extract Z coordinate range if present, e.g. _Z118-214_
    start_z = None
    end_z = None
    m_z = re.search(r"[-_]Z(\d+)[-_](\d+)(?:[-_]|$)", name, re.IGNORECASE)
    if m_z:
        try:
            sz = int(m_z.group(1))
            ez = int(m_z.group(2))
            if ez >= sz and (ez - sz + 1) == d:
                start_z = sz
                end_z = ez
        except ValueError:
            pass

    return RawFileInfo(
        file_path=str(path.resolve()),
        file_name=name,
        file_length=file_length,
        width=w,
        height=h,
        depth=d,
        data_type=data_type,
        offset=offset,
        little_endian=little_endian,
        pixel_width=v_size,
        pixel_height=v_size,
        pixel_depth=v_size,
        unit=unit,
        start_z=start_z,
        end_z=end_z,
    )


class RawVolume:
    """Manages 3D volume loading, memory mapping, and slice extraction."""

    def __init__(self, file_info: RawFileInfo, use_memmap: bool = True):
        self.file_info = file_info
        self.use_memmap = use_memmap
        self.data: Optional[np.ndarray] = self._load_data()

    def _load_data(self) -> np.ndarray:
        fi = self.file_info
        dtype = fi.numpy_dtype
        shape = (fi.depth, fi.height, fi.width)
        if fi.data_type == "rgb8":
            shape = (fi.depth, fi.height, fi.width, 3)

        if self.use_memmap:
            try:
                memmap_arr = np.memmap(
                    fi.file_path,
                    dtype=dtype,
                    mode="r",
                    offset=fi.offset,
                    shape=shape,
                )
                return memmap_arr
            except Exception:
                pass

        with open(fi.file_path, "rb") as f:
            if fi.offset > 0:
                f.seek(fi.offset)
            count = fi.total_voxels * (3 if fi.data_type == "rgb8" else 1)
            arr = np.fromfile(f, dtype=dtype, count=count)
            return arr.reshape(shape)

    def close(self):
        """Release memmap and file handles (essential on Windows)."""
        if hasattr(self, "data") and isinstance(self.data, np.memmap):
            try:
                if hasattr(self.data, "_mmap") and self.data._mmap is not None:
                    self.data._mmap.close()
            except Exception:
                pass
        self.data = None

    @property
    def depth(self) -> int:
        return self.file_info.depth

    @property
    def height(self) -> int:
        return self.file_info.height

    @property
    def width(self) -> int:
        return self.file_info.width

    def get_axial_slice(self, slice_idx: int) -> np.ndarray:
        """Extract axial (XY plane at Z index) slice."""
        if self.data is None:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        idx = max(0, min(self.depth - 1, slice_idx))
        return np.array(self.data[idx], copy=True)

    def get_coronal_slice(self, y: int) -> np.ndarray:
        """Extract coronal (XZ plane at Y index) slice.
        Returns array of shape (depth, width) or (depth, width, 3).
        """
        if self.data is None:
            shape = (self.depth, self.width, 3) if self.file_info.data_type == "rgb8" else (self.depth, self.width)
            return np.zeros(shape, dtype=np.uint8)
        y_idx = max(0, min(self.height - 1, y))
        return np.ascontiguousarray(self.data[:, y_idx])

    def get_sagittal_slice(self, x: int) -> np.ndarray:
        """Extract sagittal (YZ plane at X index) slice oriented for side view (right of XY).
        Returns array of shape (height, depth) or (height, depth, 3), where:
        - vertical axis (rows) corresponds to Y (aligning with XY height),
        - horizontal axis (cols) corresponds to Z (depth).
        """
        if self.data is None:
            shape = (self.height, self.depth, 3) if self.file_info.data_type == "rgb8" else (self.height, self.depth)
            return np.zeros(shape, dtype=np.uint8)
        x_idx = max(0, min(self.width - 1, x))
        # Swap axes 0 and 1 so axis 0 is height (Y) and axis 1 is depth (Z)
        slice_yz = np.swapaxes(self.data[:, :, x_idx], 0, 1)
        return np.ascontiguousarray(slice_yz)

    def get_voxel_value(self, x: int, y: int, z: int):
        """Get voxel value at (x, y, z)."""
        if self.data is None:
            return 0
        z_idx = max(0, min(self.depth - 1, z))
        y_idx = max(0, min(self.height - 1, y))
        x_idx = max(0, min(self.width - 1, x))
        return self.data[z_idx, y_idx, x_idx]


def normalize_slice_to_uint8(slice_arr: np.ndarray) -> np.ndarray:
    """Normalize 8-bit, 16-bit or float slice array to 8-bit uint8 grayscale image."""
    if slice_arr.dtype == np.uint8 and len(slice_arr.shape) == 2:
        return slice_arr

    min_v, max_v = float(slice_arr.min()), float(slice_arr.max())
    if max_v <= min_v:
        max_v = min_v + 1.0

    norm = np.clip(
        (slice_arr.astype(np.float32) - min_v) / (max_v - min_v) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    return norm


def slice_to_png_bytes(slice_arr: np.ndarray) -> bytes:
    """Convert a 2D slice numpy array to PNG-encoded bytes."""
    norm = normalize_slice_to_uint8(slice_arr)
    success, encoded = cv2.imencode(".png", norm)
    if success:
        return encoded.tobytes()
    return b""


def slice_to_qimage(slice_arr: np.ndarray) -> QtGui.QImage:
    """Convert a 2D slice numpy array to QImage."""
    norm = normalize_slice_to_uint8(slice_arr)
    norm = np.ascontiguousarray(norm)
    if len(norm.shape) == 2:
        h, w = norm.shape
        return QtGui.QImage(
            norm.data, w, h, norm.strides[0], QtGui.QImage.Format.Format_Grayscale8
        ).copy()
    elif len(norm.shape) == 3 and norm.shape[2] == 3:
        h, w, _ = norm.shape
        return QtGui.QImage(
            norm.data, w, h, norm.strides[0], QtGui.QImage.Format.Format_RGB888
        ).copy()
    return QtGui.QImage()
