"""Raw volume file reader, dimension parser, and slice generator.

Reference: ImageJ Raw.java (ij/plugin/Raw.java)
Supports automatic dimension detection from filename, parent directories, sibling files,
CT square slice factorization, and manual configuration override.
"""

from __future__ import annotations

import io
import mmap
import threading
from collections import OrderedDict
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui


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
        # Guards data access vs. close() (side slices are read from a worker thread)
        self._lock = threading.RLock()
        self._closed = False
        self._close_pending = False
        self._warmed_up = False
        self._prefetched = False
        self._coronal_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._sagittal_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = 16
        self.data: Optional[np.ndarray] = self._load_data()
        self.display_window: Optional[Tuple[float, float]] = self._compute_display_window()

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

    def _compute_display_window(self) -> Optional[Tuple[float, float]]:
        """Sample the volume to compute a unified (vmin, vmax) display window for 16-bit/float data."""
        if self.data is None or self.file_info.data_type in ("uint8", "rgb8"):
            return None
        try:
            num_z = min(self.depth, 16)
            z_indices = np.linspace(0, self.depth - 1, num_z, dtype=int)
            step_y = max(1, self.height // 128)
            step_x = max(1, self.width // 128)
            samples = np.concatenate(
                [self.data[int(z), ::step_y, ::step_x].ravel() for z in z_indices]
            )
            if samples.size == 0:
                return None
            p_low, p_high = np.percentile(samples, [0.5, 99.5])
            vmin, vmax = float(p_low), float(p_high)
            if vmax <= vmin:
                vmin, vmax = float(samples.min()), float(samples.max())
            return (vmin, vmax)
        except Exception:
            return None

    def _release_resources_locked(self):
        self._coronal_cache.clear()
        self._sagittal_cache.clear()
        if hasattr(self, "data") and isinstance(self.data, np.memmap):
            try:
                if hasattr(self.data, "_mmap") and self.data._mmap is not None:
                    self.data._mmap.close()
            except Exception:
                pass
        self.data = None
        self._close_pending = False

    def close(self):
        """Release memmap and file handles without blocking UI if a worker is reading."""
        self._closed = True
        if self._lock.acquire(blocking=False):
            try:
                self._release_resources_locked()
            finally:
                self._lock.release()
        else:
            self._close_pending = True

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

    def _cache_get(self, cache, key):
        arr = cache.get(key)
        if arr is not None:
            cache.move_to_end(key)
        return arr

    def _cache_put(self, cache, key, arr):
        cache[key] = arr
        cache.move_to_end(key)
        while len(cache) > self._cache_size:
            cache.popitem(last=False)

    def get_coronal_slice(self, y: int) -> np.ndarray:
        """Extract coronal (XZ plane at Y index) slice (LRU cached).
        Returns array of shape (depth, width) or (depth, width, 3).
        """
        y_idx = max(0, min(self.height - 1, y))
        with self._lock:
            if self._closed or self.data is None:
                shape = (self.depth, self.width, 3) if self.file_info.data_type == "rgb8" else (self.depth, self.width)
                return np.zeros(shape, dtype=np.uint8)
            arr = self._cache_get(self._coronal_cache, y_idx)
            if arr is not None:
                return arr

        # On a cold memmap, reading rows directly via unbuffered seek+read avoids
        # triggering gigabytes of speculative OS page-cache readahead.
        if not self._warmed_up and isinstance(self.data, np.memmap):
            arr = self._read_coronal_direct(y_idx)
            if arr is not None:
                with self._lock:
                    if self._close_pending:
                        self._release_resources_locked()
                        return arr
                    if not self._closed:
                        self._cache_put(self._coronal_cache, y_idx, arr)
                return arr

        with self._lock:
            try:
                if self._closed or self.data is None:
                    shape = (self.depth, self.width, 3) if self.file_info.data_type == "rgb8" else (self.depth, self.width)
                    return np.zeros(shape, dtype=np.uint8)
                arr = self._cache_get(self._coronal_cache, y_idx)
                if arr is None:
                    arr = np.ascontiguousarray(self.data[:, y_idx])
                    self._cache_put(self._coronal_cache, y_idx, arr)
                return arr
            finally:
                if self._close_pending:
                    self._release_resources_locked()

    def _read_coronal_direct(self, y_idx: int) -> Optional[np.ndarray]:
        """Read a single coronal slice directly from disk (~2MB I/O for 2GB volume)."""
        fi = self.file_info
        bpp = fi.bytes_per_voxel
        dtype = fi.numpy_dtype
        is_rgb = fi.data_type == "rgb8"
        slice_bytes = fi.height * fi.width * bpp
        row_bytes = fi.width * bpp
        y_off = fi.offset + y_idx * row_bytes
        out_shape = (fi.depth, fi.width, 3) if is_rgb else (fi.depth, fi.width)
        row_shape = (fi.width, 3) if is_rgb else (fi.width,)
        out = np.empty(out_shape, dtype=dtype)
        try:
            with open(fi.file_path, "rb", buffering=0) as f:
                for z in range(fi.depth):
                    if self._closed:
                        return None
                    f.seek(z * slice_bytes + y_off)
                    buf = f.read(row_bytes)
                    if len(buf) < row_bytes:
                        return None
                    out[z] = np.frombuffer(buf, dtype=dtype).reshape(row_shape)
            return out
        except Exception:
            return None

    def get_sagittal_slice(self, x: int, should_abort=None) -> Optional[np.ndarray]:
        """Extract sagittal (YZ plane at X index) slice oriented for side view (LRU cached).
        Returns array of shape (height, depth) or (height, depth, 3), where:
        - vertical axis (rows) corresponds to Y (aligning with XY height),
        - horizontal axis (cols) corresponds to Z (depth).

        The read is done in Z chunks; the lock is only held per chunk so the volume
        can be closed promptly. If ``should_abort()`` turns true (or the volume is
        closed) midway, returns None.
        """
        x_idx = max(0, min(self.width - 1, x))
        is_rgb = self.file_info.data_type == "rgb8"
        with self._lock:
            if self._closed or self.data is None:
                shape = (self.height, self.depth, 3) if is_rgb else (self.height, self.depth)
                return np.zeros(shape, dtype=np.uint8)
            arr = self._cache_get(self._sagittal_cache, x_idx)
            if arr is not None:
                return arr
            out_shape = (self.height, self.depth, 3) if is_rgb else (self.height, self.depth)
            out = np.empty(out_shape, dtype=self.data.dtype)

        chunk = 16
        for z0 in range(0, self.depth, chunk):
            if should_abort is not None and should_abort():
                with self._lock:
                    if self._close_pending:
                        self._release_resources_locked()
                return None
            with self._lock:
                try:
                    if self._closed or self.data is None:
                        return None
                    z1 = min(self.depth, z0 + chunk)
                    out[:, z0:z1] = np.swapaxes(self.data[z0:z1, :, x_idx], 0, 1)
                finally:
                    if self._close_pending:
                        self._release_resources_locked()
        with self._lock:
            if not self._closed:
                self._cache_put(self._sagittal_cache, x_idx, out)
                self._warmed_up = True
        return out

    def prefetch(self):
        """Hint the OS to read the file in the background (non-blocking, once)."""
        if self._prefetched or self._closed:
            return
        self._prefetched = True
        try:
            mm = getattr(self.data, "_mmap", None)
            if mm is not None and hasattr(mm, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
                mm.madvise(mmap.MADV_WILLNEED)
        except Exception:
            pass

    def get_voxel_value(self, x: int, y: int, z: int):
        """Get voxel value at (x, y, z)."""
        if self.data is None:
            return 0
        z_idx = max(0, min(self.depth - 1, z))
        y_idx = max(0, min(self.height - 1, y))
        x_idx = max(0, min(self.width - 1, x))
        return self.data[z_idx, y_idx, x_idx]


def normalize_slice_to_uint8(
    slice_arr: np.ndarray,
    window: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Normalize 8-bit, 16-bit or float slice array to 8-bit uint8 image."""
    if slice_arr.dtype == np.uint8:
        return slice_arr
    if window is not None:
        vmin, vmax = window
        if vmax <= vmin:
            return np.zeros(slice_arr.shape, dtype=np.uint8)
        scale = 255.0 / (vmax - vmin)
        buf = slice_arr.astype(np.float32)
        buf -= float(vmin)
        buf *= scale
        np.clip(buf, 0.0, 255.0, out=buf)
        return buf.astype(np.uint8)
    return cv2.normalize(
        np.ascontiguousarray(slice_arr), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )


def slice_to_png_bytes(
    slice_arr: np.ndarray,
    window: Optional[Tuple[float, float]] = None,
) -> bytes:
    """Convert a 2D slice numpy array to PNG-encoded bytes."""
    norm = normalize_slice_to_uint8(slice_arr, window=window)
    success, encoded = cv2.imencode(".png", norm)
    if success:
        return encoded.tobytes()
    return b""


def slice_to_qimage(
    slice_arr: np.ndarray,
    window: Optional[Tuple[float, float]] = None,
) -> QtGui.QImage:
    """Convert a 2D slice numpy array to QImage."""
    norm = normalize_slice_to_uint8(slice_arr, window=window)
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



class OrthoSliceLoader(QtCore.QObject):
    """Loads coronal/sagittal slices off the UI thread (latest request wins)."""

    # volume, y, coronal
    coronal_ready = QtCore.pyqtSignal(object, int, object)
    # volume, x, sagittal
    sagittal_ready = QtCore.pyqtSignal(object, int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._pending = None
        self._running = False

    def request(self, volume: "RawVolume", x: int, y: int):
        with self._cond:
            self._pending = (volume, x, y)
            if not self._running:
                self._running = True
                threading.Thread(target=self._run, daemon=True).start()

    def cancel(self):
        with self._cond:
            self._pending = None

    def _has_newer_or_closed(self, volume: "RawVolume") -> bool:
        if volume._closed:
            return True
        with self._cond:
            return self._pending is not None

    def _run(self):
        while True:
            with self._cond:
                job = self._pending
                self._pending = None
                if job is None:
                    self._running = False
                    return
            volume, x, y = job
            if volume._closed:
                continue

            # 1. Coronal (XZ): fast direct read (~2MB I/O)
            try:
                cor = volume.get_coronal_slice(y)
            except Exception:
                cor = None
            if cor is not None and not volume._closed:
                self.coronal_ready.emit(volume, y, cor)

            if self._has_newer_or_closed(volume):
                continue

            # 2. Full-res Sagittal (YZ) with OS prefetch
            try:
                volume.prefetch()
                sag = volume.get_sagittal_slice(
                    x, should_abort=lambda: self._has_newer_or_closed(volume)
                )
            except Exception:
                sag = None
            if sag is not None and not volume._closed:
                self.sagittal_ready.emit(volume, x, sag)
