import os
import sys


def setup_windows_dll_directories():
    """Ensure ONNX Runtime GPU can find CUDA/cuDNN DLLs in the Python environment on Windows."""
    if sys.platform != "win32":
        return
    import importlib.util

    for pkg in ("torch", "nvidia.cudnn", "nvidia.cublas"):
        try:
            spec = importlib.util.find_spec(pkg)
            if spec and spec.origin:
                pkg_dir = os.path.dirname(spec.origin)
                for d in (
                    os.path.join(pkg_dir, "lib"),
                    os.path.join(pkg_dir, "bin"),
                    pkg_dir,
                ):
                    if os.path.isdir(d):
                        if hasattr(os, "add_dll_directory"):
                            try:
                                os.add_dll_directory(d)
                            except OSError:
                                pass
                        if d not in os.environ.get("PATH", ""):
                            os.environ["PATH"] = (
                                d + os.pathsep + os.environ.get("PATH", "")
                            )
        except Exception:
            pass


setup_windows_dll_directories()

from .app_info import __appdescription__, __appname__, __version__

from anylabeling.views.common.checks import run_checks as checks
from anylabeling.views.common.converter import (
    SUPPORTED_TASKS,
    run_conversion as convert,
    list_supported_tasks,
)

__all__ = (
    "__version__",
    "__appname__",
    "__appdescription__",
    "checks",
    "convert",
    "list_supported_tasks",
    "SUPPORTED_TASKS",
)
