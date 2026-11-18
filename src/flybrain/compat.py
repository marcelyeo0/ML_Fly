"""Small runtime fixes for third-party packages on Windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def patch_datamate_windows() -> bool:
    """Fix ``datamate`` (used by flyvis) writing HDF5 files on Windows.

    ``datamate.io._write_h5`` opens the file in ``"w"`` mode, fails on the missing
    dataset, then tries to ``unlink`` the file *while it is still open*. POSIX allows
    that, Windows raises ``PermissionError: [WinError 32]`` and flyvis cannot build
    its connectome cache. The replacement simply writes the file in one go.
    Returns ``True`` if the patch was applied.
    """
    try:
        import datamate.directory as dm_dir
        import datamate.io as dm_io
        import h5py
    except ImportError:
        return False

    def _write_h5(path: Path, val) -> None:
        val = np.asarray(val)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path.rmdir()
        with h5py.File(path, libver="latest", mode="w") as f:
            f["data"] = val
            f.swmr_mode = True

    dm_io._write_h5 = _write_h5
    dm_dir._write_h5 = _write_h5
    return True
