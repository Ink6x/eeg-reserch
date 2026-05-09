"""SHA-256 content-hash parquet cache for preprocessed EEG arrays.

The cache key encodes both the source data identity and the preprocessing
config, so any config change automatically invalidates stale entries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class ParquetCache:
    """File-based cache backed by parquet files.

    Cache entries are stored as ``<cache_dir>/<key>.parquet``.
    The key should be a SHA-256 hex digest produced by :func:`make_cache_key`.

    Parameters
    ----------
    cache_dir : Path
        Directory for parquet files. Created on first write if absent.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = Path(cache_dir)

    def get(self, key: str) -> np.ndarray | None:
        """Return cached array if present, else ``None``.

        Parameters
        ----------
        key : str
            SHA-256 hex digest.
        """
        path = self._path(key)
        if not path.exists():
            return None
        return pd.read_parquet(path).to_numpy(dtype=np.float32)

    def put(self, key: str, array: np.ndarray) -> Path:
        """Persist ``array`` and return its cache path.

        Parameters
        ----------
        key : str
            SHA-256 hex digest.
        array : np.ndarray
            2-D array ``(n_samples, n_features)`` to cache.

        Returns
        -------
        Path
            Path to the written parquet file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        df = pd.DataFrame(array.astype(np.float32))
        df.to_parquet(path, index=False)
        return path

    def invalidate(self, key: str) -> bool:
        """Delete a cache entry. Returns ``True`` if the file existed."""
        path = self._path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.parquet"


def make_cache_key(subject_id: int, series_id: int, config_dict: dict) -> str:
    """Produce a stable SHA-256 key from (subject, series, config).

    Parameters
    ----------
    subject_id : int
        1-based subject index.
    series_id : int
        1-based series index.
    config_dict : dict
        Preprocessing configuration (must be JSON-serialisable).

    Returns
    -------
    str
        64-character lowercase hex SHA-256 digest.
    """
    payload = json.dumps(
        {"subj": subject_id, "series": series_id, "cfg": config_dict},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
