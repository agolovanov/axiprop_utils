"""Persist Axiprop transform matrices while recreating backend runtime state.

Cache identity includes installed Python distribution versions.
"""

import hashlib
from importlib.metadata import distributions
import json
import logging
from pathlib import Path
import platform
import re
import tempfile

from axiprop.lib import PropagatorResampling
import h5py
import numpy as np

_CACHE_VERSION = 1
_LOGGER = logging.getLogger(__name__)


def _environment_versions():
    """Return an ordered, location-independent inventory of Python packages."""
    packages = sorted(
        {
            (re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower(), dist.version)
            for dist in distributions()
            if dist.metadata['Name']
        }
    )
    return {
        'python': platform.python_version(),
        'implementation': platform.python_implementation(),
        'packages': packages,
    }


def _array_signature(array):
    """Identify an array by exact dtype, shape, and contiguous bytes."""
    array = np.ascontiguousarray(array)
    return {
        'dtype': array.dtype.str,
        'shape': array.shape,
        'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _read_matrices(path, metadata, shapes, dtype):
    """Load compatible matrices, treating missing or invalid files as misses."""
    try:
        with h5py.File(path, 'r') as cache:
            if cache.attrs['metadata'] != metadata:
                raise ValueError('cache metadata does not match')
            matrices = []
            for name, shape in shapes.items():
                dataset = cache[name]
                # The NumPy backend leaves invTM real; other backends cast it.
                allowed_dtypes = (dtype,) if name == 'TM' else (dtype, np.dtype('float64'))
                if dataset.shape != shape or dataset.dtype not in allowed_dtypes:
                    raise ValueError(f'invalid {name} shape or dtype')
                matrices.append(dataset[...])
            return matrices
    except FileNotFoundError:
        return None
    except (OSError, KeyError, ValueError) as exc:
        _LOGGER.warning('Rebuilding propagator cache %s: %s', path, exc)
        return None


def _write_matrices(path, metadata, matrices):
    """Publish a complete HDF5 file atomically in the cache directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.prop_', suffix='.h5', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with h5py.File(temporary_path, 'w') as cache:
            cache.attrs['metadata'] = metadata
            for name, matrix in matrices.items():
                cache.create_dataset(name, data=matrix, fletcher32=True)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


class PropagatorResamplingCache(PropagatorResampling):
    """Reuse disk-cached resampling matrices with fresh backend buffers.

    Parameters
    ----------
    *args
        Positional arguments for Axiprop's ``PropagatorResampling``.
    cache_dir : str or pathlib.Path
        Directory holding ``prop_<sha256>.h5`` files. Created when needed.
    **kwargs
        Keyword arguments for Axiprop's ``PropagatorResampling``.

    Notes
    -----
    This adapter relies on Axiprop's ``init_TST`` setup convention. Cache hits
    still initialize grids and the backend but skip matrix construction and
    inversion. Concurrent writers publish complete files atomically, though
    they may independently compute the same matrices.
    """

    def __init__(self, *args, cache_dir: str | Path, **kwargs):
        """Initialize the cache location before Axiprop calls ``init_TST``."""
        self._cache_dir = Path(cache_dir)
        super().__init__(*args, **kwargs)

    def init_TST(self):
        """Load transform matrices or build and save them on a cache miss."""
        # Use resolved grids/defaults rather than the envelope or raw quantities.
        configuration = {
            'cache_version': _CACHE_VERSION,
            'class': 'axiprop.lib.PropagatorResampling',
            'environment': _environment_versions(),
            'backend': self.bcknd.name,
            'mode': int(self.mode),
            'dtype': np.dtype(self.dtype).str,
            'grids': {name: _array_signature(getattr(self, name)) for name in ('r', 'r_new', 'r_ext', 'kr', 'kz')},
        }
        metadata = json.dumps(configuration, sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(metadata.encode('utf-8')).hexdigest()
        path = self._cache_dir / f'prop_{digest}.h5'
        shapes = {'TM': (self.kr.size, self.Nr), 'invTM': (self.Nr_new, self.kr.size)}
        matrices = _read_matrices(path, metadata, shapes, np.dtype(self.dtype))
        if matrices is None:
            super().init_TST()
            _write_matrices(
                path,
                metadata,
                {name: self.bcknd.to_host(getattr(self, name)) for name in shapes},
            )
            return

        self.TM, self.invTM = (self.bcknd.to_device(matrix) for matrix in matrices)
        # Match Axiprop's runtime setup without recomputing its matrices.
        self.shape_trns = (self.Nr,)
        self.shape_trns_new = (self.Nr_new,)
        self.u_loc = self.bcknd.zeros(self.Nr, self.dtype)
        self.u_ht = self.bcknd.zeros(self.Nr, self.dtype)
        self.u_iht = self.bcknd.zeros(self.Nr_new, self.dtype)
        self.TST_matmul = self.bcknd.make_matmul(self.TM, self.u_loc, self.u_ht)
        self.iTST_matmul = self.bcknd.make_matmul(self.invTM, self.u_ht, self.u_iht)
