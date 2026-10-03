"""Check persisted transforms against freshly constructed propagators."""

from types import SimpleNamespace

from axiprop.backends import BACKEND_NP
from axiprop.lib import PropagatorResampling
import h5py
import numpy as np
import pint
import pytest

from axiprop_utils import create_resampling_propagator
from axiprop_utils import cache


@pytest.fixture
def envelope(monkeypatch):
    """Use small physical grids and a deterministic CPU backend."""
    monkeypatch.setenv('AXIPROP_BACKEND', 'NP')
    return SimpleNamespace(r=np.linspace(0, 1e-3, 24), k_freq=np.linspace(7e6, 9e6, 5))


def _create(envelope, cache_dir=None, radius=0.5, count=16):
    """Construct a resampling propagator through the public factory."""
    return create_resampling_propagator(
        envelope, radius * pint.get_application_registry().mm, count, cache_dir=cache_dir
    )


def _fail_inversion(*args, **kwargs):
    """Fail if a cache hit unexpectedly invokes matrix inversion."""
    pytest.fail('cache hit recomputed the matrix inverse')


def test_cache_hit_propagates_without_inversion(envelope, tmp_path, monkeypatch):
    """Loaded transforms reproduce propagation and own independent work buffers."""
    fresh = _create(envelope)
    first = _create(envelope, tmp_path / 'nested')
    monkeypatch.setattr(BACKEND_NP, 'inv_sqr_on_host', _fail_inversion)
    monkeypatch.setattr(BACKEND_NP, 'inv_on_host', _fail_inversion)
    cached = _create(envelope, tmp_path / 'nested')
    assert isinstance(cached, PropagatorResampling)
    assert cached.u_loc is not first.u_loc
    files = list((tmp_path / 'nested').glob('prop_*.h5'))
    assert len(files) == 1
    assert len(files[0].stem.removeprefix('prop_')) == 64
    rng = np.random.default_rng(42)
    field = rng.normal(size=(5, 24)) + 1j * rng.normal(size=(5, 24))
    for distance in (0.0, 0.01):
        np.testing.assert_allclose(
            cached.step(field.copy(), distance), fresh.step(field.copy(), distance), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize('change', ['radius', 'count', 'r', 'k_freq', 'environment', 'schema'])
def test_changed_configuration_builds_new_entry(envelope, tmp_path, monkeypatch, change):
    """Different grids and software metadata never reuse an old entry."""
    _create(envelope, tmp_path)
    kwargs = {}
    if change == 'radius':
        kwargs['radius'] = 0.6
    elif change == 'count':
        kwargs['count'] = 18
    elif change in ('r', 'k_freq'):
        setattr(envelope, change, getattr(envelope, change) * 1.001)
    elif change == 'environment':
        monkeypatch.setattr(cache, '_environment_versions', lambda: {'packages': [('changed', '1')]})
    else:
        monkeypatch.setattr(cache, '_CACHE_VERSION', 2)
    _create(envelope, tmp_path, **kwargs)
    assert len(list(tmp_path.glob('prop_*.h5'))) == 2


@pytest.mark.parametrize('damage', ['truncated', 'metadata', 'shape', 'dtype', 'missing'])
def test_invalid_cache_is_rebuilt(envelope, tmp_path, damage):
    """Unreadable and structurally invalid files are replaced by valid data."""
    first = _create(envelope, tmp_path)
    path = next(tmp_path.glob('prop_*.h5'))
    if damage == 'truncated':
        path.write_bytes(b'incomplete')
    else:
        with h5py.File(path, 'r+') as cache:
            if damage == 'metadata':
                cache.attrs['metadata'] = '{}'
            else:
                del cache['TM']
                if damage == 'shape':
                    cache['TM'] = np.zeros((2, 2), dtype=complex)
                elif damage == 'dtype':
                    cache['TM'] = np.zeros(first.TM.shape, dtype=np.int32)
    rebuilt = _create(envelope, tmp_path)
    np.testing.assert_array_equal(rebuilt.TM, first.TM)
    with h5py.File(path, 'r') as cache:
        np.testing.assert_array_equal(cache['TM'][...], first.TM)
    assert len(list(tmp_path.iterdir())) == 1


def test_equivalent_units_share_cache(envelope, tmp_path, monkeypatch):
    """Quantity representation and field amplitudes do not affect cache identity."""
    _create(envelope, tmp_path)
    envelope.Field_ft = np.ones((5, 24))
    monkeypatch.setattr(BACKEND_NP, 'inv_sqr_on_host', _fail_inversion)
    create_resampling_propagator(envelope, 500 * pint.get_application_registry().um, 16, cache_dir=tmp_path)


def test_uncached_factory_uses_original_class(envelope, tmp_path, monkeypatch):
    """Omitting cache_dir preserves construction without environment inspection."""
    monkeypatch.setattr(cache, '_environment_versions', _fail_inversion)
    monkeypatch.chdir(tmp_path)
    assert type(_create(envelope)) is PropagatorResampling
    assert not list(tmp_path.iterdir())
