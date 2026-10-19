"""Tests for radial amplitude masks and their deprecated helpers."""

from types import SimpleNamespace

import numpy as np
import pint
import pytest

from axiprop_utils import AxisymmetricMask, Hole, SuperGaussianHole, hole, hole_supergaussian
from axiprop_utils.pulse import create_gaussian_pulse

ureg = pint.get_application_registry()


def test_hole_transmission():
    """A hard hole blocks the interior and transmits its boundary and exterior."""
    mask = Hole(1 * ureg.mm)
    assert isinstance(mask, AxisymmetricMask)
    np.testing.assert_array_equal(mask.transmission(np.array([0, 0.5, 1, 2]) * ureg.mm), [0, 0, 1, 1])
    assert mask.transmission(1 * ureg.mm).shape == ()
    assert mask.transmission(1 * ureg.mm) == 1


def test_smooth_hole_transmission():
    """Smoothing retains the original interior formula and unit boundary value."""
    r = np.array([[0, 0.5], [1, 2]]) * ureg.mm
    mask = Hole(0.1 * ureg.cm, l_smooth=500 * ureg.um, order_smooth=4)
    expected = np.array([[np.exp(-16), np.exp(-1)], [1, 1]])
    np.testing.assert_allclose(mask.transmission(r), expected, rtol=1e-14)
    np.testing.assert_allclose(mask.transmission(r.to('m')), expected, rtol=1e-14)


def test_supergaussian_hole_transmission():
    """The supergaussian blocks the axis and approaches unit transmission."""
    r = np.array([0, 0.5, 1, 2]) * ureg.mm
    mask = SuperGaussianHole(1000 * ureg.um, order=4)
    expected = 1 - np.exp(-(np.array([0, 0.5, 1, 2]) ** 4))
    np.testing.assert_allclose(mask.transmission(r), expected, rtol=1e-14)
    np.testing.assert_allclose(mask.transmission(r.to('cm')), expected, rtol=1e-14)


@pytest.mark.parametrize(
    'factory',
    [
        lambda radius: Hole(radius),
        lambda width: Hole(1 * ureg.mm, l_smooth=width),
        lambda radius: SuperGaussianHole(radius, order=4),
    ],
)
def test_mask_constructor_requires_lengths(factory):
    """Mask lengths must carry explicit, dimensionally correct units."""
    with pytest.raises(TypeError, match='pint.Quantity'):
        factory(1.0)
    with pytest.raises(pint.DimensionalityError):
        factory(1 * ureg.fs)


@pytest.mark.parametrize('mask', [Hole(1 * ureg.mm), SuperGaussianHole(1 * ureg.mm, order=4)])
def test_transmission_requires_lengths(mask):
    """Radial coordinates must also carry explicit length units."""
    with pytest.raises(TypeError, match='pint.Quantity'):
        mask.transmission(np.array([0, 1]))
    with pytest.raises(pint.DimensionalityError):
        mask.transmission(np.array([0, 1]) * ureg.fs)


@pytest.mark.parametrize(
    'mask',
    [
        Hole(0.5 * ureg.mm),
        Hole(0.5 * ureg.mm, l_smooth=0.2 * ureg.mm),
        SuperGaussianHole(0.5 * ureg.mm, order=4),
    ],
)
def test_mask_apply(mask):
    """Masks attenuate all frequencies and preserve the original envelope."""
    envelope = create_gaussian_pulse(
        800 * ureg.nm,
        1 * ureg.J,
        30 * ureg.fs,
        1 * ureg.mm,
        t_axis=np.linspace(-120, 120, 64) * ureg.fs,
        r_axis=np.linspace(0, 3, 48) * ureg.mm,
    )
    field_before = envelope.Field.copy()
    spectrum_before = envelope.Field_ft.copy()
    transmitted = mask.apply(envelope)
    transmission = mask.transmission(envelope.r * ureg.m)

    assert transmitted is not envelope
    np.testing.assert_array_equal(envelope.Field, field_before)
    np.testing.assert_array_equal(envelope.Field_ft, spectrum_before)
    np.testing.assert_allclose(transmitted.Field_ft, spectrum_before * transmission, rtol=1e-14)
    # Axiprop's complex64 inverse FFT introduces rounding, especially in the weak pulse tails.
    np.testing.assert_allclose(transmitted.Field, field_before * transmission, rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(transmitted.r, envelope.r)
    np.testing.assert_array_equal(transmitted.t, envelope.t)
    assert 0 < transmitted.Energy_ft < envelope.Energy_ft


@pytest.mark.parametrize('mask', [Hole(1 * ureg.mm), SuperGaussianHole(1 * ureg.mm, order=4)])
def test_mask_apply_rejects_3d(mask):
    """Cartesian envelopes are explicitly unsupported for now."""
    envelope = SimpleNamespace(Field=np.zeros((4, 5, 6)))
    with pytest.raises(NotImplementedError, match='3D envelopes'):
        mask.apply(envelope)


@pytest.mark.parametrize('with_units', [False, True])
@pytest.mark.parametrize('smooth', [False, True])
def test_deprecated_hole(with_units, smooth):
    """The legacy hole delegates to the class and accepts bare numeric lengths."""
    r = np.array([0, 0.5, 1, 2])
    radius = 1
    width = 0.5 if smooth else None
    expected = np.array([np.exp(-16), np.exp(-1), 1, 1]) if smooth else np.array([0, 0, 1, 1])
    if with_units:
        r = r * ureg.mm
        radius = radius * ureg.mm
        width = width * ureg.mm if smooth else None
    with pytest.warns(DeprecationWarning, match='use Hole instead'):
        actual = hole(r, radius, l_smooth=width, order_smooth=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


@pytest.mark.parametrize('with_units', [False, True])
def test_deprecated_supergaussian_hole(with_units):
    """The legacy supergaussian delegates without changing its amplitude mask."""
    r = np.array([0, 0.5, 1, 2])
    radius = 1
    expected = 1 - np.exp(-(r**4))
    if with_units:
        r = r * ureg.mm
        radius = radius * ureg.mm
    with pytest.warns(DeprecationWarning, match='use SuperGaussianHole instead'):
        actual = hole_supergaussian(r, radius, order=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
