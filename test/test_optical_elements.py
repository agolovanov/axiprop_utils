"""Tests for optical element classes."""

import warnings

import numpy as np
import pint
import pytest
from pic_utils import PlasmaUnits

from axiprop_utils import ConstantIntensityAxiparabola, ParabolicMirror, mirror_parabolic, mirror_reflection
from axiprop_utils.optical_elements import AxisymmetricMirror
from axiprop_utils.pulse import create_gaussian_pulse

ureg = pint.get_application_registry()


@pytest.fixture(params=[None, -0.1, 0, 0.1])
def ray_mirror(request):
    """Provide a parabola and axiparabolas with both signs of focal depth."""
    if request.param is None:
        return ParabolicMirror(50 * ureg.cm)
    return ConstantIntensityAxiparabola(50 * ureg.cm, request.param * ureg.m, 20 * ureg.mm)


@pytest.mark.parametrize('r0_mm', [0, 10])
def test_ray_radius(ray_mirror, r0_mm):
    """Rays should start on the surface, focus, and continue across the axis."""
    r0 = r0_mm * ureg.mm
    sag = ray_mirror.sag(r0)
    focus = ray_mirror.focal_distance(r0)
    z = np.array([sag.m_as('cm'), focus.m_as('cm'), (2 * focus - sag).m_as('cm')]) * ureg.cm

    r = ray_mirror.ray_radius(z, r0)

    assert r.shape == z.shape
    np.testing.assert_allclose(r.m_as('mm'), [r0_mm, 0, -r0_mm], rtol=1e-13, atol=1e-13)
    scalar_r = ray_mirror.ray_radius(sag, r0)
    assert scalar_r.shape == ()
    np.testing.assert_allclose(scalar_r.m_as('mm'), r0_mm, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('z0_cm', [30, 70])
def test_ray_snapshot(ray_mirror, z0_cm):
    """Snapshots should preserve equal travel times and the reflection direction."""
    r0 = np.array([0, 1, 5, 20]) * ureg.mm
    z0 = z0_cm * ureg.cm
    sag = ray_mirror.sag(r0)

    z, r = ray_mirror.ray_snapshot(z0, r0)

    assert z.shape == r.shape == r0.shape
    np.testing.assert_allclose(z[0].m_as('cm'), z0_cm, rtol=1e-13)
    np.testing.assert_allclose(r[0].m_as('mm'), 0, atol=1e-13)
    traveled = np.hypot(z - sag, r - r0)
    np.testing.assert_allclose((traveled - sag).m_as('cm'), z0_cm, rtol=1e-13)
    slope = ray_mirror.sag_der(r0)
    np.testing.assert_allclose(((r - r0) / traveled).m_as(''), -2 * slope / (1 + slope**2), atol=1e-14)
    np.testing.assert_allclose(((z - sag) / traveled).m_as(''), (1 - slope**2) / (1 + slope**2), atol=1e-14)
    if z0_cm == 70:
        assert np.all(r[1:] < 0 * ureg.mm)
    else:
        assert np.all(r[1:] > 0 * ureg.mm)


def test_ray_snapshot_scalar(ray_mirror):
    """A scalar snapshot should agree with the ray's longitudinal parameterization."""
    r0 = 10 * ureg.mm
    z, r = ray_mirror.ray_snapshot(60 * ureg.cm, r0)

    assert z.shape == r.shape == ()
    np.testing.assert_allclose(r.m_as('mm'), ray_mirror.ray_radius(z, r0).m_as('mm'), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('threshold, expected', [(1, 9), (0.5, 17), (2, 5)])
@pytest.mark.parametrize('k0', [np.pi, np.pi / ureg.m])
def test_minimum_nr(threshold, expected, k0):
    """Match the analytic parabolic slope and satisfy the phase-step limit."""
    mirror = ParabolicMirror(0.5 * ureg.m)
    r = np.array([0, 2000]) * ureg.mm
    nr = mirror.minimum_nr(r, k0=k0, nyquist_threshold=threshold)
    assert nr == expected
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mirror.spectral_multiplier(np.array([np.pi]), np.linspace(r[0], r[-1], nr), k0=k0, nyquist_threshold=threshold)


def test_flat_minimum_nr():
    """A flat mirror needs only the endpoints for its phase correction."""

    class FlatMirror(AxisymmetricMirror):
        def sag(self, r):
            raise AssertionError('minimum_nr should use sag_der directly')

        def _H_func(self, r):
            return np.zeros(r.shape) / ureg.m

    assert FlatMirror().minimum_nr(np.linspace(0, 1, 3) * ureg.m, k0=1) == 2


@pytest.mark.parametrize('base_unit', ['wavelength', 'frequency', 'density'])
def test_minimum_nr_plasma_units(base_unit):
    """Equivalent PlasmaUnits inputs should give the same radial point count."""
    units = PlasmaUnits(800 * ureg.nm)
    mirror = ParabolicMirror(0.5 * ureg.m)
    r = np.array([0, 1]) * ureg.mm
    expected = mirror.minimum_nr(r, k0=units.wavenumber)

    assert mirror.minimum_nr(r, k0=getattr(units, base_unit)) == expected
    assert expected > 2


def test_minimum_nr_wavelength_requires_scalar():
    """Wavelength arrays should fail the scalar wavenumber validation."""
    mirror = ParabolicMirror(0.5 * ureg.m)
    r = np.array([0, 1]) * ureg.mm

    with pytest.raises(ValueError, match='positive, finite scalar wavenumber'):
        mirror.minimum_nr(r, k0=np.array([800, 900]) * ureg.nm)


@pytest.mark.parametrize('k0', [np.pi, np.pi / ureg.m])
def test_phase_step_warning(k0):
    """Detect aliased phase steps independently of the spectral extremes."""
    mirror = ParabolicMirror(0.5 * ureg.m)
    r = np.array([0, 2]) * ureg.m
    kz = np.array([0.01, 100])
    expected = mirror.spectral_multiplier(kz, r)
    with pytest.warns(UserWarning, match='reaches 4 \\* pi'):
        actual = mirror.spectral_multiplier(kz, r, k0=k0)
    np.testing.assert_array_equal(actual, expected)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mirror.spectral_multiplier(kz, r, k0=k0, nyquist_threshold=4)
        mirror.spectral_multiplier(kz, r[:1], k0=k0)


def test_parabolic_mirror_sag():
    """The parabolic mirror should evaluate its analytic sag with units."""
    focal_length = 0.5 * ureg.m
    r = np.linspace(0, 10, 11) * ureg.mm

    sag = ParabolicMirror(focal_length).sag(r)

    np.testing.assert_allclose(sag.m_as('m'), (r**2 / (4 * focal_length)).m_as('m'))
    np.testing.assert_allclose(ParabolicMirror(focal_length).sag_der(r), r.m_as('m') / (2 * focal_length.m_as('m')))


@pytest.mark.parametrize('r_mm', [[], [0], [0, 1, 10, 800]])
def test_parabolic_mirror_focal_distance(r_mm):
    """All parabolic rays should focus at the same distance, including at the vertex."""
    mirror = ParabolicMirror(50 * ureg.cm)
    r = np.array(r_mm) * ureg.mm

    np.testing.assert_allclose(mirror._H_func(r).m_as('1/m'), np.ones(r.shape))
    analytical = mirror.focal_distance(r)
    generic = AxisymmetricMirror.focal_distance(mirror, r)
    assert analytical.shape == r.shape
    np.testing.assert_allclose(analytical.m_as('m'), 0.5, rtol=1e-14)
    np.testing.assert_allclose(analytical.m_as('m'), generic.m_as('m'), rtol=1e-14)


@pytest.mark.parametrize('r_mm', [[], [0], [0, 1, 10, 800, 1200]])
def test_parabolic_mirror_optical_path(r_mm):
    """The incident sag correction should make all parabolic paths equal."""
    mirror = ParabolicMirror(50 * ureg.cm)
    r = np.array(r_mm) * ureg.mm
    sag = mirror.sag(r)
    geometric = np.hypot(r, mirror.focal_distance(r) - sag) - sag
    path = mirror.optical_path(r)

    assert path.shape == r.shape
    np.testing.assert_allclose(path.m_as('m'), 0.5, rtol=1e-14)
    np.testing.assert_allclose(path.m_as('m'), geometric.m_as('m'), rtol=1e-14)
    np.testing.assert_allclose(path.m_as('m'), AxisymmetricMirror.optical_path(mirror, r).m_as('m'), rtol=1e-14)


def test_mirror_parabolic_delegates_to_class():
    """The deprecated function should produce the class spectral multiplier."""
    focal_length = 0.5 * ureg.m
    wavelength = 800 * ureg.nm
    kz = np.linspace(0.95, 1.05, 5) * 2 * np.pi / wavelength
    r = np.linspace(0, 10, 11) * ureg.mm
    mirror = ParabolicMirror(focal_length)

    with pytest.warns(DeprecationWarning, match='ParabolicMirror'):
        actual = mirror_parabolic(kz, r, focal_length)

    expected = mirror.spectral_multiplier(kz, r)
    np.testing.assert_allclose(actual, expected)


def test_parabolic_mirror_apply_returns_new_envelope():
    """Applying a lossless mirror should preserve energy and its input."""
    wavelength = 800 * ureg.nm
    duration = 30 * ureg.fs
    radius = 1 * ureg.mm
    t_axis = np.linspace(-4 * duration, 4 * duration, 64)
    r_axis = np.linspace(0, 3 * radius, 48)
    envelope = create_gaussian_pulse(
        wavelength,
        1 * ureg.J,
        duration,
        radius,
        t_axis=t_axis,
        r_axis=r_axis,
    )
    field_ft_before = envelope.Field_ft.copy()
    mirror = ParabolicMirror(0.5 * ureg.m)

    reflected = mirror.apply(envelope)
    with pytest.warns(UserWarning, match='at k0'):
        reflected_strict = mirror.apply(envelope, nyquist_threshold=0.01)
    np.testing.assert_array_equal(reflected_strict.Field_ft, reflected.Field_ft)
    sag = mirror.sag(envelope.r * ureg.m)
    expected_multiplier = mirror_reflection(envelope.k_freq / ureg.m, sag)

    assert reflected is not envelope
    np.testing.assert_allclose(envelope.Field_ft, field_ft_before)
    np.testing.assert_allclose(reflected.Field_ft, envelope.Field_ft * expected_multiplier)
    np.testing.assert_allclose(np.abs(reflected.Field_ft), np.abs(envelope.Field_ft), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(reflected.Energy_ft, envelope.Energy_ft, rtol=1e-12)
