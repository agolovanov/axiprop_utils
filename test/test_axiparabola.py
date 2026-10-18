"""Tests for axiparabola optics."""

from unittest.mock import patch

import numpy as np
import pint
import pytest

from axiprop_utils import ConstantIntensityAxiparabola, ParabolicMirror, mirror_axiparabola, mirror_parabolic
from axiprop_utils.axiparabola import add_constant_velocity_phase
from axiprop_utils.optical_elements import AxisymmetricMirror
from axiprop_utils.pulse import create_gaussian_pulse

ureg = pint.get_application_registry()


def test_axiparabola_reduces_to_parabolic_mirror():
    """An axiparabola with zero focal depth should be parabolic."""
    wavelength = 800 * ureg.nm
    kz = np.linspace(0.95, 1.05, 5) * 2 * np.pi / wavelength
    radius = 10 * ureg.mm
    r = np.linspace(0, radius, 101)
    focal_length = 0.5 * ureg.m
    N_cut = 4

    mirror = ConstantIntensityAxiparabola(focal_length, 0 * ureg.m, radius)
    np.testing.assert_allclose(
        mirror.sag(r).m_as('m'),
        ParabolicMirror(focal_length).sag(r).m_as('m'),
        rtol=1e-11,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        mirror.sag_der(r),
        ParabolicMirror(focal_length).sag_der(r),
        rtol=1e-11,
        atol=1e-16,
    )

    with pytest.warns(DeprecationWarning, match='ConstantIntensityAxiparabola'):
        axiparabola = mirror_axiparabola(kz, r, focal_length, 0 * ureg.m, radius, N_cut=N_cut)
    with pytest.warns(DeprecationWarning, match='ParabolicMirror'):
        parabola = mirror_parabolic(kz, r, focal_length)

    np.testing.assert_allclose(axiparabola, parabola, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('depth', [-0.1, 0, 0.1])
@pytest.mark.parametrize('beam_radius', [None, 10 * ureg.mm])
def test_axiparabola_focal_distance(depth, beam_radius):
    """The surface should reproduce the prescribed focal mapping with a finite vertex H."""
    mirror = ConstantIntensityAxiparabola(50 * ureg.cm, depth * ureg.m, 20 * ureg.mm, beam_radius)
    r = np.array([0, 1, 5, 20]) * ureg.mm

    H = mirror._H_func(r).m_as('1/m')
    assert np.all(np.isfinite(H))
    np.testing.assert_allclose(H[0], 1.0, rtol=1e-14)
    expected = mirror.focal_length + mirror.focal_depth * (r / mirror.beam_radius) ** 2
    analytical = mirror.focal_distance(r)
    generic = AxisymmetricMirror.focal_distance(mirror, r)
    assert analytical.shape == r.shape
    np.testing.assert_allclose(analytical.m_as('m'), expected.m_as('m'), rtol=1e-13)
    np.testing.assert_allclose(analytical.m_as('m'), generic.m_as('m'), rtol=1e-13)


@pytest.mark.parametrize('depth', [-0.1, 0, 0.1])
@pytest.mark.parametrize('r_mm', [[], [0], [0, 1, 5, 20]])
def test_axiparabola_optical_path(depth, r_mm):
    """The optical path should include both incident and reflected segments."""
    mirror = ConstantIntensityAxiparabola(50 * ureg.cm, depth * ureg.m, 20 * ureg.mm)
    r = np.array(r_mm) * ureg.mm
    sag = mirror.sag(r)
    focal_distance = mirror.focal_distance(r)
    path = mirror.optical_path(r)
    reflected_length = np.sqrt(r**2 + (focal_distance - sag) ** 2)

    assert path.shape == r.shape
    np.testing.assert_allclose((path + sag).m_as('m'), reflected_length.m_as('m'), rtol=1e-13)
    np.testing.assert_allclose(path.m_as('m'), AxisymmetricMirror.optical_path(mirror, r).m_as('m'), rtol=1e-13)
    if r.size:
        np.testing.assert_allclose(path[0].m_as('m'), mirror.focal_length.m_as('m'), rtol=1e-14)
    if depth == 0:
        np.testing.assert_allclose(path.m_as('m'), mirror.focal_length.m_as('m'), rtol=1e-13)


@pytest.mark.parametrize('depth', [-0.1, 0.1])
def test_sag_der(depth):
    """Integrating the surface slope should reproduce the sag difference."""
    from scipy.integrate import simpson

    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, depth * ureg.m, 20 * ureg.mm)
    r = np.linspace(0, 20, 101) * ureg.mm
    slope = mirror.sag_der(r)
    sag = mirror.sag(r).m_as('m')
    assert slope[0] == 0
    np.testing.assert_allclose(simpson(slope, x=r.m_as('m')), sag[-1] - sag[0], rtol=1e-8)


def test_axiparabola_sag_starts_at_vertex():
    """The nonsingular ODE should be evaluated directly at the vertex."""
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 10 * ureg.cm, 20 * ureg.mm)
    r = np.array([0, 1, 5, 20]) * ureg.mm

    sag = mirror.sag(r)

    assert sag.units == ureg.m
    assert sag[0].m_as('m') == 0
    assert np.all(np.isfinite(sag.magnitude))
    assert np.all(np.diff(sag) > 0)


def test_sag_cache_reuses_equal_grids_and_protects_values():
    """Equivalent grids should reuse the solve despite input or output mutation."""
    from scipy.integrate import solve_ivp

    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 10 * ureg.cm, 20 * ureg.mm)
    r = np.linspace(0, 20, 101) * ureg.mm
    original_r = r.copy()
    with patch('scipy.integrate.solve_ivp', wraps=solve_ivp) as solver:
        sag = mirror.sag(r)
        expected = sag.copy()
        sag.magnitude[:] = -1
        r.magnitude[:] = -1
        repeated = mirror.sag(original_r.to('m'))
        np.testing.assert_array_equal(repeated.magnitude, expected.magnitude)
        repeated.magnitude[:] = -2
        np.testing.assert_array_equal(mirror.sag(original_r.copy()).magnitude, expected.magnitude)
        assert solver.call_count == 1


@pytest.mark.parametrize('depth', [-0.1, 0, 0.1])
def test_sag_cache_interpolates_contained_radii(depth):
    """A sparse initial grid should support accurate reuse at arbitrary radii."""
    from scipy.integrate import solve_ivp

    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, depth * ureg.m, 20 * ureg.mm)
    queries = [
        np.linspace(0, 20, 301) * ureg.mm,
        7 * ureg.mm,
        np.array([7]) * ureg.mm,
    ]
    expected = [
        ConstantIntensityAxiparabola(mirror.focal_length, mirror.focal_depth, mirror.radius).sag(r) for r in queries
    ]
    with patch('scipy.integrate.solve_ivp', wraps=solve_ivp) as solver:
        mirror.sag(np.array([1, 20]) * ureg.mm)
        for r, reference in zip(queries, expected, strict=True):
            actual = mirror.sag(r)
            assert np.shape(actual.magnitude) == np.shape(r.magnitude)
            assert actual.units == ureg.m
            np.testing.assert_allclose(actual.magnitude, reference.magnitude, rtol=1e-10, atol=2e-16)
        assert solver.call_count == 1


def test_sag_cache_extends_range():
    """Larger ranges should be solved once and then support previous grids."""
    from scipy.integrate import solve_ivp

    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 10 * ureg.cm, 20 * ureg.mm)
    r = np.linspace(0, 10, 51) * ureg.mm
    with patch('scipy.integrate.solve_ivp', wraps=solve_ivp) as solver:
        initial = mirror.sag(r)
        mirror.sag(np.linspace(0, 20, 101) * ureg.mm)
        assert solver.call_count == 2
        np.testing.assert_allclose(mirror.sag(r).magnitude, initial.magnitude, rtol=1e-10, atol=2e-16)
        assert solver.call_count == 2


@pytest.mark.parametrize('name', ['focal_length', 'focal_depth', 'radius', 'beam_radius'])
def test_axiparabola_parameters_are_read_only(name):
    """Inputs and retrieved quantities should not alter the stored parameters."""
    parameters = {
        'focal_length': ureg.Quantity(np.array(50.0), 'cm'),
        'focal_depth': ureg.Quantity(np.array(10.0), 'cm'),
        'radius': ureg.Quantity(np.array(20.0), 'mm'),
        'beam_radius': ureg.Quantity(np.array(20.0), 'mm'),
    }
    mirror = ConstantIntensityAxiparabola(**parameters)
    original = getattr(mirror, name)

    parameters[name].magnitude[...] *= 2
    retrieved = getattr(mirror, name)
    retrieved.ito('m')
    retrieved *= 2

    assert getattr(mirror, name) == original
    assert getattr(mirror, name).units == original.units
    with pytest.raises(AttributeError):
        setattr(mirror, name, retrieved)


def test_default_beam_radius_is_immutable():
    """The default beam radius should be independent of the supplied radius."""
    radius = ureg.Quantity(np.array(20.0), 'mm')
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 10 * ureg.cm, radius)
    radius.magnitude[...] = 10
    assert mirror.radius == 20 * ureg.mm
    assert mirror.beam_radius == 20 * ureg.mm


def test_axiparabola_parameters_must_be_scalar():
    """Array parameters should be rejected rather than silently converted."""
    with pytest.raises(ValueError, match='scalar'):
        ConstantIntensityAxiparabola(np.array([0.5]) * ureg.m, 10 * ureg.cm, 20 * ureg.mm)


@pytest.mark.parametrize(
    'r',
    [
        -1 * ureg.mm,
        np.nan * ureg.mm,
        21 * ureg.mm,
        np.array([2, 1]) * ureg.mm,
        np.array([1, 1]) * ureg.mm,
        np.array([[1, 2]]) * ureg.mm,
    ],
)
def test_sag_cache_preserves_validation(r):
    """Cached solutions should still reject invalid scalar and array queries."""
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 10 * ureg.cm, 20 * ureg.mm)
    mirror.sag(np.linspace(0, 20, 101) * ureg.mm)
    with pytest.raises(ValueError):
        mirror.sag(r)


@pytest.mark.parametrize(
    ('focal_length', 'focal_depth', 'radius', 'message'),
    [
        (0 * ureg.m, 10 * ureg.cm, 20 * ureg.mm, 'focal_length'),
        (0.5 * ureg.m, -0.5 * ureg.m, 20 * ureg.mm, 'focal_length \\+ focal_depth'),
        (0.5 * ureg.m, 10 * ureg.cm, 0 * ureg.mm, 'radius'),
    ],
)
def test_axiparabola_parameters_are_validated_at_creation(focal_length, focal_depth, radius, message):
    """Invalid mirror parameters should prevent construction."""
    with pytest.raises(ValueError, match=message):
        ConstantIntensityAxiparabola(focal_length, focal_depth, radius)


def test_axiparabola_requires_positive_focal_distance_at_mirror_edge():
    """A reduced beam radius should scale the focal-distance validation."""
    with pytest.raises(ValueError, match='focal_length \\+ focal_depth'):
        ConstantIntensityAxiparabola(
            focal_length=0.5 * ureg.m,
            focal_depth=-0.2 * ureg.m,
            radius=20 * ureg.mm,
            beam_radius=10 * ureg.mm,
        )


def test_constant_velocity_phase_is_axiparabola_method():
    """The compatibility function should delegate to the mirror method."""
    focal_length = 0.5 * ureg.m
    focal_depth = 10 * ureg.cm
    radius = 20 * ureg.mm
    duration = 30 * ureg.fs
    envelope = create_gaussian_pulse(
        800 * ureg.nm,
        1 * ureg.J,
        duration,
        5 * ureg.mm,
        t_axis=np.linspace(-4 * duration, 4 * duration, 64),
        r_axis=np.linspace(0, radius, 48),
    )
    mirror = ConstantIntensityAxiparabola(focal_length, focal_depth, radius)

    expected, expected_delay = mirror.add_constant_velocity_phase(
        envelope,
        0.01,
        return_correction_terms=True,
    )
    with pytest.warns(DeprecationWarning, match='ConstantIntensityAxiparabola'):
        actual, actual_delay = add_constant_velocity_phase(
            envelope,
            0.01,
            f0=focal_length,
            d0=focal_depth,
            R=radius,
            return_correction_terms=True,
        )

    np.testing.assert_allclose(actual.Field_ft, expected.Field_ft)
    np.testing.assert_allclose(actual_delay.m_as('s'), expected_delay.m_as('s'))


def test_calculate_constant_velocity_coefficients():
    """Velocity coefficients should be available without an envelope."""
    focal_length = 0.5 * ureg.m
    focal_depth = 10 * ureg.cm
    radius = 20 * ureg.mm
    dv = 0.01
    c = ureg('speed_of_light')
    mirror = ConstantIntensityAxiparabola(focal_length, focal_depth, radius)

    alpha, beta = mirror.calculate_constant_velocity_coefficients(dv)

    expected_alpha = -dv * focal_depth / radius**2 / c
    expected_beta = focal_depth / (4 * focal_length**2 * radius**2 * c) * (1 + 2 * dv)
    np.testing.assert_allclose(alpha.m_as('s/m**2'), expected_alpha.m_as('s/m**2'))
    np.testing.assert_allclose(beta.m_as('s/m**4'), expected_beta.m_as('s/m**4'))
