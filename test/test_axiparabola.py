"""Tests for axiparabola optics."""

import numpy as np
import pint
import pytest

from axiprop_utils import ConstantIntensityAxiparabola, ParabolicMirror, mirror_axiparabola, mirror_parabolic
from axiprop_utils.axiparabola import add_constant_velocity_phase
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


@pytest.mark.parametrize(
    'r',
    [
        np.array([0, 2, 1]) * ureg.mm,
        np.array([0, 1, 1]) * ureg.mm,
    ],
)
def test_axiparabola_requires_strictly_increasing_radius(r):
    """The radial grid should be ordered and contain no duplicates."""
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 10 * ureg.cm, 20 * ureg.mm)

    with pytest.raises(ValueError, match='strictly increasing'):
        mirror.sag(r)


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
