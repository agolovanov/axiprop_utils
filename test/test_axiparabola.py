"""Tests for axiparabola optics."""

from unittest.mock import patch

import numpy as np
import pint
import pytest

from axiprop_utils import (
    AxiparabolaPolynomial,
    ConstantIntensityAxiparabola,
    ParabolicMirror,
    mirror_axiparabola,
    mirror_parabolic,
)
from axiprop_utils.axiparabola import add_constant_velocity_phase
from axiprop_utils.optical_elements import AxisymmetricMirror
from axiprop_utils.pulse import create_gaussian_pulse

ureg = pint.get_application_registry()


def test_polynomial_coefficients_and_beam_radius():
    """Physical-radius coefficients reproduce the normalized theory surface."""
    f0, d0, kappa = 50 * ureg.cm, 10 * ureg.mm, -8 * ureg.mm
    radius, beam_radius = 30 * ureg.mm, 20 * ureg.mm
    mirror = AxiparabolaPolynomial(f0, d0, kappa, radius, beam_radius)
    r = np.linspace(0, 30, 31) * ureg.mm
    rho = r / beam_radius
    expected = beam_radius**2 / f0 * (rho**2 / 4 - d0 / f0 * rho**4 / 8 - kappa / f0 * rho**6 / 12)
    np.testing.assert_allclose(mirror.sag(r).m_as('m'), expected.m_as('m'), rtol=1e-14)
    np.testing.assert_allclose(
        mirror.sag(r).m_as('m'), (mirror.b2 * r**2 + mirror.b4 * r**4 + mirror.b6 * r**6).m_as('m')
    )
    assert mirror.b2.check('1/[length]')
    assert mirror.b4.check('1/[length]**3')
    assert mirror.b6.check('1/[length]**5')
    default = AxiparabolaPolynomial(f0, d0, kappa, radius)
    explicit = AxiparabolaPolynomial(f0, d0, kappa, radius, radius)
    np.testing.assert_array_equal(default.sag(r).magnitude, explicit.sag(r).magnitude)


@pytest.mark.parametrize('r_mm', [0, 10, [], [20, 0, 10, 10], [[0, 5], [10, 20]]])
@pytest.mark.parametrize(('d0_mm', 'kappa_mm'), [(10, 0), (-10, 5), (10, -10), (0, 10), (0, 0)])
def test_polynomial_geometry(r_mm, d0_mm, kappa_mm):
    """Exact geometry agrees with reflection and the base-class identities."""
    mirror = AxiparabolaPolynomial(50 * ureg.cm, d0_mm * ureg.mm, kappa_mm * ureg.mm, 30 * ureg.mm)
    r = np.asarray(r_mm) * ureg.mm
    sag, focal, path = mirror.sag(r), mirror.focal_distance(r), mirror.optical_path(r)
    slope = mirror.sag_der(r.to('cm'))
    derivative = mirror.focal_distance_der(r)
    assert sag.shape == focal.shape == path.shape == slope.shape == derivative.shape == r.shape
    assert isinstance(derivative, np.ndarray)
    np.testing.assert_allclose(focal.m_as('m'), AxisymmetricMirror.focal_distance(mirror, r).m_as('m'), rtol=1e-14)
    np.testing.assert_allclose(path.m_as('m'), AxisymmetricMirror.optical_path(mirror, r).m_as('m'), rtol=1e-14)
    np.testing.assert_allclose(path.m_as('m'), (np.hypot(r, focal - sag) - sag).m_as('m'), rtol=1e-14)
    # Specular reflection gives the outgoing radial/axial direction ratio.
    np.testing.assert_allclose((r / (focal - sag)).m_as(''), 2 * slope / (1 - slope**2), rtol=1e-14)
    np.testing.assert_allclose(mirror.focal_distance(0 * ureg.mm).m_as('cm'), 50)


@pytest.mark.parametrize(('d0_mm', 'kappa_mm'), [(10, -10), (-10, 8), (0, 10)])
def test_polynomial_focal_derivative_and_velocity(d0_mm, kappa_mm):
    """Analytic derivatives resolve a turning point and support exact delays."""
    mirror = AxiparabolaPolynomial(50 * ureg.cm, d0_mm * ureg.mm, kappa_mm * ureg.mm, 30 * ureg.mm)
    r = np.array([1, 10, 20, 29]) * ureg.mm
    dr = 1e-3 * ureg.mm
    numerical = (mirror.focal_distance(r + dr) - mirror.focal_distance(r - dr)) / (2 * dr)
    # Centered differences include O(dr**2) truncation and metre-scale cancellation.
    np.testing.assert_allclose(mirror.focal_distance_der(r), numerical.m_as(''), rtol=2e-7, atol=1e-10)
    for dv in [-2, 0, 0.1]:
        delay = mirror.constant_velocity_delay(r, dv)
        arrival = mirror.optical_path(r) + ureg.speed_of_light * delay
        expected = mirror.f0 + (mirror.focal_distance(r) - mirror.f0) / (1 + dv)
        np.testing.assert_allclose(arrival.m_as('m'), expected.m_as('m'), rtol=1e-14)
        np.testing.assert_allclose(
            mirror.velocity_on_axis(r, tau_delay_der=mirror.constant_velocity_delay_der(r, dv)),
            1 + dv,
            rtol=1e-14,
        )
    if d0_mm == 10 and kappa_mm == -10:
        turning_radius = mirror.beam_radius / np.sqrt(2)
        # Cancellation at the analytic turning radius leaves floating-point roundoff.
        np.testing.assert_allclose(mirror.focal_distance_der(turning_radius), 0, atol=5e-16)
        assert mirror.focal_distance_der(20 * ureg.mm) > 0
        assert mirror.focal_distance_der(29 * ureg.mm) < 0


def test_polynomial_parabolic_limit():
    """Zero nominal depth and curvature recover a parabola and equal paths."""
    mirror = AxiparabolaPolynomial(50 * ureg.cm, 0 * ureg.mm, 0 * ureg.mm, 30 * ureg.mm)
    parabola = ParabolicMirror(mirror.f0)
    r = np.linspace(0, 30, 31) * ureg.mm
    for method in ('sag', 'focal_distance', 'optical_path'):
        np.testing.assert_allclose(getattr(mirror, method)(r).m_as('m'), getattr(parabola, method)(r).m_as('m'))
    np.testing.assert_array_equal(mirror.focal_distance_der(r), 0)
    np.testing.assert_allclose(
        mirror.spectral_multiplier(np.array([1e7]), r), parabola.spectral_multiplier(np.array([1e7]), r)
    )


def test_polynomial_parameters_are_read_only():
    """Input quantities and retrieved quantities cannot mutate mirror geometry."""
    f0 = 50 * ureg.cm
    mirror = AxiparabolaPolynomial(f0, 10 * ureg.mm, -5 * ureg.mm, 30 * ureg.mm)
    f0 *= 2
    assert mirror.f0.m_as('cm') == 50
    for name in ('f0', 'd0', 'kappa', 'radius', 'beam_radius', 'b2', 'b4', 'b6'):
        value = getattr(mirror, name)
        expected = value.copy()
        value *= 2
        assert getattr(mirror, name) == expected
        with pytest.raises(AttributeError):
            setattr(mirror, name, value)


@pytest.mark.parametrize(
    ('name', 'value', 'error'),
    [
        ('f0', 0 * ureg.m, ValueError),
        ('radius', -1 * ureg.mm, ValueError),
        ('beam_radius', 0 * ureg.mm, ValueError),
        ('beam_radius', 31 * ureg.mm, ValueError),
        ('d0', np.nan * ureg.mm, ValueError),
        ('kappa', np.inf * ureg.mm, ValueError),
        ('kappa', np.ones(2) * ureg.mm, ValueError),
        ('d0', 1 * ureg.s, pint.DimensionalityError),
        ('kappa', 1, TypeError),
    ],
)
def test_polynomial_invalid_parameters(name, value, error):
    """Parameters require finite scalar lengths and a valid aperture."""
    parameters = dict(f0=50 * ureg.cm, d0=10 * ureg.mm, kappa=-5 * ureg.mm, radius=30 * ureg.mm)
    parameters[name] = value
    with pytest.raises(error):
        AxiparabolaPolynomial(**parameters)


@pytest.mark.parametrize('method', ['sag', '_H_func', 'focal_distance', 'focal_distance_der', 'optical_path'])
@pytest.mark.parametrize('r_mm', [-1, 31, np.nan, np.inf])
def test_polynomial_invalid_radii(method, r_mm):
    """Geometry evaluations reject radii outside the finite mirror domain."""
    mirror = AxiparabolaPolynomial(50 * ureg.cm, 10 * ureg.mm, 0 * ureg.mm, 30 * ureg.mm)
    with pytest.raises(ValueError):
        getattr(mirror, method)(r_mm * ureg.mm)


@pytest.mark.parametrize('depth', [-0.1, 0.1])
def test_intensity_on_axis_top_hat(depth):
    """Uniform illumination gives the predicted plateau for either focal direction."""
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, depth * ureg.m, 20 * ureg.mm, 10 * ureg.mm)
    z = np.linspace(0.11, 0.89, 79) * ureg.m
    intensity = 2 * ureg('TW/cm**2')
    k0 = 2 * np.pi / (800 * ureg.nm)
    actual = mirror.intensity_on_axis(z, intensity, k0=k0, nr=10001)
    plateau = (k0 * np.pi * mirror.beam_radius**2 * intensity / abs(mirror.focal_depth)).m_as('W/cm**2')
    low, high = sorted([0.5, 0.5 + 4 * depth])
    fraction = np.clip((np.minimum(z.m_as('m') + 0.005, high) - np.maximum(z.m_as('m') - 0.005, low)) / 0.01, 0, 1)
    np.testing.assert_allclose(actual.m_as('W/cm**2'), plateau * fraction, rtol=1e-12, atol=plateau * 1e-12)


def test_intensity_on_axis_sums_branches_and_conserves_power():
    """Overlapping focal branches add and preserve total power."""

    class FoldedMirror(ConstantIntensityAxiparabola):
        def focal_distance(self, r):
            return self.focal_length + self.focal_depth * np.abs(2 * (r / self.radius) ** 2 - 1)

    mirror = FoldedMirror(0.5 * ureg.m, 0.1 * ureg.m, 20 * ureg.mm)
    z = np.linspace(0.48, 0.62, 29) * ureg.m
    k0 = 1e7 / ureg.m
    incident = 3 * ureg('W/cm**2')
    actual = mirror.intensity_on_axis(z, incident, k0=k0, nr=10001)
    edges = np.r_[0.4775, (z[:-1].magnitude + z[1:].magnitude) / 2, 0.6225]
    fraction = np.maximum(0, np.minimum(edges[1:], 0.6) - np.maximum(edges[:-1], 0.5)) / np.diff(edges)
    expected = k0 * np.pi * mirror.radius**2 * incident / mirror.focal_depth
    np.testing.assert_allclose(actual.m_as('W/cm**2'), expected.m_as('W/cm**2') * fraction, rtol=1e-12)
    recovered_power = np.sum(actual * np.diff(edges) * ureg.m / k0)
    np.testing.assert_allclose(recovered_power.m_as('W'), (np.pi * mirror.radius**2 * incident).m_as('W'))


def test_intensity_on_axis_fold_caustic():
    """A smooth fold reproduces the integrable square-root ray density."""

    class FoldedMirror(ConstantIntensityAxiparabola):
        def focal_distance(self, r):
            return self.focal_length + self.focal_depth * (2 * (r / self.radius) ** 2 - 1) ** 2

    mirror = FoldedMirror(0.5 * ureg.m, 0.1 * ureg.m, 20 * ureg.mm)
    z = np.linspace(0.51, 0.59, 81) * ureg.m
    incident = 1 * ureg('W/cm**2')
    k0 = 1e7 / ureg.m
    actual = mirror.intensity_on_axis(z, incident, k0=k0, nr=100001)
    power = np.pi * mirror.radius**2 * incident
    expected = k0 * power / (2 * np.sqrt(mirror.focal_depth * (z - mirror.focal_length)))
    # Linear deposition smooths a curved density; dz / (z - f0) <= 0.1.
    np.testing.assert_allclose(actual.m_as('W/cm**2'), expected.m_as('W/cm**2'), rtol=1e-3)


def test_intensity_on_axis_envelope_profile_and_aperture():
    """The radial peak profile and carrier come from the unmodified envelope."""
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 0.1 * ureg.m, 10 * ureg.mm)
    envelope = create_gaussian_pulse(
        800 * ureg.nm,
        1 * ureg.J,
        30 * ureg.fs,
        8 * ureg.mm,
        t_axis=np.linspace(-100, 100, 65) * ureg.fs,
        r_axis=np.linspace(0, 20, 1001) * ureg.mm,
    )
    original = envelope.Field.copy()
    z = np.linspace(0.502, 0.598, 49) * ureg.m
    actual = mirror.intensity_on_axis(z, envelope, nr=20001)
    # Linear deposition applies a triangular kernel to the exponential profile.
    conversion = (ureg.speed_of_light * ureg.vacuum_permittivity * (ureg.V / ureg.m) ** 2 / 2).m_as('W/m**2')
    incident = conversion * np.max(np.abs(envelope.Field[:, 0]) ** 2)
    rate = 2 * (mirror.radius / (8 * ureg.mm)).m_as('') ** 2 / mirror.focal_depth.m_as('m')
    dz = (z[1] - z[0]).m_as('m')
    mean_profile = np.exp(-rate * (z.m_as('m') - 0.5)) * (np.sinh(rate * dz / 2) / (rate * dz / 2)) ** 2
    expected = envelope.k0 * np.pi * mirror.radius.m_as('m') ** 2 / 0.1 * incident * mean_profile
    np.testing.assert_allclose(actual.m_as('W/m**2'), expected, rtol=3e-5)
    outside = mirror.intensity_on_axis(np.array([0.61, 0.62]) * ureg.m, envelope)
    np.testing.assert_array_equal(outside.magnitude, 0)
    np.testing.assert_array_equal(envelope.Field, original)


@pytest.mark.parametrize(
    ('z', 'incident', 'kwargs', 'message'),
    [
        (np.array([0.5]), 1, {'k0': 1e7 / ureg.m}, 'at least two'),
        (np.array([0.6, 0.5]), 1, {'k0': 1e7 / ureg.m}, 'strictly increasing'),
        (np.array([0.5, np.nan]), 1, {'k0': 1e7 / ureg.m}, 'finite'),
        (np.array([0.5, 0.51, 0.6]), 1, {'k0': 1e7 / ureg.m}, 'uniformly spaced'),
        (np.array([0.5, 0.6]), 1, {}, 'k0 is required'),
        (np.array([0.5, 0.6]), 1, {'k0': 1e7 / ureg.m, 'nr': None}, 'nr is required'),
        (np.array([0.5, 0.6]), 1, {'k0': -1 / ureg.m}, 'wavenumber'),
        (np.array([0.5, 0.6]), -1, {'k0': 1e7 / ureg.m}, 'non-negative'),
        (np.array([0.5, 0.6]), 1, {'k0': 1e7 / ureg.m, 'nr': 1}, 'nr'),
    ],
)
def test_intensity_on_axis_validation(z, incident, kwargs, message):
    """Ambiguous bin grids and invalid illumination should fail explicitly."""
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 0.1 * ureg.m, 20 * ureg.mm)
    with pytest.raises(ValueError, match=message):
        mirror.intensity_on_axis(z * ureg.m, incident * ureg('W/cm**2'), **({'nr': 10001} | kwargs))


def test_intensity_on_axis_rejects_collapsed_focal_line():
    """A parabolic limit cannot be predicted using the stationary-phase density."""
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 0 * ureg.m, 20 * ureg.mm)
    with pytest.raises(ValueError, match='diffraction'):
        mirror.intensity_on_axis(np.array([0.5, 0.6]) * ureg.m, 1 * ureg('W/cm**2'), k0=1e7 / ureg.m, nr=10001)


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


@pytest.mark.parametrize('method', ['series', 'oubrerie', None])
@pytest.mark.parametrize('compensate_elongation', [False, True])
def test_constant_velocity_phase_is_axiparabola_method(method, compensate_elongation):
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
    method_kwargs = {} if method is None else {'method': method}
    expected_method = 'oubrerie' if method is None else method

    expected, *expected_terms = mirror.add_constant_velocity_phase(
        envelope,
        0.01,
        method=expected_method,
        compensate_elongation=compensate_elongation,
        return_correction_terms=True,
    )
    with pytest.warns(DeprecationWarning, match='ConstantIntensityAxiparabola'):
        actual, *actual_terms = add_constant_velocity_phase(
            envelope,
            0.01,
            f0=focal_length,
            d0=focal_depth,
            R=radius,
            **method_kwargs,
            compensate_elongation=compensate_elongation,
            return_correction_terms=True,
        )

    np.testing.assert_allclose(actual.Field_ft, expected.Field_ft)
    for actual_term, expected_term in zip(actual_terms, expected_terms, strict=True):
        np.testing.assert_allclose(actual_term.m_as(expected_term.units), expected_term.magnitude)
    alpha, beta = mirror.calculate_constant_velocity_coefficients(0.01, method=expected_method)
    r = envelope.r * ureg.m
    np.testing.assert_allclose(actual_terms[0].m_as('s'), (alpha * r**2 + beta * r**4).m_as('s'))


@pytest.mark.parametrize('method', ['series', 'oubrerie'])
@pytest.mark.parametrize('dv', [-0.1, 0.0, 0.1])
def test_calculate_constant_velocity_coefficients(method, dv):
    """Velocity coefficients should be available without an envelope."""
    focal_length = 0.5 * ureg.m
    focal_depth = 10 * ureg.cm
    radius = 20 * ureg.mm
    c = ureg('speed_of_light')
    mirror = ConstantIntensityAxiparabola(focal_length, focal_depth, radius)

    alpha, beta = mirror.calculate_constant_velocity_coefficients(dv, method=method)

    expected_alpha = -dv * focal_depth / radius**2 / c
    expected_beta = focal_depth / (4 * focal_length**2 * radius**2 * c)
    if method == 'series':
        expected_alpha /= 1 + dv
    else:
        expected_beta *= 1 + 2 * dv
    np.testing.assert_allclose(alpha.m_as('s/m**2'), expected_alpha.m_as('s/m**2'))
    np.testing.assert_allclose(beta.m_as('s/m**4'), expected_beta.m_as('s/m**4'))


@pytest.mark.parametrize('dv', [-0.1, 0.0, 0.1])
@pytest.mark.parametrize('focal_depth', [-5 * ureg.cm, 10 * ureg.cm])
def test_constant_velocity_series_matches_ray_geometry(dv, focal_depth):
    """The series should cancel the velocity error through quadratic order."""
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, focal_depth, 20 * ureg.mm, beam_radius=15 * ureg.mm)
    r = np.array([0.5, 1.0, 2.0]) * ureg.mm
    alpha, beta = mirror.calculate_constant_velocity_coefficients(dv)
    derivative = 2 * alpha * r + 4 * beta * r**3
    velocity = mirror.velocity_on_axis(r, tau_delay_der=derivative)
    error = np.abs(velocity - (1 + dv))
    # The first omitted delay term is r**6, giving a velocity error of r**4.
    np.testing.assert_allclose(error[1:] / error[:-1], 16, rtol=0.01)
    np.testing.assert_allclose(velocity, 1 + dv, rtol=3e-7)

    old_alpha, old_beta = mirror.calculate_constant_velocity_coefficients(dv, method='oubrerie')
    old_velocity = mirror.velocity_on_axis(r, tau_delay_der=2 * old_alpha * r + 4 * old_beta * r**3)
    if dv != 0:
        assert np.max(error) < np.min(np.abs(old_velocity - (1 + dv)))


@pytest.mark.parametrize(
    ('dv', 'method', 'message'),
    [
        (0.01, 'unknown', 'method'),
        (np.nan, 'series', 'finite'),
        (np.inf, 'oubrerie', 'finite'),
        (-1.0, 'series', 'differ from -1'),
    ],
)
def test_constant_velocity_coefficients_reject_invalid_input(dv, method, message):
    """Invalid methods and undefined velocities should fail explicitly."""
    mirror = ConstantIntensityAxiparabola(0.5 * ureg.m, 10 * ureg.cm, 20 * ureg.mm)
    with pytest.raises(ValueError, match=message):
        mirror.calculate_constant_velocity_coefficients(dv, method=method)
