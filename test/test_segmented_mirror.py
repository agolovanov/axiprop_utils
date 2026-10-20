"""Tests for continuous mirrors with piecewise constant focal distances."""

import numpy as np
import pint
import pytest

from axiprop_utils import AxisymmetricMirror, ParabolicMirror, SegmentedParabolicMirror

ureg = pint.get_application_registry()


@pytest.fixture
def mirror():
    """Provide a mirror with both increasing and decreasing focal positions."""
    return SegmentedParabolicMirror(
        np.array([10, 20, 30]) * ureg.mm,
        np.array([50, 60, 45]) * ureg.cm,
    )


def test_segmented_focal_geometry(mirror):
    """Every annulus obeys the exact reflection geometry and optical path."""
    r = np.array([[0, 3, 9], [10, 15, 19], [20, 25, 30]]) * ureg.mm
    expected = np.broadcast_to(np.array([0.5, 0.6, 0.45])[:, None], r.shape)
    np.testing.assert_allclose(mirror.focal_distance(r).m_as('m'), expected, rtol=1e-14)
    np.testing.assert_allclose(AxisymmetricMirror.focal_distance(mirror, r).m_as('m'), expected, rtol=1e-14)
    sag = mirror.sag(r)
    slope = mirror.sag_der(r)
    np.testing.assert_allclose((r / (mirror.focal_distance(r) - sag)).m_as(''), 2 * slope / (1 - slope**2), rtol=1e-14)
    np.testing.assert_allclose(
        mirror.optical_path(r).m_as('m'),
        (np.hypot(r, mirror.focal_distance(r) - sag) - sag).m_as('m'),
        rtol=1e-14,
    )
    # A paraboloidal annulus has equal path lengths to its own focus.
    path = mirror.optical_path(r).m_as('m')
    np.testing.assert_allclose(path, np.broadcast_to(path[:, :1], path.shape), rtol=1e-14)
    np.testing.assert_allclose(mirror.ray_radius(mirror.focal_distance(r), r).m_as('m'), 0, atol=1e-16)


def test_segmented_continuity_and_boundary_convention(mirror):
    """Sag joins continuously while each crease uses the outer segment."""
    dr = 1e-9 * ureg.mm
    for edge, focus in zip(mirror.radial_edges[:-1], mirror.focal_distances[1:]):
        # The probe displacement changes the sag by less than 1e-13 m.
        np.testing.assert_allclose(mirror.sag(edge - dr).m_as('m'), mirror.sag(edge).m_as('m'), atol=1e-13)
        np.testing.assert_allclose(mirror.sag(edge + dr).m_as('m'), mirror.sag(edge).m_as('m'), atol=1e-13)
        assert mirror.focal_distance(edge) == focus
        assert mirror.sag_der(edge) == pytest.approx(mirror.sag_der(edge + dr), rel=1e-9)
        assert not np.isclose(mirror.sag_der(edge - dr), mirror.sag_der(edge))
        assert np.isnan(mirror.focal_distance_der(edge))
    assert mirror.sag(0 * ureg.mm) == 0 * ureg.m
    np.testing.assert_array_equal(mirror.focal_distance_der(np.array([0, 5, 15, 25, 30]) * ureg.mm), 0)
    assert mirror.focal_distance(mirror.radius) == mirror.focal_distances[-1]


def test_segmented_sag_derivative(mirror):
    """Analytic slopes agree with numerical slopes away from the creases."""
    r = np.array([3, 15, 25]) * ureg.mm
    dr = 1e-4 * ureg.mm
    numerical = ((mirror.sag(r + dr) - mirror.sag(r - dr)) / (2 * dr)).m_as('')
    np.testing.assert_allclose(mirror.sag_der(r), numerical, rtol=1e-9)


@pytest.mark.parametrize('edges', [[30], [10, 20, 30]])
def test_segmented_parabolic_limit(edges):
    """Equal focal positions reproduce one paraboloid, including its phase."""
    mirror = SegmentedParabolicMirror(np.array(edges) * ureg.mm, np.full(len(edges), 0.5) * ureg.m)
    parabola = ParabolicMirror(50 * ureg.cm)
    r = np.linspace(0, 30, 31) * ureg.mm
    for method, unit in [('sag', 'm'), ('focal_distance', 'm'), ('optical_path', 'm')]:
        np.testing.assert_allclose(
            getattr(mirror, method)(r).m_as(unit), getattr(parabola, method)(r).m_as(unit), rtol=1e-14
        )
    np.testing.assert_allclose(mirror.sag_der(r), parabola.sag_der(r), rtol=1e-14)
    np.testing.assert_array_equal(mirror.focal_distance_der(r), 0)
    kz = np.array([7e6, 8e6]) / ureg.m
    np.testing.assert_allclose(mirror.spectral_multiplier(kz, r), parabola.spectral_multiplier(kz, r), atol=3e-12)


def test_segmented_phase_and_units(mirror):
    """The inherited phase preserves amplitude and uses the full continuous sag."""
    r = np.array([0, 5, 10, 15, 20, 25, 30]) * ureg.mm
    kz = np.array([7e6, 8e6]) / ureg.m
    actual = mirror.spectral_multiplier(kz, r)
    np.testing.assert_allclose(np.abs(actual), 1, atol=1e-15)
    np.testing.assert_allclose(actual, np.exp(-2j * kz.m_as('1/m')[:, None] * mirror.sag(r).m_as('m')), atol=1e-15)
    equivalent = SegmentedParabolicMirror(mirror.radial_edges.to('m'), mirror.focal_distances.to('mm'))
    np.testing.assert_allclose(equivalent.sag(r.to('cm')).m_as('m'), mirror.sag(r).m_as('m'), rtol=1e-14)
    assert mirror.sag(np.empty(0) * ureg.mm).shape == (0,)


def test_segmented_parameters_are_protected():
    """Changing constructor inputs or returned properties cannot change the mirror."""
    edges = np.array([10, 20]) * ureg.mm
    foci = np.array([50, 60]) * ureg.cm
    mirror = SegmentedParabolicMirror(edges, foci)
    edges[0] = 5 * ureg.mm
    foci[0] = 70 * ureg.cm
    mirror.radial_edges[0] = 3 * ureg.mm
    mirror.focal_distances[0] = 80 * ureg.cm
    np.testing.assert_array_equal(mirror.radial_edges.m_as('mm'), [10, 20])
    np.testing.assert_allclose(mirror.focal_distances.m_as('cm'), [50, 60])
    assert mirror.radius == 20 * ureg.mm
    with pytest.raises(AttributeError):
        mirror.radius = 1 * ureg.m


@pytest.mark.parametrize('edges_as_collection', [False, True])
@pytest.mark.parametrize('foci_as_collection', [False, True])
def test_segmented_quantity_collections(edges_as_collection, foci_as_collection):
    """Collections with mixed units give the same geometry as quantity arrays."""
    edges = [15 * ureg.mm, 3 * ureg.cm] if edges_as_collection else np.array([15, 30]) * ureg.mm
    foci = (50 * ureg.cm, 0.6 * ureg.m) if foci_as_collection else np.array([0.5, 0.6]) * ureg.m
    mirror = SegmentedParabolicMirror(edges, foci)
    reference = SegmentedParabolicMirror(np.array([15, 30]) * ureg.mm, np.array([0.5, 0.6]) * ureg.m)
    r = np.array([0, 10, 15, 25, 30]) * ureg.mm
    for method in ['sag', 'focal_distance', 'optical_path']:
        np.testing.assert_allclose(
            getattr(mirror, method)(r).m_as('m'), getattr(reference, method)(r).m_as('m'), rtol=1e-14
        )
    np.testing.assert_allclose(mirror.radial_edges.m_as('mm'), [15, 30])
    np.testing.assert_allclose(mirror.focal_distances.m_as('m'), [0.5, 0.6])


def test_segmented_scalar_quantities():
    """Scalar quantities describe a single segment."""
    mirror = SegmentedParabolicMirror(30 * ureg.mm, 50 * ureg.cm)
    r = np.linspace(0, 30, 11) * ureg.mm
    np.testing.assert_allclose(mirror.sag(r).m_as('m'), ParabolicMirror(0.5 * ureg.m).sag(r).m_as('m'), rtol=1e-14)
    assert mirror.radial_edges.shape == mirror.focal_distances.shape == (1,)


def test_segmented_collection_validation():
    """Reject unitless entries, incompatible units, and nonscalar entries."""
    with pytest.raises(TypeError, match='pint.Quantity'):
        SegmentedParabolicMirror([15 * ureg.mm, 30], [0.5 * ureg.m, 0.6 * ureg.m])
    with pytest.raises(pint.DimensionalityError):
        SegmentedParabolicMirror([15 * ureg.mm, 30 * ureg.s], [0.5 * ureg.m, 0.6 * ureg.m])
    with pytest.raises(ValueError, match='scalar'):
        SegmentedParabolicMirror([np.array([15, 30]) * ureg.mm], [0.5 * ureg.m])
    with pytest.raises(ValueError, match='at least one'):
        SegmentedParabolicMirror([], [])


def test_segmented_collection_inputs_are_protected():
    """Changing scalar quantities in an input collection cannot alter the mirror."""
    edge = ureg.Quantity(np.array(15.0), 'mm')
    focus = ureg.Quantity(np.array(0.5), 'm')
    mirror = SegmentedParabolicMirror([edge], [focus])
    edge.magnitude[...] = 30
    focus.magnitude[...] = 0.6
    assert mirror.radius == 15 * ureg.mm
    assert mirror.focal_distance(0 * ureg.mm) == 0.5 * ureg.m


@pytest.mark.parametrize(
    ('edges', 'foci', 'message'),
    [
        ([], [], 'at least one'),
        ([[1]], [1], 'one-dimensional'),
        ([1, 2], [1], 'one entry per segment'),
        ([1], [[1]], 'one entry per segment'),
        ([0, 1], [1, 2], 'positive'),
        ([1, 1, 2], [1, 2, 3], 'strictly increasing'),
        ([2, 1], [1, 2], 'strictly increasing'),
        ([-1], [1], 'positive'),
        ([np.inf], [1], 'finite'),
        ([1], [np.nan], 'finite'),
        ([1], [0], 'positive'),
        ([1], [-1], 'positive'),
    ],
)
def test_segmented_invalid_parameters(edges, foci, message):
    """Reject invalid shapes, edges, and focal positions during construction."""
    with pytest.raises(ValueError, match=message):
        SegmentedParabolicMirror(np.array(edges) * ureg.mm, np.array(foci) * ureg.m)


def test_segmented_requires_lengths():
    """All constructor parameters and query radii require length quantities."""
    with pytest.raises(TypeError, match='pint.Quantity'):
        SegmentedParabolicMirror([1], np.array([1]) * ureg.m)
    with pytest.raises(TypeError, match='pint.Quantity'):
        SegmentedParabolicMirror(np.array([1]) * ureg.mm, [1])
    with pytest.raises(pint.DimensionalityError):
        SegmentedParabolicMirror(np.array([1]) * ureg.s, np.array([1]) * ureg.m)


@pytest.mark.parametrize(
    'method', ['sag', '_H_func', 'sag_der', 'focal_distance', 'focal_distance_der', 'optical_path']
)
@pytest.mark.parametrize('r_mm', [-1, 31, np.nan, np.inf])
def test_segmented_invalid_radii(mirror, method, r_mm):
    """All geometry methods reject radii outside the physical mirror."""
    with pytest.raises(ValueError):
        getattr(mirror, method)(r_mm * ureg.mm)


def test_segmented_intensity_requires_diffraction(mirror):
    """Prevent grid-dependent caustic estimates for discrete focal positions."""
    with pytest.raises(NotImplementedError, match='diffraction'):
        mirror.intensity_on_axis(np.linspace(0.4, 0.7, 31) * ureg.m, 1 * ureg('W/cm**2'), k0=8e6 / ureg.m, nr=101)
