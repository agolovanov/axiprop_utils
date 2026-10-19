"""Module for axiparabola utilities."""

import typing
import warnings

import numpy as np
import pint

from .optical_elements import AxisymmetricMirror

ureg = pint.get_application_registry()

if typing.TYPE_CHECKING:
    from axiprop import ScalarFieldEnvelope
    from pint import Quantity


class AxiparabolaPolynomial(AxisymmetricMirror):
    """Axiparabola with an exact sixth-order polynomial sag.

    The surface is ``s(r) = b2*r**2 + b4*r**4 + b6*r**6``. With
    ``rho = r / beam_radius``, its shallow-mirror focal map is approximately
    ``f0 + d0*rho**2 + kappa*rho**4``. The geometry methods evaluate the
    exact polynomial surface rather than this approximate focal map.

    Parameters
    ----------
    f0 : pint.Quantity
        Positive nominal focal length, equal to the on-axis focal distance.
    d0 : pint.Quantity
        Signed nominal focal depth, with length units.
    kappa : pint.Quantity
        Signed focal-depth curvature coefficient, with length units.
    radius : pint.Quantity
        Positive mirror radius.
    beam_radius : pint.Quantity, optional
        Radius defining the nominal depth and curvature. Defaults to radius
        and cannot exceed it.

    Notes
    -----
    Parameters and coefficients are read-only. The dimensional coefficients
    multiply physical radius, unlike the coefficients multiplying normalized
    radius in the theory document:
    ``b2 = 1/(4*f0)``, ``b4 = -d0/(8*f0**2*beam_radius**2)``, and
    ``b6 = -kappa/(12*f0**2*beam_radius**4)``.

    Signed depths and curvatures can produce non-monotonic focal maps.
    A zero of H away from the axis represents a ray parallel to the axis;
    its focal distance and optical path are singular.
    """

    def __init__(
        self,
        f0: 'Quantity',
        d0: 'Quantity',
        kappa: 'Quantity',
        radius: 'Quantity',
        beam_radius: 'Quantity' = None,
    ):
        parameters = (f0, d0, kappa, radius, radius if beam_radius is None else beam_radius)
        if any(not isinstance(value, pint.Quantity) for value in parameters):
            raise TypeError('Mirror parameters must be pint.Quantity objects')
        if any(np.ndim(value.magnitude) != 0 for value in parameters):
            raise ValueError('Mirror parameters must be scalar')
        values_m = [value.m_as('m') for value in parameters]
        if not np.all(np.isfinite(values_m)):
            raise ValueError('Mirror parameters must be finite')
        if values_m[0] <= 0:
            raise ValueError('f0 must be positive')
        if values_m[3] <= 0:
            raise ValueError('radius must be positive')
        if values_m[4] <= 0:
            raise ValueError('beam_radius must be positive')
        if values_m[4] > values_m[3]:
            raise ValueError('beam_radius cannot exceed mirror radius')
        self._f0, self._d0, self._kappa, self._radius, self._beam_radius = (value.copy() for value in parameters)
        self._b2 = (1 / (4 * self._f0)).to('1/m')
        self._b4 = (-self._d0 / (8 * self._f0**2 * self._beam_radius**2)).to('1/m**3')
        self._b6 = (-self._kappa / (12 * self._f0**2 * self._beam_radius**4)).to('1/m**5')

    @property
    def f0(self) -> 'Quantity':
        """Nominal on-axis focal length."""
        return self._f0.copy()

    @property
    def d0(self) -> 'Quantity':
        """Signed nominal focal depth at the beam radius."""
        return self._d0.copy()

    @property
    def kappa(self) -> 'Quantity':
        """Focal-depth curvature coefficient at the beam radius."""
        return self._kappa.copy()

    @property
    def radius(self) -> 'Quantity':
        """Mirror radius."""
        return self._radius.copy()

    @property
    def beam_radius(self) -> 'Quantity':
        """Reference beam radius for the nominal depth and curvature."""
        return self._beam_radius.copy()

    @property
    def b2(self) -> 'Quantity':
        """Quadratic sag coefficient in inverse-length units."""
        return self._b2.copy()

    @property
    def b4(self) -> 'Quantity':
        """Quartic sag coefficient in inverse-length-cubed units."""
        return self._b4.copy()

    @property
    def b6(self) -> 'Quantity':
        """Sextic sag coefficient in inverse-length-to-the-fifth units."""
        return self._b6.copy()

    def _radial_coordinate(self, r: 'Quantity') -> 'Quantity':
        """Validate radial coordinates while preserving their shape."""
        if not isinstance(r, pint.Quantity):
            raise TypeError('r must be a pint.Quantity')
        r_m = np.asarray(r.m_as('m'))
        if not np.all(np.isfinite(r_m)):
            raise ValueError('r must contain only finite values')
        if np.any(r_m < 0):
            raise ValueError('r must be non-negative')
        if np.any(r_m > self._radius.m_as('m') * (1 + 1e-12)):
            raise ValueError('r cannot extend beyond the mirror radius')
        return r_m * ureg.m

    @typing.override
    def sag(self, r: 'Quantity') -> 'Quantity':
        r = self._radial_coordinate(r)
        return r**2 * (self._b2 + r**2 * (self._b4 + r**2 * self._b6))

    @typing.override
    def _H_func(self, r: 'Quantity') -> 'Quantity':
        r = self._radial_coordinate(r)
        return 2 * self._b2 + r**2 * (4 * self._b4 + 6 * self._b6 * r**2)

    @typing.override
    def focal_distance(self, r: 'Quantity') -> 'Quantity':
        r = self._radial_coordinate(r)
        return 1 / (2 * self._H_func(r)) - r**4 * (self._b4 + 2 * self._b6 * r**2)

    @typing.override
    def focal_distance_der(self, r: 'Quantity') -> np.ndarray:
        r = self._radial_coordinate(r)
        H = self._H_func(r)
        H_der = 8 * r * (self._b4 + 3 * self._b6 * r**2)
        return np.asarray((-H_der * (1 / H**2 + r**2) / 2).m_as(''))

    @typing.override
    def optical_path(self, r: 'Quantity') -> 'Quantity':
        r = self._radial_coordinate(r)
        return 1 / (2 * self._H_func(r)) + r**4 * (self._b4 + 2 * self._b6 * r**2)


class ConstantIntensityAxiparabola(AxisymmetricMirror):
    """Axiparabola designed to produce a constant on-axis intensity.

    It corresponds to a focal line of
    ```math
        f(r) = focal_length + focal_depth * (r / beam_radius)**2
    ```

    Parameters
    ----------
    focal_length : pint.Quantity
        Distance from the mirror vertex to the beginning of the focal line.
    focal_depth : pint.Quantity
        Length of the focal line.
    radius : pint.Quantity
        Mirror radius.
    beam_radius : pint.Quantity, optional
        Radius of the beam on the mirror corresponding to the desired focal depth. If not provided, the mirror radius is used.

    Notes
    -----
    Mirror parameters are read-only.
    """

    def __init__(
        self,
        focal_length: 'Quantity',
        focal_depth: 'Quantity',
        radius: 'Quantity',
        beam_radius: 'Quantity' = None,
    ):
        parameters = (focal_length, focal_depth, radius, radius if beam_radius is None else beam_radius)
        if any(np.ndim(value.magnitude) != 0 for value in parameters):
            raise ValueError('Mirror parameters must be scalar')
        self._focal_length = focal_length.copy()
        self._focal_depth = focal_depth.copy()
        self._radius = radius.copy()
        self._beam_radius = (radius if beam_radius is None else beam_radius).copy()
        self._sag_cache: tuple | None = None
        self._validate_parameters()

    @property
    def focal_length(self) -> 'Quantity':
        """Distance from the mirror vertex to the start of the focal line."""
        return self._focal_length.copy()

    @property
    def focal_depth(self) -> 'Quantity':
        """Length of the focal line."""
        return self._focal_depth.copy()

    @property
    def radius(self) -> 'Quantity':
        """Mirror radius."""
        return self._radius.copy()

    @property
    def beam_radius(self) -> 'Quantity':
        """Beam radius on the mirror corresponding to the desired focal depth."""
        return self._beam_radius.copy()

    def _validate_parameters(self):
        """Validate the mirror parameters."""
        focal_length_m = self.focal_length.m_as('m')
        focal_depth_m = self.focal_depth.m_as('m')
        radius_m = self.radius.m_as('m')
        beam_radius_m = self.beam_radius.m_as('m')

        if not np.all(np.isfinite([focal_length_m, focal_depth_m, radius_m, beam_radius_m])):
            raise ValueError('Mirror parameters must be finite')
        if focal_length_m <= 0:
            raise ValueError('focal_length must be positive')
        if radius_m <= 0:
            raise ValueError('radius must be positive')
        if beam_radius_m <= 0:
            raise ValueError('beam_radius must be positive')
        if beam_radius_m > radius_m:
            raise ValueError('beam_radius cannot exceed mirror radius')
        edge_focal_distance_m = focal_length_m + focal_depth_m * (radius_m / beam_radius_m) ** 2
        if edge_focal_distance_m <= 0:
            raise ValueError('focal_length + focal_depth * (radius / beam_radius) ** 2 must be positive')

    @typing.override
    def sag(self, r: 'Quantity') -> 'Quantity':
        from scipy.integrate import solve_ivp

        r_values = np.asarray(r.m_as('m'))
        r_m = np.atleast_1d(r_values)
        focal_length_m = self.focal_length.m_as('m')
        focal_depth_m = self.focal_depth.m_as('m')
        radius_m = self.radius.m_as('m')
        beam_radius_m = self.beam_radius.m_as('m')

        if r_m.ndim != 1:
            raise ValueError('r must be scalar or one-dimensional')
        if not np.all(np.isfinite(r_m)):
            raise ValueError('r must contain only finite values')
        if np.any(r_m < 0):
            raise ValueError('r must be non-negative')
        if np.any(np.diff(r_m) <= 0):
            raise ValueError('r must be strictly increasing')
        if r_m.size == 0:
            return np.empty(0) * ureg.m
        if np.max(r_m) > radius_m * (1 + 1e-12):
            raise ValueError('r cannot extend beyond the mirror radius')
        if np.max(r_m) == 0:
            return np.zeros_like(r_values) * ureg.m

        cache = self._sag_cache
        if cache is not None:
            cached_radius, cached_r, cached_sag, interpolant = cache
            if np.array_equal(r_m, cached_r):
                return cached_sag.copy().reshape(r_values.shape) * ureg.m
            if r_m[-1] <= cached_radius:
                sag_m = interpolant(r_m)[0]
                self._sag_cache = (cached_radius, r_m.copy(), sag_m.copy(), interpolant)
                return sag_m.reshape(r_values.shape) * ureg.m

        def sag_equation(radial_coordinate, sag):
            focal_distance = focal_length_m + focal_depth_m * (radial_coordinate / beam_radius_m) ** 2
            distance_to_focus = focal_distance - sag[0]
            denominator = np.hypot(radial_coordinate, distance_to_focus) + distance_to_focus
            return [radial_coordinate / denominator]

        solution = solve_ivp(
            sag_equation,
            (0, r_m[-1]),
            [0.0],
            t_eval=r_m,
            dense_output=True,
            method='DOP853',
            rtol=1e-13,
            atol=1e-16,
        )
        if not solution.success:
            raise RuntimeError(f'Failed to calculate axiparabola sag: {solution.message}')

        self._sag_cache = (float(r_m[-1]), r_m.copy(), solution.y[0].copy(), solution.sol)
        return solution.y[0].reshape(r_values.shape) * ureg.m

    @typing.override
    def _H_func(self, r: 'Quantity') -> 'Quantity':
        sag = self.sag(r)
        focal_distance = self.focal_distance(r)
        distance_to_focus = focal_distance - sag
        return 1 / (np.hypot(r, distance_to_focus) + distance_to_focus)

    @typing.override
    def focal_distance(self, r: 'Quantity') -> 'Quantity':
        return self.focal_length + self.focal_depth * (r / self.beam_radius) ** 2

    @typing.override
    def focal_distance_der(self, r: 'Quantity') -> np.ndarray:
        return np.asarray((2 * self.focal_depth * r / self.beam_radius**2).m_as(''))

    @typing.override
    def optical_path(self, r: 'Quantity') -> 'Quantity':
        sag = self.sag(r)
        return np.hypot(r, self.focal_distance(r) - sag) - sag

    def calculate_constant_velocity_coefficients(
        self, dv: float, *, method: str = 'series'
    ) -> tuple['Quantity', 'Quantity']:
        """Calculate the radial-delay coefficients for a focal velocity.

        The coefficients define the delay
        ``tau_delay = alpha * r**2 + beta * r**4``.

        Parameters
        ----------
        dv : float
            Velocity difference relative to the speed of light, ``v/c - 1``.
        method : {'series', 'oubrerie'}, optional
            Default ``'series'`` expands the geometrical arrival time through
            fourth order in radius, keeping the exact dependence on dv.
            ``'oubrerie'`` preserves the previous calculation from Eq. (21) of
            Oubrerie et al., J. Opt. 24, 045503 (2022).

        Returns
        -------
        alpha : pint.Quantity
            Quadratic radial-delay coefficient.
        beta : pint.Quantity
            Quartic radial-delay coefficient.

        Notes
        -----
        Write ``a = focal_depth / beam_radius**2`` and ``f0 = focal_length``.
        The optical path is ``p(r) = f0 + a*r**2 - a*r**4/(4*f0**2) + O(r**6)``.
        Requiring ``p(r) + c*tau_delay(r) = f0 + a*r**2/(1+dv)`` therefore gives
        ``alpha = -a*dv/(c*(1+dv))`` and ``beta = a/(4*c*f0**2)``.
        This is a local series about the axis, not an exact constant velocity
        over the entire focal line. It leaves errors of order r**4 in v/c.
        The series method is undefined for ``dv = -1`` (zero focal velocity).

        The Oubrerie approximation instead uses ``alpha = -a*dv/c`` and
        ``beta = a*(1+2*dv)/(4*c*f0**2)``. See
        https://doi.org/10.1088/2040-8986/ac57d2, Eqs. (18)-(21).
        """
        if method not in ('series', 'oubrerie'):
            raise ValueError("method must be 'series' or 'oubrerie'")
        if not np.isfinite(dv):
            raise ValueError('dv must be finite')
        if method == 'series' and dv == -1:
            raise ValueError('dv must differ from -1 for the series method')

        c = ureg('speed_of_light')
        alpha = -dv * self.focal_depth / self.beam_radius**2 / c
        beta0 = self.focal_depth / (4 * self.focal_length**2 * self.beam_radius**2 * c)
        if method == 'oubrerie':
            return alpha, beta0 * (1 + 2 * dv)
        return alpha / (1 + dv), beta0

    @typing.override
    def add_constant_velocity_phase(
        self,
        envelope: 'ScalarFieldEnvelope',
        dv: float,
        *,
        method: str = 'series',
        compensate_elongation: bool = False,
        return_correction_terms: bool = False,
    ) -> 'ScalarFieldEnvelope':
        """Add a phase that produces a constant focal velocity.

        Parameters
        ----------
        envelope : ScalarFieldEnvelope
            Envelope to which the phase correction is applied.
        dv : float
            Velocity difference relative to the speed of light, ``v/c - 1``.
        method : {'series', 'oubrerie', 'exact'}, optional
            Defaults to ``'series'``; see :meth:`calculate_constant_velocity_coefficients`.
            Use ``'exact'`` for the geometrical delay from
            :meth:`AxisymmetricMirror.constant_velocity_delay`.
        compensate_elongation : bool, optional
            Add a radial GDD term to compensate for pulse elongation.
            Supported only for ``'series'`` and ``'oubrerie'``.
        return_correction_terms : bool, optional
            Return the applied delay and, when requested, GDD.

        Returns
        -------
        ScalarFieldEnvelope
            Envelope with the constant-velocity phase correction. When
            ``return_correction_terms`` is true, the correction terms are also
            returned.
        """
        from . import apply_radial_delay, apply_radial_GDD

        if method == 'exact':
            if compensate_elongation:
                raise ValueError("compensate_elongation is not supported for method='exact'")
            return super().add_constant_velocity_phase(envelope, dv, return_correction_terms=return_correction_terms)

        c = ureg('speed_of_light')
        r = envelope.r * ureg.m

        alpha, beta = self.calculate_constant_velocity_coefficients(dv, method=method)
        beta0 = self.focal_depth / (4 * self.focal_length**2 * self.beam_radius**2 * c)
        tau_delay = alpha * r**2 + beta * r**4

        envelope = apply_radial_delay(envelope, tau_delay)

        if compensate_elongation:
            omega0 = envelope.omega0 / ureg.s
            gdd = (alpha**2 / 2 + 2 * alpha * beta * r**2 + 2 * beta**2 * r**4) / (beta0 * omega0)
            envelope = apply_radial_GDD(envelope, gdd)

        if return_correction_terms:
            if compensate_elongation:
                return envelope, tau_delay, gdd
            return envelope, tau_delay
        return envelope


@ureg.wraps(None, (ureg.m**-1, ureg.m, ureg.m, ureg.m, ureg.m, None))
def mirror_axiparabola(kz, r, f0, d0, R, N_cut=4):
    """Calculate the spectral multiplier produced by an axiparabola.

    .. deprecated::
        Use :class:`ConstantIntensityAxiparabola` instead.

    [Smartsev et al Opt. Lett. 44, 3414 (2019)]

    Parameters
    ----------
    kz : pint.Quantity of shape (N_kz,)
        the longitudinal wavevector axis
    r : pint.Quantity of shape (N_r,)
        the radial coordinate
    f0 : pint.Quantity
        the focal length of the axiparabola
    d0 : pint.Quantity
        the focal depth of the axiparabola
    R : pint.Quantity
        the radius of the axiparabola
    N_cut : int, optional
        Retained for backward compatibility and ignored.

    Returns
    -------
    np.ndarray of shape (N_kz, N_r)
        the phase term representing the reflection on the axiparabola
    """
    warnings.warn(
        'mirror_axiparabola() is deprecated; use ConstantIntensityAxiparabola instead',
        DeprecationWarning,
        stacklevel=2,
    )
    mirror = ConstantIntensityAxiparabola(f0 * ureg.m, d0 * ureg.m, R * ureg.m)
    return mirror.spectral_multiplier(kz / ureg.m, r * ureg.m)


def add_constant_velocity_phase(
    envelope: 'ScalarFieldEnvelope',
    dv: float,
    *,
    f0: 'Quantity',
    d0: 'Quantity',
    R: 'Quantity',
    method: str = 'oubrerie',
    compensate_elongation: bool = False,
    return_correction_terms: bool = False,
) -> 'ScalarFieldEnvelope':
    """Add a phase that produces a constant focal velocity.

    .. deprecated::
        Use :meth:`ConstantIntensityAxiparabola.add_constant_velocity_phase`
        instead.

    Parameters
    ----------
    envelope : ScalarFieldEnvelope
        The envelope to which the phase term will be added
    dv : float
        The difference of the velocity from the speed of light relative to the speed of light (v/c - 1)
    f0 : pint.Quantity
        The focal length of the axiparabola
    d0 : pint.Quantity
        The focal depth of the axiparabola
    R : pint.Quantity
        The radius of the axiparabola
    method : {'series', 'oubrerie', 'exact'}, optional
        Coefficient calculation. Defaults to ``'oubrerie'`` to preserve the
        deprecated function's behavior; the mirror methods default to ``'series'``.
    compensate_elongation : bool, optional
        Whether to compensate for the elongation of the pulse due to the delay term by adding a GDD term, by default False
        (unsupported for ``method='exact'``).
    return_correction_terms : bool, optional
        Whether to return the correction terms (tau_delay and gdd) along with the envelope, by default False

    Returns
    -------
    ScalarFieldEnvelope
        The envelope with the added phase term.
        If `return_correction_terms` is True, also returns either (envelope, tau_delay) or (envelope, tau_delay, gdd) depending on the value of `compensate_elongation`
    """
    warnings.warn(
        (
            'add_constant_velocity_phase() is deprecated; use '
            'ConstantIntensityAxiparabola.add_constant_velocity_phase() instead'
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    mirror = ConstantIntensityAxiparabola(f0, d0, R)
    return mirror.add_constant_velocity_phase(
        envelope,
        dv,
        method=method,
        compensate_elongation=compensate_elongation,
        return_correction_terms=return_correction_terms,
    )
