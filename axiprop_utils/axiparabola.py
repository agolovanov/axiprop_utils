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
