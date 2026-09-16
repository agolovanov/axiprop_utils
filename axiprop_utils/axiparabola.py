"""Module for axiparabola utilities."""

import typing
import warnings
from dataclasses import dataclass

import numpy as np
import pint

from .optical_elements import AxisymmetricMirror

ureg = pint.get_application_registry()

if typing.TYPE_CHECKING:
    from axiprop import ScalarFieldEnvelope
    from pint import Quantity


@dataclass(frozen=True)
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
    """

    focal_length: 'Quantity'
    focal_depth: 'Quantity'
    radius: 'Quantity'
    beam_radius: 'Quantity' = None

    def __post_init__(self):
        """Validate the mirror parameters."""
        focal_length_m = self.focal_length.m_as('m')
        focal_depth_m = self.focal_depth.m_as('m')
        radius_m = self.radius.m_as('m')
        if self.beam_radius is None:
            object.__setattr__(self, 'beam_radius', self.radius)
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

    def sag(self, r: 'Quantity') -> 'Quantity':
        """Evaluate the axiparabola sag by solving its surface equation."""
        from scipy.integrate import solve_ivp

        r_m = np.asarray(r.m_as('m'))
        focal_length_m = self.focal_length.m_as('m')
        focal_depth_m = self.focal_depth.m_as('m')
        radius_m = self.radius.m_as('m')
        beam_radius_m = self.beam_radius.m_as('m')

        if r_m.ndim != 1:
            raise ValueError('r must be one-dimensional')
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
            return np.zeros_like(r_m) * ureg.m

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
            method='DOP853',
            rtol=1e-13,
            atol=1e-16,
        )
        if not solution.success:
            raise RuntimeError(f'Failed to calculate axiparabola sag: {solution.message}')

        return solution.y[0] * ureg.m

    def sag_der(self, r: 'Quantity') -> np.ndarray:
        """Evaluate ds/dr directly from the axiparabola surface equation."""
        sag = self.sag(r)
        focal_distance = self.focal_length + self.focal_depth * (r / self.beam_radius) ** 2
        distance_to_focus = focal_distance - sag
        return (r / (np.hypot(r, distance_to_focus) + distance_to_focus)).m_as('')

    def calculate_constant_velocity_coefficients(self, dv: float) -> tuple['Quantity', 'Quantity']:
        """Calculate the radial-delay coefficients for a focal velocity.

        The coefficients define the delay
        ``tau_delay = alpha * r**2 + beta * r**4``.

        Parameters
        ----------
        dv : float
            Velocity difference relative to the speed of light, ``v/c - 1``.

        Returns
        -------
        alpha : pint.Quantity
            Quadratic radial-delay coefficient.
        beta : pint.Quantity
            Quartic radial-delay coefficient.
        """
        c = ureg('speed_of_light')
        alpha = -dv * self.focal_depth / self.beam_radius**2 / c
        beta0 = self.focal_depth / (4 * self.focal_length**2 * self.beam_radius**2 * c)
        beta = beta0 * (1 + 2 * dv)
        return alpha, beta

    def add_constant_velocity_phase(
        self,
        envelope: 'ScalarFieldEnvelope',
        dv: float,
        *,
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
        compensate_elongation : bool, optional
            Add a radial GDD term to compensate for pulse elongation.
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

        c = ureg('speed_of_light')
        r = envelope.r * ureg.m

        alpha, beta = self.calculate_constant_velocity_coefficients(dv)
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
    compensate_elongation : bool, optional
        Whether to compensate for the elongation of the pulse due to the delay term by adding a GDD term, by default False
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
        compensate_elongation=compensate_elongation,
        return_correction_terms=return_correction_terms,
    )
