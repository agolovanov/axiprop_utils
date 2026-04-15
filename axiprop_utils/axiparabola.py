"""
Module for axiparabola utilities
"""

import pint
import numpy as np
import typing

ureg = pint.get_application_registry()

if typing.TYPE_CHECKING:
    from axiprop import ScalarFieldEnvelope
    from pint import Quantity


@ureg.wraps(None, (ureg.m**-1, ureg.m, ureg.m, ureg.m, ureg.m, None))
def mirror_axiparabola(kz, r, f0, d0, R, N_cut=4):
    """Calculates the phase term for a mirror with an axiparabolic shape.

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
        the number of points to cut from the beginning of the radial axis, by default 4

    Returns
    -------
    np.ndarray of shape (N_kz, N_r)
        the phase term representing the reflection on the axiparabola
    """
    from . import mirror_reflection
    from scipy.integrate import solve_ivp

    s_ax = np.zeros_like(r)
    r_loc = r[N_cut:]

    def sag_equation(r, s):
        return (s - (f0 + d0 * r**2 / R**2) + np.sqrt(r**2 + ((f0 + d0 * r**2 / R**2) - s) ** 2)) / r

    s_ax[N_cut:] = solve_ivp(
        sag_equation,
        (r_loc[0], r_loc[-1]),
        [
            r_loc[0] / (4 * f0),
        ],
        t_eval=r_loc,
        method='DOP853',
        rtol=1e-13,
        atol=1e-16,
    ).y.flatten()

    s_ax[:N_cut] = s_ax[N_cut]
    s_ax -= s_ax[0]

    return mirror_reflection(kz, s_ax)


def add_constant_velocity_phase(
    envelope: 'ScalarFieldEnvelope',
    dv: float,
    *,
    f0: 'Quantity',
    d0: 'Quantity',
    R: 'Quantity',
    compensate_elongation: bool = False,
) -> 'ScalarFieldEnvelope':
    """Add a phase term to the envelope representing a constant velocity along the optical axis for an axiparabola.

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

    Returns
    -------
    ScalarFieldEnvelope
        The envelope with the added phase term
    """
    from . import apply_radial_delay, apply_radial_GDD

    c = ureg('speed_of_light')

    r = envelope.r * ureg.m

    alpha = -dv * d0 / R**2 / c
    beta0 = 1 / (4 * f0**2) * d0 / R**2 / c
    beta = beta0 * (1 + 2 * dv)

    tau_delay = alpha * r**2 + beta * r**4

    envelope = apply_radial_delay(envelope, tau_delay)

    if compensate_elongation:
        omega0 = envelope.omega0 / ureg.s

        gdd = (alpha**2 / 2 + 2 * alpha * beta * r**2 + 2 * beta**2 * r**4) / (beta0 * omega0)

        envelope = apply_radial_GDD(envelope, gdd)

    return envelope
