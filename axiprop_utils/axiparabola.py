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
    """
    Spectra-radial phase representing on-axis Axiparabola
    [Smartsev et al Opt. Lett. 44, 3414 (2019)]
    """
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

    kz2d = (kz * np.ones((*kz.shape, *r.shape)).T).T
    phi = -2 * s_ax[None, :] * kz2d

    return np.exp(1j * phi)


def add_constant_velocity_phase(
    envelope: 'ScalarFieldEnvelope', dv: float, *, f0: 'Quantity', d0: 'Quantity', R: 'Quantity'
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

    Returns
    -------
    ScalarFieldEnvelope
        The envelope with the added phase term
    """
    from . import apply_spectral_multiplier

    c = ureg['speed_of_light']

    r = envelope.r * ureg.m
    tau_delay = (d0 / R**2 * (-dv * r**2 + 1 / (2 * f0**2) * (dv + 0.5) * r**4) / c).m_as('s')

    spectral_phase = np.exp(1j * (envelope.omega[:, np.newaxis] - envelope.omega0) * tau_delay)
    return apply_spectral_multiplier(envelope, spectral_phase)
