import numpy as np
from axiprop.containers import ScalarFieldEnvelope

import typing


def create_gaussian_pulse(wavelength, energy, duration_fwhm, radius, *, t_axis, r_axis, transverse_order=2):
    """Initialize a Gaussian pulse as a ScalarFieldEnvelope.

    Parameters
    ----------
    wavelength : pint.Quantity
        central wavelength of the pulse
    energy : pint.Quantity
        the total energy in the pulse
    duration_fwhm : pint.Quantity
        the FWHM duration
    radius : pint.Quantity
        the radius of the distribution
    t_axis : pint.Quantity (array)
        array of times
    r_axis : pint.Quantity (array)
        array of r coordinates
    transverse_order : int, optional
        the number N in exp(-(r/R)^N) transverse profile, default 0.


    Returns
    -------
    axiprop.containers.ScalarFieldEnvelope
        The envelope containing the Gaussian pulse
    """
    k0 = 2 * np.pi / wavelength
    tau = duration_fwhm / np.sqrt(2 * np.log(2))

    envelope = ScalarFieldEnvelope(k0.m_as('1/m'), t_axis.m_as('s'), 4)
    envelope.make_gaussian_pulse(
        r_axis.m_as('m'), tau.m_as('s'), radius.m_as('m'), a0=1.0, n_ord=transverse_order, Energy=energy.m_as('J')
    )
    return envelope


def create_gaussian_pulse_3d(wavelength, energy, duration_fwhm, radius, *, t_axis, x_axis, y_axis, transverse_order=2):
    from axiprop_utils.utils import ensure_tuple

    radius_x, radius_y = ensure_tuple(radius)
    transverse_order_x, transverse_order_y = ensure_tuple(transverse_order)

    radius_x = radius_x.m_as('m')
    radius_y = radius_y.m_as('m')
    t_axis = t_axis.m_as('s')
    x_axis = x_axis.m_as('m')
    y_axis = y_axis.m_as('m')

    r_axis = np.sqrt(x_axis[:, None] ** 2 + y_axis[None, :] ** 2)

    k0 = 2 * np.pi / wavelength.m_as('m')
    tau = duration_fwhm.m_as('s') / np.sqrt(2 * np.log(2))

    x_profile = np.exp(-((x_axis / radius_x) ** transverse_order_x))
    y_profile = np.exp(-((y_axis / radius_y) ** transverse_order_y))
    t_profile = np.exp(-((t_axis / tau) ** 2))

    field = t_profile[:, None, None] * x_profile[None, :, None] * y_profile[None, None, :]

    envelope = ScalarFieldEnvelope(k0, t_axis, 4)
    envelope.import_field(field, r_axis=(r_axis, x_axis, y_axis), make_copy=True)

    energy_scaling = np.sqrt(energy.m_as('J') / envelope.Energy)
    envelope.Field *= energy_scaling
    envelope.Field_ft *= energy_scaling

    return envelope
