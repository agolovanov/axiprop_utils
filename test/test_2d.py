import pint
import numpy as np

from axiprop_utils import analyze_field
from axiprop_utils.pulse import create_gaussian_pulse

ureg = pint.get_application_registry()


def test_analyze_field_gaussian():
    from pic_utils.functions import fwhm_radial

    wavelength = 1 * ureg.um
    energy = 1.6 * ureg.J
    duration_fwhm = 30 * ureg.fs
    radius = 10 * ureg.um

    t_axis = np.linspace(-4 * duration_fwhm, 4 * duration_fwhm, 200)
    r_axis = np.linspace(0, 3 * radius, 200)

    scl = create_gaussian_pulse(
        wavelength, energy, duration_fwhm=duration_fwhm, radius=radius, t_axis=t_axis, r_axis=r_axis
    )

    np.testing.assert_allclose(energy.m_as('J'), scl.Energy, rtol=1e-5)
    np.testing.assert_allclose(energy.m_as('J'), scl.Energy_ft, rtol=1e-2)

    stats = analyze_field(scl)

    np.testing.assert_allclose(stats['energy'].m_as('J'), energy.m_as('J'), rtol=1e-4)
    np.testing.assert_allclose(stats['duration_global'].m_as('fs'), duration_fwhm.m_as('fs'), rtol=1e-3)
    np.testing.assert_allclose(stats['duration_on_axis'].m_as('fs'), duration_fwhm.m_as('fs'), rtol=1e-3)

    fwhm = fwhm_radial(stats['fluence'], stats['r'])
    np.testing.assert_allclose(fwhm.m_as('m'), radius.m_as('m') * np.sqrt(2 * np.log(2)), rtol=1e-3)

    np.testing.assert_allclose(np.trapezoid(stats['power'], stats['t']).m_as('J'), energy.m_as('J'), rtol=1e-3)
    np.testing.assert_allclose(
        np.trapezoid(stats['spectral_power'], stats['omega']).m_as('J'), energy.m_as('J'), rtol=1e-3
    )
    np.testing.assert_allclose(
        -np.trapezoid(stats['spectral_power_lambda'], stats['lambda']).m_as('J'), energy.m_as('J'), rtol=1e-3
    )
