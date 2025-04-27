import pint
import numpy as np

from axiprop_utils import analyze_field
from axiprop_utils.pulse import create_gaussian_pulse_3d, create_gaussian_pulse

ureg = pint.get_application_registry()


def test_analyze_field_gaussian_3d():
    from pic_utils.functions import fwhm

    wavelength = 1 * ureg.um
    energy = 1.6 * ureg.J
    duration_fwhm = 30 * ureg.fs
    radius_x = 10 * ureg.um
    radius_y = 15 * ureg.um

    t_axis = np.linspace(-4 * duration_fwhm, 4 * duration_fwhm, 200)
    x_axis = np.linspace(-3 * radius_x, 3 * radius_x, 201)
    y_axis = np.linspace(-3 * radius_y, 3 * radius_y, 181)

    scl = create_gaussian_pulse_3d(
        wavelength,
        energy,
        duration_fwhm=duration_fwhm,
        radius=(radius_x, radius_y),
        t_axis=t_axis,
        x_axis=x_axis,
        y_axis=y_axis,
    )

    np.testing.assert_allclose(energy.m_as('J'), scl.Energy, rtol=1e-5)
    np.testing.assert_allclose(energy.m_as('J'), scl.Energy_ft, rtol=1e-2)

    stats = analyze_field(scl)

    np.testing.assert_allclose(stats['energy'].m_as('J'), energy.m_as('J'), rtol=1e-4)
    np.testing.assert_allclose(stats['duration_global'].m_as('fs'), duration_fwhm.m_as('fs'), rtol=1e-3)
    np.testing.assert_allclose(stats['duration_on_axis'].m_as('fs'), duration_fwhm.m_as('fs'), rtol=1e-3)

    fwhm_x = fwhm(stats['fluence_x'], stats['x'])
    np.testing.assert_allclose(fwhm_x.m_as('m'), radius_x.m_as('m') * np.sqrt(2 * np.log(2)), rtol=1e-3)
    fwhm_y = fwhm(stats['fluence_y'], stats['y'])
    np.testing.assert_allclose(fwhm_y.m_as('m'), radius_y.m_as('m') * np.sqrt(2 * np.log(2)), rtol=1e-3)

    np.testing.assert_allclose(np.trapezoid(stats['power'], stats['t']).m_as('J'), energy.m_as('J'), rtol=1e-3)
    np.testing.assert_allclose(
        np.trapezoid(stats['spectral_power'], stats['omega']).m_as('J'), energy.m_as('J'), rtol=1e-3
    )
    np.testing.assert_allclose(
        -np.trapezoid(stats['spectral_power_lambda'], stats['lambda']).m_as('J'), energy.m_as('J'), rtol=1e-3
    )


def test_analyze_field_gaussian_2d_vs_3d():
    from pic_utils.functions import fwhm, fwhm_radial

    wavelength = 1 * ureg.um
    energy = 1.6 * ureg.J
    duration_fwhm = 30 * ureg.fs
    radius = 10 * ureg.um

    t_axis = np.linspace(-4 * duration_fwhm, 4 * duration_fwhm, 200)
    r_axis = np.linspace(0, 3 * radius, 100)
    x_axis = np.linspace(-3 * radius, 3 * radius, 199)

    scl_2d = create_gaussian_pulse(
        wavelength, energy, duration_fwhm=duration_fwhm, radius=radius, t_axis=t_axis, r_axis=r_axis
    )

    scl_3d = create_gaussian_pulse_3d(
        wavelength,
        energy,
        duration_fwhm=duration_fwhm,
        radius=(radius, radius),
        t_axis=t_axis,
        x_axis=x_axis,
        y_axis=x_axis,
    )

    stats_2d = analyze_field(scl_2d)
    stats_3d = analyze_field(scl_3d)

    np.testing.assert_allclose(stats_2d['energy'].m_as('J'), stats_3d['energy'].m_as('J'), rtol=1e-4)
    np.testing.assert_allclose(
        stats_2d['duration_global'].m_as('fs'), stats_3d['duration_global'].m_as('fs'), rtol=1e-3
    )
    np.testing.assert_allclose(
        stats_2d['duration_on_axis'].m_as('fs'), stats_3d['duration_on_axis'].m_as('fs'), rtol=1e-3
    )

    fwhm_2d = fwhm_radial(stats_2d['fluence'], stats_2d['r'])
    fwhm_3d = fwhm(stats_3d['fluence_x'], stats_3d['x'])
    np.testing.assert_allclose(fwhm_2d.m_as('m'), fwhm_3d.m_as('m'), rtol=1e-3)
    np.testing.assert_allclose(fwhm_2d.m_as('m'), radius.m_as('m') * np.sqrt(2 * np.log(2)), rtol=1e-3)

    np.testing.assert_allclose(stats_2d['power'].m_as('TW'), stats_3d['power'].m_as('TW'), rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(
        stats_2d['spectral_power'].m_as('J s'), stats_3d['spectral_power'].m_as('J s'), rtol=1e-3, atol=1e-2
    )
    np.testing.assert_allclose(
        stats_2d['spectral_power_lambda'].m_as('J/nm'),
        stats_3d['spectral_power_lambda'].m_as('J/nm'),
        rtol=1e-3,
        atol=1e-2,
    )
