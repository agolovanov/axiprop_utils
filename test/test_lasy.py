from pathlib import Path

import numpy as np
import pint

from axiprop_utils import create_gaussian_pulse, read_from_lasy, save_to_lasy

ureg = pint.get_application_registry()


def test_write_read(tmp_path: Path):
    wavelength = 1 * ureg.um
    energy = 1 * ureg.J
    duration_fwhm = 30 * ureg.fs
    radius = 10 * ureg.um

    t_axis = np.linspace(-4 * duration_fwhm, 4 * duration_fwhm, 100)
    r_axis = np.linspace(0, 3 * radius, 100)

    scl = create_gaussian_pulse(
        wavelength, energy, duration_fwhm=duration_fwhm, radius=radius, t_axis=t_axis, r_axis=r_axis
    )

    lasy_file = tmp_path / 'profile.h5'

    save_to_lasy(scl, lasy_file)

    scl_read = read_from_lasy(lasy_file, attenuate_boundaries=False)

    np.testing.assert_allclose(scl.r, scl_read.r)
    np.testing.assert_allclose(scl.t, scl_read.t)
    np.testing.assert_allclose(scl.Field, scl_read.Field)
    np.testing.assert_allclose(scl.Field_ft, scl_read.Field_ft)
