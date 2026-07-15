import numpy as np
import pint

from axiprop_utils import mirror_axiparabola, mirror_parabolic

ureg = pint.get_application_registry()


def test_axiparabola_reduces_to_parabolic_mirror():
    wavelength = 800 * ureg.nm
    kz = np.linspace(0.95, 1.05, 5) * 2 * np.pi / wavelength
    radius = 10 * ureg.mm
    r = np.linspace(0, radius, 101)
    focal_length = 0.5 * ureg.m
    N_cut = 4

    axiparabola = mirror_axiparabola(kz, r, focal_length, 0 * ureg.m, radius, N_cut=N_cut)
    parabola = mirror_parabolic(kz, r, focal_length)

    np.testing.assert_allclose(axiparabola, parabola, rtol=1e-7, atol=1e-7)
