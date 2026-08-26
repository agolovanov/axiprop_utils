import numpy as np
import pint
from axiprop.lib import PropagatorResamplingFresnel

from axiprop_utils import (
    create_resampling_propagator_fresnel,
    get_propagator_r_axis,
    propagate_envelope,
)
from axiprop_utils.pulse import create_gaussian_pulse

ureg = pint.get_application_registry()


def create_test_envelope():
    return create_gaussian_pulse(
        wavelength=1 * ureg.um,
        energy=1 * ureg.mJ,
        duration_fwhm=10 * ureg.fs,
        radius=10 * ureg.um,
        t_axis=np.linspace(-20, 20, 16) * ureg.fs,
        r_axis=np.linspace(0, 30, 16) * ureg.um,
    )


def test_resampling_propagator_fresnel():
    envelope = create_test_envelope()
    R_new = 20 * ureg.um
    Nr_new = 10
    distance = 0.25 * ureg.mm

    propagator = create_resampling_propagator_fresnel(
        envelope,
        R_new,
        Nr_new,
        design_distance=distance,
        backend='NP',
        verbose=False,
    )

    assert isinstance(propagator, PropagatorResamplingFresnel)
    np.testing.assert_allclose(propagator.r, envelope.r)
    assert propagator.Nr_new == Nr_new
    np.testing.assert_allclose(propagator.Rmax_new, R_new.m_as('m'))
    assert get_propagator_r_axis(propagator) is propagator.r_new

    propagated = propagate_envelope(envelope, distance, propagator)

    assert propagated.Field_ft.shape == (envelope.Field_ft.shape[0], Nr_new)
    np.testing.assert_allclose(propagated.r, propagator.r_new)
