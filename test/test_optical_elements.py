"""Tests for optical element classes."""

import numpy as np
import pint
import pytest

from axiprop_utils import ParabolicMirror, mirror_parabolic, mirror_reflection
from axiprop_utils.pulse import create_gaussian_pulse

ureg = pint.get_application_registry()


def test_parabolic_mirror_sag():
    """The parabolic mirror should evaluate its analytic sag with units."""
    focal_length = 0.5 * ureg.m
    r = np.linspace(0, 10, 11) * ureg.mm

    sag = ParabolicMirror(focal_length).sag(r)

    np.testing.assert_allclose(sag.m_as('m'), (r**2 / (4 * focal_length)).m_as('m'))


def test_mirror_parabolic_delegates_to_class():
    """The deprecated function should produce the class spectral multiplier."""
    focal_length = 0.5 * ureg.m
    wavelength = 800 * ureg.nm
    kz = np.linspace(0.95, 1.05, 5) * 2 * np.pi / wavelength
    r = np.linspace(0, 10, 11) * ureg.mm
    mirror = ParabolicMirror(focal_length)

    with pytest.warns(DeprecationWarning, match='ParabolicMirror'):
        actual = mirror_parabolic(kz, r, focal_length)

    expected = mirror.spectral_multiplier(kz, r)
    np.testing.assert_allclose(actual, expected)


def test_parabolic_mirror_apply_returns_new_envelope():
    """Applying a lossless mirror should preserve energy and its input."""
    wavelength = 800 * ureg.nm
    duration = 30 * ureg.fs
    radius = 1 * ureg.mm
    t_axis = np.linspace(-4 * duration, 4 * duration, 64)
    r_axis = np.linspace(0, 3 * radius, 48)
    envelope = create_gaussian_pulse(
        wavelength,
        1 * ureg.J,
        duration,
        radius,
        t_axis=t_axis,
        r_axis=r_axis,
    )
    field_ft_before = envelope.Field_ft.copy()
    mirror = ParabolicMirror(0.5 * ureg.m)

    reflected = mirror.apply(envelope)
    sag = mirror.sag(envelope.r * ureg.m)
    expected_multiplier = mirror_reflection(envelope.k_freq / ureg.m, sag)

    assert reflected is not envelope
    np.testing.assert_allclose(envelope.Field_ft, field_ft_before)
    np.testing.assert_allclose(reflected.Field_ft, envelope.Field_ft * expected_multiplier)
    np.testing.assert_allclose(np.abs(reflected.Field_ft), np.abs(envelope.Field_ft), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(reflected.Energy_ft, envelope.Energy_ft, rtol=1e-12)
