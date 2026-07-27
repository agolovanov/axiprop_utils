"""Optical elements acting on scalar field envelopes."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pint

if typing.TYPE_CHECKING:
    from axiprop.containers import ScalarFieldEnvelope
    from pint import Quantity

ureg = pint.get_application_registry()


class AxisymmetricMirror(ABC):
    """Base class for an axisymmetric mirror described by its surface sag."""

    @abstractmethod
    def sag(self, r: Quantity) -> Quantity:
        """Evaluate the mirror sag.

        Parameters
        ----------
        r : pint.Quantity of shape (N_r,)
            Radial coordinates at which to evaluate the sag.

        Returns
        -------
        pint.Quantity of shape (N_r,)
            Mirror sag relative to the reference plane.
        """

    def spectral_multiplier(self, kz: np.ndarray | Quantity, r: Quantity) -> np.ndarray:
        """Calculate the spectral multiplier produced by the mirror.

        Parameters
        ----------
        kz : np.ndarray or pint.Quantity of shape (N_kz,)
            Longitudinal wavevector axis. Unitless arrays are interpreted in
            inverse metres.
        r : pint.Quantity of shape (N_r,)
            Radial coordinates.

        Returns
        -------
        np.ndarray of shape (N_kz, N_r)
            Complex spectral multiplier representing the reflection.
        """
        sag = self.sag(r)
        if not isinstance(sag, pint.Quantity):
            raise TypeError('sag() must return a pint.Quantity')

        if isinstance(kz, pint.Quantity):
            kz = kz.m_as('1/m')
        else:
            kz = np.asarray(kz)

        sag_m = sag.m_as('m')
        if kz.ndim != 1 or sag_m.ndim != 1:
            raise ValueError('kz and the evaluated sag must be one-dimensional')
        if sag_m.shape != r.shape:
            raise ValueError('sag() must return one value for each radial coordinate')

        return np.exp(-2j * kz[:, np.newaxis] * sag_m[np.newaxis, :])

    def apply(self, envelope: ScalarFieldEnvelope) -> ScalarFieldEnvelope:
        """Apply the mirror reflection to an envelope.

        Reflection is represented as a thin spectral-phase element in a folded
        propagation frame. The input envelope is not modified.

        Parameters
        ----------
        envelope : ScalarFieldEnvelope
            Axisymmetric scalar field envelope incident on the mirror.

        Returns
        -------
        ScalarFieldEnvelope
            A new envelope containing the reflected field.

        Raises
        ------
        ValueError
            If the envelope is not axisymmetric.
        """
        from . import apply_spectral_multiplier, is_envelope_3d

        if is_envelope_3d(envelope):
            raise ValueError('AxisymmetricMirror can only be applied to an axisymmetric envelope')

        multiplier = self.spectral_multiplier(envelope.k_freq, envelope.r * ureg.m)
        return apply_spectral_multiplier(envelope, multiplier)


@dataclass(frozen=True)
class ParabolicMirror(AxisymmetricMirror):
    """Axisymmetric parabolic mirror.

    Parameters
    ----------
    focal_length : pint.Quantity
        Mirror focal length.
    """

    focal_length: Quantity

    def sag(self, r: Quantity) -> Quantity:
        """Evaluate the parabolic sag ``r**2 / (4 * focal_length)``."""
        return r**2 / (4 * self.focal_length)
