import typing
from typing import override
from abc import ABC, abstractmethod

import numpy as np

if typing.TYPE_CHECKING:
    import pint


class ThinAxialMirror(ABC):
    """Abstract base class for thin axial mirrors."""

    @abstractmethod
    def sag(self, r: pint.Quantity) -> pint.Quantity:
        """Calculate the sag function s(r) of the mirror.

        Parameters
        ----------
        r : pint.Quantity
            the radial coordinate
        """
        pass

    def phase_correction(self, r: pint.Quantity, kz: pint.Quantity) -> pint.Quantity:
        """Calculate the phase correction due to the mirror.

        It is assumed that kz is a 2D array with shape (Nz, Nr), and r is a 1D array with length Nr.

        Parameters
        ----------
        r : pint.Quantity
            the radial coordinate
        kz : pint.Quantity
            the wavenumber in the z direction
        """
        sag_value = self.sag(r)
        return -(2 * sag_value[None, :] * kz).m_as('')


class ParabolicMirror(ThinAxialMirror):
    """Create a parabolic mirror with focal length f0.

    Parameters
    ----------
    f0 : pint.Quantity
        the focal length
    """

    def __init__(self, f0: pint.Quantity):
        if not f0.check('[length]'):
            raise ValueError(f'f0 must have units of length, provided units [{f0.units}]')
        if not np.isscalar(f0.magnitude):
            raise ValueError('f0 must be a scalar')

        self.f0 = f0

    @override
    def sag(self, r):
        return r**2 / (4 * self.f0)
