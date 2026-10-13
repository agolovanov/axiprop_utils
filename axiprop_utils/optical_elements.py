"""Optical elements acting on scalar field envelopes."""

from __future__ import annotations

import typing
import warnings
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

    @abstractmethod
    def _H_func(self, r: Quantity) -> Quantity:
        """Evaluate the mirror's auxiliary function H(r) = sag_der(r) / r.

        For a smooth surface, its value at r=0 is the limit H(0) = s''(0).

        Parameters
        ----------
        r : pint.Quantity of shape (N_r,)
            Radial coordinates at which to evaluate the H function.

        Returns
        -------
        pint.Quantity of shape (N_r,)
            The H function values in inverse-length units at each radial coordinate.
        """

    def sag_der(self, r: Quantity) -> np.ndarray:
        """Evaluate the radial derivative of the mirror sag.

        Parameters
        ----------
        r : pint.Quantity of shape (N_r,)
            Radial coordinates.

        Returns
        -------
        np.ndarray of shape (N_r,)
            Dimensionless slope ds/dr at each radial coordinate.
        """
        return (self._H_func(r) * r).m_as('')

    def focal_distance(self, r: Quantity) -> Quantity:
        """Evaluate the local focal distance from the reference plane.

        Parameters
        ----------
        r : pint.Quantity of shape (N_r,)
            Radial coordinates.

        Returns
        -------
        pint.Quantity of shape (N_r,)
            Local focal distance at each radial coordinate.
        """
        H_value = self._H_func(r)
        return self.sag(r) + (1 - (r * H_value) ** 2) / (2 * H_value)

    def spectral_multiplier(
        self,
        kz: np.ndarray | Quantity,
        r: Quantity,
        *,
        k0: float | Quantity | None = None,
        nyquist_threshold: float = 1.0,
    ) -> np.ndarray:
        """Calculate the spectral multiplier produced by the mirror.

        Parameters
        ----------
        kz : np.ndarray or pint.Quantity of shape (N_kz,)
            Longitudinal wavevector axis. Unitless arrays are interpreted in
            inverse metres.
        r : pint.Quantity of shape (N_r,)
            Radial coordinates.
        k0 : float or pint.Quantity, optional
            Central optical wavenumber, in inverse metres if unitless. When
            omitted, the sampling check is skipped; the carrier cannot be
            inferred reliably from the frequency axis.
        nyquist_threshold : float, optional
            Positive, finite phase-step limit in units of pi, default 1.
            Warn when the magnitude of the adjacent-point phase correction
            at k0 exceeds this limit. Use 0.5 for a stricter sampling margin.

        Returns
        -------
        np.ndarray of shape (N_kz, N_r)
            Complex spectral multiplier representing the reflection.

        Notes
        -----
        The check uses the unwrapped sag difference over the entire radial
        grid. It does not account for pulse amplitude, spectral bandwidth,
        or the incident wavefront, and is not a propagation accuracy guarantee.
        """
        if not np.isfinite(nyquist_threshold) or nyquist_threshold <= 0:
            raise ValueError('nyquist_threshold must be positive and finite')

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

        if k0 is not None:
            k0_m = k0.m_as('1/m') if isinstance(k0, pint.Quantity) else k0
            if np.ndim(k0_m) != 0 or not np.isfinite(k0_m) or k0_m <= 0:
                raise ValueError('k0 must be a positive, finite scalar wavenumber')
            phase_steps = 2 * k0_m * np.abs(np.diff(sag_m))
            max_step = np.max(phase_steps, initial=0.0)
            if max_step > nyquist_threshold * np.pi:
                warnings.warn(
                    f'Mirror phase correction between adjacent radial points reaches {max_step / np.pi:.3g} * pi '
                    f'at k0, exceeding nyquist_threshold={nyquist_threshold:g}. '
                    'The radial grid may undersample the reflected field; consider refining it.',
                    UserWarning,
                    stacklevel=2,
                )

        return np.exp(-2j * kz[:, np.newaxis] * sag_m[np.newaxis, :])

    def minimum_nr(self, r: Quantity, *, k0: float | Quantity, nyquist_threshold: float = 1.0) -> int:
        """Estimate the minimum uniform radial point count from the sag slope.

        Parameters
        ----------
        r : pint.Quantity of shape (N_r,)
            Strictly increasing, non-negative coordinates at which to evaluate
            sag_der. At least two points are required. The estimated
            grid spans r[0] to r[-1], including both endpoints.
        k0 : float or pint.Quantity
            Positive central optical wavenumber, in inverse metres if unitless.
            Also accepts wavelength, frequency, or density quantities supported
            by pic_utils.PlasmaUnits, using their corresponding wavenumber.
        nyquist_threshold : float, optional
            Positive, finite phase-step limit in units of pi, default 1.

        Returns
        -------
        int
            At least two points, including both endpoints. Construct the grid
            with np.linspace(r[0], r[-1], nr).

        Notes
        -----
        Uses ``1 + ceil(2 * k0 * (r[-1] - r[0]) * max(abs(ds/dr)) / (threshold * pi))``,
        with the slope supplied by sag_der. This is a
        conservative local-slope estimate, not an exact minimum for the
        discrete adjacent-point check. The input coordinates must capture the
        maximum absolute slope; endpoints suffice when it is monotonic.
        Otherwise, sample the interior too. Check the resulting grid with
        spectral_multiplier. Only the mirror phase at k0 is considered.
        """
        if not np.isfinite(nyquist_threshold) or nyquist_threshold <= 0:
            raise ValueError('nyquist_threshold must be positive and finite')
        if isinstance(k0, pint.Quantity):
            if not k0.check('[length] ** -1'):
                from pic_utils import PlasmaUnits

                k0 = PlasmaUnits(k0).wavenumber
            k0_m = k0.m_as('1/m')
        else:
            k0_m = k0
        if np.ndim(k0_m) != 0 or not np.isfinite(k0_m) or k0_m <= 0:
            raise ValueError('k0 must be a positive, finite scalar wavenumber')
        r_m = np.asarray(r.m_as('m'))
        if r_m.ndim != 1 or r_m.size < 2:
            raise ValueError('r must be one-dimensional with at least two points')
        if not np.all(np.isfinite(r_m)) or np.any(r_m < 0) or np.any(np.diff(r_m) <= 0):
            raise ValueError('r must be finite, non-negative, and strictly increasing')
        slope = np.asarray(self.sag_der(r))
        max_slope = np.max(np.abs(slope))
        intervals = 2 * k0_m * (r_m[-1] - r_m[0]) * max_slope / (nyquist_threshold * np.pi)
        return max(2, int(np.ceil(intervals)) + 1)

    def apply(self, envelope: ScalarFieldEnvelope, *, nyquist_threshold: float = 1.0) -> ScalarFieldEnvelope:
        """Apply the mirror reflection to an envelope.

        Reflection is represented as a thin spectral-phase element in a folded
        propagation frame. The input envelope is not modified.

        Parameters
        ----------
        envelope : ScalarFieldEnvelope
            Axisymmetric scalar field envelope incident on the mirror.
        nyquist_threshold : float, optional
            Positive, finite adjacent-point phase-step limit in units of pi,
            default 1. The sampling check uses envelope.k0 over the entire
            radial grid; see spectral_multiplier for its limitations.

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

        multiplier = self.spectral_multiplier(
            envelope.k_freq, envelope.r * ureg.m, k0=envelope.k0, nyquist_threshold=nyquist_threshold
        )
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

    def _H_func(self, r: Quantity) -> Quantity:
        """Evaluate the constant parabolic H function ``1 / (2 * focal_length)``."""
        return np.ones(r.shape) / (2 * self.focal_length)

    @typing.override
    def focal_distance(self, r: Quantity) -> Quantity:
        return np.ones(r.shape) * self.focal_length
