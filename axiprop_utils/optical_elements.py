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

    def focal_distance_der(self, r: Quantity) -> np.ndarray:
        """Evaluate the analytic radial derivative of the focal distance.

        Parameters
        ----------
        r : pint.Quantity, scalar or array
            Non-negative radial coordinates.

        Returns
        -------
        np.ndarray, same shape as r
            Dimensionless derivative df/dr.

        Raises
        ------
        NotImplementedError
            If the subclass does not supply an analytic derivative.
        """
        raise NotImplementedError('This mirror does not implement an analytic focal-distance derivative')

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

    def optical_path(self, r: Quantity) -> Quantity:
        """Evaluate the optical path from the incident reference plane to the axis.

        For collimated axial incidence, the reflected segment has length
        ``L(r) = sqrt(r**2 + (f(r) - s(r))**2)``. The incident segment contributes
        ``-s(r)`` relative to the vertex plane, so ``p(r) = L(r) - s(r)``.

        Parameters
        ----------
        r : pint.Quantity of shape (N_r,)
            Radial coordinates at which to evaluate the path.

        Returns
        -------
        pint.Quantity of shape (N_r,)
            Vacuum optical path to each ray's focal position, with the common
            incident propagation distance to the mirror vertex subtracted.
        """
        return 1 / self._H_func(r) - self.focal_distance(r)

    def ray_shift_on_axis(self, r: Quantity, tau_delay: Quantity | None = None) -> Quantity:
        """Evaluate each ray's axial shift at its axis crossing relative to z = ct.

        Parameters
        ----------
        r : pint.Quantity, scalar or array
            Non-negative radii at which the rays strike the mirror.
        tau_delay : pint.Quantity, scalar or array, optional
            Incident radial delay in time units, with the same shape as r.
            Defaults to zero. Positive values delay the rays.

        Returns
        -------
        pint.Quantity, same shape as r
            Axial shifts ``f(r) - p(r) - c * tau_delay(r)``. Positive shifts are
            ahead of the reference ray and negative shifts are behind it.

        Notes
        -----
        Assumes collimated axial incidence, vacuum propagation, and zero sag
        at the vertex. Time is measured from the undelayed incident pulse
        reaching the vertex, so the reference ray is at z = ct. The delay
        changes arrival times without changing the reflected ray directions.
        """
        if tau_delay is not None:
            if not isinstance(tau_delay, pint.Quantity):
                raise TypeError('tau_delay must be a pint.Quantity with time units')
            if np.shape(tau_delay.magnitude) != np.shape(r.magnitude):
                raise ValueError('tau_delay must have the same shape as r')
            tau_delay = tau_delay.to('s')
        shift = self.focal_distance(r) - self.optical_path(r)
        if tau_delay is not None:
            shift = shift - ureg.speed_of_light * tau_delay
        return shift

    def velocity_on_axis(self, r: Quantity, tau_delay_der: Quantity | None = None) -> np.ndarray:
        """Evaluate the geometrical axis-crossing velocity normalized to c.

        Parameters
        ----------
        r : pint.Quantity, scalar or array
            Non-negative radii at which the rays strike the mirror.
        tau_delay_der : pint.Quantity, scalar or array, optional
            Radial derivative of the incident delay, d(tau_delay)/dr, in time per
            length units and with the same shape as r. Defaults to zero.

        Returns
        -------
        np.ndarray, same shape as r
            Dimensionless signed focal-locus velocity v/c at z = focal_distance(r).
            Returns NaN where both position and arrival-time derivatives
            vanish, and infinity for nonzero position derivative and zero
            arrival-time derivative.

        Notes
        -----
        Uses ``v/c = f_prime / (f_prime * cos(theta) + c * tau_delay_der)``, with
        ``f_prime = focal_distance_der(r)`` and
        ``cos(theta) = (1 - sag_der**2) / (1 + sag_der**2)``. This is the
        velocity of successive axis crossings, not an individual ray's axial
        velocity or the diffraction-resolved intensity peak. It assumes the
        same geometry and time origin as ray_shift_on_axis.

        Requires an analytic focal_distance_der implementation. No numerical differentiation
        or limiting values at stationary points are inferred. In particular,
        an undelayed parabola has no moving focal locus and returns NaN.
        Non-monotonic focal maps may give multiple velocities at the same z.
        """
        delay_slope = 0.0
        if tau_delay_der is not None:
            if not isinstance(tau_delay_der, pint.Quantity):
                raise TypeError('tau_delay_der must be a pint.Quantity with time per length units')
            if np.shape(tau_delay_der.magnitude) != np.shape(r.magnitude):
                raise ValueError('tau_delay_der must have the same shape as r')
            delay_slope = (ureg.speed_of_light * tau_delay_der).m_as('')
        focal_slope = np.asarray(self.focal_distance_der(r))
        slope_squared = self.sag_der(r) ** 2
        cos_theta = (1 - slope_squared) / (1 + slope_squared)
        with np.errstate(divide='ignore', invalid='ignore'):
            velocity_ratio = focal_slope / (focal_slope * cos_theta + delay_slope)
        return np.asarray(velocity_ratio)

    def intensity_on_axis(
        self,
        z: Quantity,
        scl: Quantity | ScalarFieldEnvelope,
        *,
        k0: Quantity | None = None,
        radius: Quantity | None = None,
        nr: int | None = None,
    ) -> Quantity:
        """Estimate the on-axis intensity from the incident beam.

        Parameters
        ----------
        z : pint.Quantity of shape (N_z,)
            Uniformly spaced, strictly increasing longitudinal coordinates
            relative to the mirror vertex. At least two points are required.
        scl : pint.Quantity or ScalarFieldEnvelope
            Constant incident intensity, or an axisymmetric incident envelope.
            For an envelope, use the temporal maximum of intensity at each
            radius. The mirror aperture clips the incident beam.
        k0 : pint.Quantity, optional
            Positive carrier wavenumber in inverse-length units. Required
            for constant intensity; defaults to the envelope's carrier otherwise.
        radius : pint.Quantity, optional
            Positive illuminated radius, clipped to the mirror's radius when
            defined. Defaults to the mirror radius, or the envelope's radial
            extent for mirrors without a radius. Required for constant intensity
            when the mirror has no radius.
        nr : int, optional
            Number of radial points, uniform in squared radius. Required for
            constant illumination. For an envelope, defaults to its radial grid;
            supply nr to resample it. Refine sampling to check convergence.

        Returns
        -------
        pint.Quantity of shape (N_z,)
            Estimated intensity in W/cm**2 on the supplied grid.

        Notes
        -----
        Deposit powers using trapezoidal weights for ``pi * I(r) * d(r**2)``
        at the original radii's focal positions with
        pic_utils.functions.density_1d, then multiply by k0.
        Linear deposition sums all focal branches and smooths over one z grid
        spacing. Sum(I_axis) * dz / k0 recovers the sampled incident power when
        the grid includes the full deposition support.

        This stationary-phase estimate ignores interference, pulse timing, and
        diffraction at caustics. Collapsed focal lines require diffraction.
        """
        from axiprop.containers import ScalarFieldEnvelope
        from pic_utils.functions import density_1d

        if k0 is not None and not isinstance(k0, pint.Quantity):
            raise TypeError('k0 must be a pint.Quantity with inverse-length units')
        z_m = np.asarray(z.m_as('m'))
        if z_m.ndim != 1 or z_m.size < 2:
            raise ValueError('z must be one-dimensional with at least two bin centers')
        if not np.all(np.isfinite(z_m)) or np.any(np.diff(z_m) <= 0):
            raise ValueError('z must be finite and strictly increasing')
        if not np.allclose(np.diff(z_m), z_m[1] - z_m[0], rtol=1e-10, atol=0):
            raise ValueError('z must be uniformly spaced')
        if nr is not None and (isinstance(nr, (bool, np.bool_)) or not isinstance(nr, (int, np.integer)) or nr < 2):
            raise ValueError('nr must be an integer of at least two')

        mirror_radius = getattr(self, 'radius', None)
        radius_m = None
        for aperture in (mirror_radius, radius):
            if aperture is None:
                continue
            aperture_m = aperture.m_as('m')
            if np.ndim(aperture_m) != 0 or not np.isfinite(aperture_m) or aperture_m <= 0:
                raise ValueError('radius must be a positive, finite scalar length')
            radius_m = aperture_m if radius_m is None else min(radius_m, aperture_m)
        if isinstance(scl, ScalarFieldEnvelope):
            if scl.Field.ndim != 2:
                raise ValueError('scl must be an axisymmetric envelope')
            source_r = np.asarray(scl.r)
            if (
                source_r.ndim != 1
                or source_r.size < 2
                or not np.all(np.isfinite(source_r))
                or np.any(source_r < 0)
                or np.any(np.diff(source_r) <= 0)
            ):
                raise ValueError('The envelope radial grid must be finite, non-negative, and strictly increasing')
            if k0 is None:
                k0 = scl.k0 / ureg.m
            # Sample only illuminated radii; avoid interpolating across a hard
            # aperture or across the boundary of the envelope's radial domain.
            r_min = source_r[0]
            r_max = source_r[-1] if radius_m is None else min(radius_m, source_r[-1])
            if r_min >= r_max:
                return np.zeros(z_m.shape) * ureg('W/cm**2')
            if nr is None:
                r_m = np.append(source_r[source_r < r_max], r_max)
            else:
                r_m = np.sqrt(np.linspace(r_min**2, r_max**2, nr))
            conversion = (ureg.speed_of_light * ureg.vacuum_permittivity * (ureg.V / ureg.m) ** 2 / 2).m_as('W/m**2')
            incident = np.interp(r_m, source_r, conversion * np.max(np.abs(scl.Field) ** 2, axis=0))
        elif isinstance(scl, pint.Quantity):
            intensity = np.asarray(scl.m_as('W/m**2'))
            if intensity.ndim != 0:
                raise ValueError('Constant incident intensity must be scalar')
            if k0 is None:
                raise ValueError('k0 is required for a constant incident intensity')
            if radius_m is None:
                raise ValueError('radius is required for constant intensity when the mirror has no radius')
            if nr is None:
                raise ValueError('nr is required for constant incident intensity')
            r_m = np.sqrt(np.linspace(0, radius_m**2, nr))
            incident = np.full(nr, intensity)
        else:
            raise TypeError('scl must be an intensity quantity or a ScalarFieldEnvelope')

        k0_m = k0.m_as('1/m')
        if np.ndim(k0_m) != 0 or not np.isfinite(k0_m) or k0_m <= 0:
            raise ValueError('k0 must be a positive, finite scalar wavenumber')
        if not np.all(np.isfinite(incident)) or np.any(incident < 0):
            raise ValueError('Incident intensity must be finite and non-negative')
        focal_m = np.asarray(self.focal_distance(r_m * ureg.m).m_as('m'))
        if focal_m.shape != r_m.shape or not np.all(np.isfinite(focal_m)):
            raise ValueError('focal_distance must return one finite value per radius')
        if np.ptp(focal_m) == 0:
            raise ValueError('A collapsed focal line requires a diffraction calculation')

        area = np.pi * np.diff(r_m**2)
        power = incident * (np.r_[0, area] + np.r_[area, 0]) / 2
        density = density_1d(focal_m, power, z_m)
        return (k0_m * density * ureg('W/m**2')).to('W/cm**2')

    def a0_on_axis(
        self,
        z: Quantity,
        scl: Quantity | ScalarFieldEnvelope,
        *,
        k0: Quantity | None = None,
        radius: Quantity | None = None,
        nr: int | None = None,
    ) -> np.ndarray:
        """Estimate the on-axis normalized vector potential.

        Parameters
        ----------
        z : pint.Quantity of shape (N_z,)
            Uniform longitudinal grid, as for intensity_on_axis.
        scl : pint.Quantity or ScalarFieldEnvelope
            Incident intensity or envelope, as for intensity_on_axis.
        k0 : pint.Quantity, optional
            Carrier wavenumber in inverse-length units. Defaults to the
            envelope's carrier; required for constant incident intensity.
        radius : pint.Quantity, optional
            Illuminated radius, as for intensity_on_axis.
        nr : int, optional
            Radial point count, as for intensity_on_axis.

        Returns
        -------
        np.ndarray of shape (N_z,)
            Dimensionless amplitude ``a0 = e * E / (m_e * c**2 * k0)``,
            with ``E = sqrt(2 * I_axis / (epsilon_0 * c))``.

        Notes
        -----
        Shares the approximations and sampling of intensity_on_axis.
        """
        intensity = self.intensity_on_axis(z, scl, k0=k0, radius=radius, nr=nr)
        if k0 is None:
            k0 = scl.k0 / ureg.m
        c = ureg.speed_of_light
        field = np.sqrt(2 * intensity / (ureg.vacuum_permittivity * c))
        return (ureg.elementary_charge * field / (ureg.electron_mass * c**2 * k0)).m_as('')

    def spot_radius_first_zero(self, r: Quantity, *, k0: Quantity) -> Quantity:
        """Estimate the first-zero radius of an annulus's focal spot.

        Parameters
        ----------
        r : pint.Quantity, scalar or array
            Non-negative radii on the mirror.
        k0 : pint.Quantity
            Positive scalar carrier wavenumber in inverse-length units.

        Returns
        -------
        pint.Quantity, scalar or array
            First-zero spot radius at z = focal_distance(r), in metres.
            Returns infinity at r = 0.

        Notes
        -----
        Assumes the paraxial annular intensity profile
        ``J0(k0 * spot_r * r / focal_distance(r))**2``. The first zero is at
        ``spot_r = j_01 * focal_distance(r) / (k0 * r)``.
        This describes each annulus separately, not the combined profile of
        overlapping focal branches. The approximation fails near r = 0.
        """
        from scipy.special import jn_zeros

        if not isinstance(k0, pint.Quantity):
            raise TypeError('k0 must be a pint.Quantity with inverse-length units')
        k0_m = k0.m_as('1/m')
        if np.ndim(k0_m) != 0 or not np.isfinite(k0_m) or k0_m <= 0:
            raise ValueError('k0 must be a positive, finite scalar wavenumber')
        r_m = np.asarray(r.m_as('m'))
        if not np.all(np.isfinite(r_m)) or np.any(r_m < 0):
            raise ValueError('r must be finite and non-negative')
        focal_m = self.focal_distance(r).m_as('m')
        radius_m = np.divide(
            jn_zeros(0, 1)[0] * focal_m,
            k0_m * r_m,
            out=np.full(r_m.shape, np.inf),
            where=r_m > 0,
        )
        return radius_m * ureg.m

    def spot_fwhm(self, r: Quantity, *, k0: Quantity) -> Quantity:
        """Estimate the full width at half maximum of an annulus's focal spot.

        Parameters
        ----------
        r : pint.Quantity, scalar or array
            Non-negative radii on the mirror.
        k0 : pint.Quantity
            Positive scalar carrier wavenumber in inverse-length units.

        Returns
        -------
        pint.Quantity, scalar or array
            Full intensity width at half maximum at z = focal_distance(r),
            in metres. Returns infinity at r = 0.

        Notes
        -----
        Uses the same paraxial J0**2 profile and limitations as
        spot_radius_first_zero. The full width is twice the radius where
        the central lobe reaches half its peak intensity.
        """
        # 2 * x_half / j_01, where J0(x_half)**2 = 1/2 in the central lobe.
        return 0.9367533838558382 * self.spot_radius_first_zero(r, k0=k0)

    def ray_radius(self, z: Quantity, r0: Quantity) -> Quantity:
        """Evaluate the signed radius along a reflected ray.

        Parameters
        ----------
        z : pint.Quantity, scalar or of shape (N_z,)
            Longitudinal positions relative to the mirror vertex plane.
        r0 : pint.Quantity, scalar
            Non-negative radius at which the ray strikes the mirror.

        Returns
        -------
        pint.Quantity, scalar or of shape (N_z,)
            Signed radial positions, negative after the ray crosses the axis.

        Notes
        -----
        Assumes collimated axial incidence and a nonzero difference between
        focal distance and sag. The reflected straight line is extrapolated
        for all supplied z values.
        """
        sag = self.sag(r0)
        focal_distance = self.focal_distance(r0)
        return r0 * (focal_distance - z) / (focal_distance - sag)

    def ray_snapshot(self, z0: Quantity, r0: Quantity, tau_delay: Quantity | None = None) -> tuple[Quantity, Quantity]:
        """Evaluate reflected ray positions when the undelayed reference ray is at z0.

        Parameters
        ----------
        z0 : pint.Quantity, scalar
            Longitudinal position of the undelayed axial reference ray relative
            to the mirror vertex.
        r0 : pint.Quantity, scalar or of shape (N_r,)
            Non-negative radii at which the rays strike the mirror.
        tau_delay : pint.Quantity, scalar or of shape (N_r,), optional
            Incident radial delay in time units, with the same shape as r0.
            Defaults to zero. Positive values delay the rays.

        Returns
        -------
        z : pint.Quantity, scalar or of shape (N_r,)
            Longitudinal positions of the rays at the same time.
        r : pint.Quantity, scalar or of shape (N_r,)
            Signed radial positions, negative after each ray crosses the axis.

        Notes
        -----
        Assumes collimated axial incidence, vacuum propagation, and zero sag
        at the vertex. The incident sag and delay corrections make the distance
        traveled after reflection ``z0 + s(r0) - c * tau_delay(r0)``. The reference
        ray does not include tau_delay(0). Delays change arrival times without
        changing the reflected ray directions. Only reflected trajectories
        are described, extrapolated even when that distance is negative.
        """
        if tau_delay is not None:
            if not isinstance(tau_delay, pint.Quantity):
                raise TypeError('tau_delay must be a pint.Quantity with time units')
            if np.shape(tau_delay.magnitude) != np.shape(r0.magnitude):
                raise ValueError('tau_delay must have the same shape as r0')
            tau_delay = tau_delay.to('s')
        sag = self.sag(r0)
        focal_distance = self.focal_distance(r0)
        reflected_length = np.hypot(r0, focal_distance - sag)
        distance = z0 + sag
        if tau_delay is not None:
            distance = distance - ureg.speed_of_light * tau_delay
        return (
            sag + distance * (focal_distance - sag) / reflected_length,
            r0 * (1 - distance / reflected_length),
        )


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
        return np.ones(np.shape(r)) / (2 * self.focal_length)

    @typing.override
    def focal_distance(self, r: Quantity) -> Quantity:
        return np.ones(np.shape(r)) * self.focal_length

    @typing.override
    def focal_distance_der(self, r: Quantity) -> np.ndarray:
        return np.zeros(np.shape(r))

    @typing.override
    def optical_path(self, r: Quantity) -> Quantity:
        return np.ones(np.shape(r)) * self.focal_length
