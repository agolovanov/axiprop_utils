import typing
from pathlib import Path

import numpy as np
import pint
from axiprop.containers import ScalarFieldEnvelope

from .pulse import create_gaussian_pulse

if typing.TYPE_CHECKING:
    from typing import Any, Tuple

    import matplotlib
    import matplotlib.figure
    from axiprop.lib import PropagatorResampling, PropagatorSymmetric

ureg = pint.get_application_registry()


def mirror_parabolic(kz, r, f0):
    kz2d = (kz * np.ones((*kz.shape, *r.shape)).T).T

    phi = -(0.5 * kz2d * r**2 / f0).m_as('')

    return np.exp(1j * phi)


def hole(r, r_hole):
    mirror_mask = np.ones(r.shape)
    mirror_mask[r < r_hole] = 0.0

    return mirror_mask


def is_envelope_3d(envelope: ScalarFieldEnvelope) -> bool:
    """Check if the envelope is 3D (2D in the transverse direction).

    Parameters
    ----------
    envelope : ScalarFieldEnvelope
        the envelope.

    Returns
    -------
    bool
        True if the envelope is 3D, False otherwise.
    """
    if len(envelope.Field.shape) == 3:
        return True
    else:
        return False


@ureg.wraps(None, (ureg.m**-1, ureg.m, ureg.m, ureg.m, ureg.m, None))
def mirror_axiparabola(kz, r, f0, d0, R, N_cut=4):
    """
    Spectra-radial phase representing on-axis Axiparabola
    [Smartsev et al Opt. Lett. 44, 3414 (2019)]
    """
    from scipy.integrate import solve_ivp

    s_ax = np.zeros_like(r)
    r_loc = r[N_cut:]

    def sag_equation(r, s):
        return (s - (f0 + d0 * r**2 / R**2) + np.sqrt(r**2 + ((f0 + d0 * r**2 / R**2) - s) ** 2)) / r

    s_ax[N_cut:] = solve_ivp(
        sag_equation,
        (r_loc[0], r_loc[-1]),
        [
            r_loc[0] / (4 * f0),
        ],
        t_eval=r_loc,
        method='DOP853',
        rtol=1e-13,
        atol=1e-16,
    ).y.flatten()

    s_ax[:N_cut] = s_ax[N_cut]
    s_ax -= s_ax[0]

    kz2d = (kz * np.ones((*kz.shape, *r.shape)).T).T
    phi = -2 * s_ax[None, :] * kz2d

    return np.exp(1j * phi)


def plot_field(
    field: ScalarFieldEnvelope | dict,
    *,
    r_units: pint.Unit | str = 'um',
    lambda_lim: 'Tuple[pint.Quantity, pint.Quantity] | None' = None,
    t_lim: 'Tuple[pint.Quantity, pint.Quantity] | None' = None,
    r_lim: pint.Quantity | None = None,
    figtitle: str | None = None,
    return_figure: bool = False,
) -> 'Tuple[matplotlib.figure.Figure, Any]':
    """Plot the scalar field envelope.

    Parameters
    ----------
    field : ScalarFieldEnvelope | dict
        either the envelope or the result of `analyze_field` run on it.
    r_units : pint.Unit | str, optional
        units to be used for the r axis, by default 'um'
    lambda_lim : Tuple[pint.Quantity, pint.Quantity], optional
        limits for the wavelength axis, by default None
    t_lim : Tuple[pint.Quantity, pint.Quantity], optional
        limits for the time axis, by default None
    r_lim : pint.Quantity, optional
        upper limit for the r axis, by default None
    figtitle : str, optional
        the title of the figure, by default None
    return_figure : bool, optional
        whether to return the figure and axes, by default False

    Returns
    -------
    Tuple[matplotlib.figure.Figure, Any] or None
        figure and axes created by matplotlib for further customization (if return_figure is True)
    """
    import matplotlib.pyplot as plt
    import mpl_utils
    from mpl_utils import remove_grid
    from axiprop_utils.utils import ensure_tuple

    c = ureg['speed_of_light']

    if isinstance(field, ScalarFieldEnvelope):
        field = analyze_field(field)

    is_3d = field['3d']

    omega_axis = field['omega']
    t_axis = field['t']
    lambda_axis = field['lambda']
    if is_3d:
        x_axis = field['x'].to(r_units)
        y_axis = field['y'].to(r_units)
    else:
        r_axis = field['r'].to(r_units)
    omega0 = field['omega0']
    power = field['power']

    if is_3d:
        fig, axes_grid = plt.subplots(figsize=(7, 6), ncols=3, nrows=3)
    else:
        fig, axes_grid = plt.subplots(figsize=(7, 4), ncols=3, nrows=2)
    axes_iter = iter(axes_grid.flatten())

    print(f'Peak field {field["max_field"]:.3g~}')
    print(f'Peak a0 {field["a0"]:.3g}')
    print(f'Peak power {np.max(power):.3g~}')
    print(f'Energy {field["energy"]:.3g~}')
    print(f'Duration {field["duration_global"]:.3g~} (global), {field["duration_on_axis"]:.3g~} (on-axis)')

    if t_lim is None:
        t_lim = np.min(t_axis), np.max(t_axis)

    if lambda_lim is None:
        lambda_lim = (np.min(lambda_axis), np.max(lambda_axis))

    if r_lim is None:
        if is_3d:
            x_lim = (np.min(x_axis), np.max(x_axis))
            y_lim = (np.min(y_axis), np.max(y_axis))
        else:
            r_lim = (np.min(r_axis), np.max(r_axis))
    else:
        if is_3d:
            xmax, ymax = ensure_tuple(r_lim.to(r_units))
            x_lim = (-xmax, xmax)
            y_lim = (-ymax, ymax)
        else:
            r_lim = (np.min(r_axis), r_lim.to(r_units))

    omega_lim = (2 * np.pi * c / lambda_lim[1], 2 * np.pi * c / lambda_lim[0])
    omega_relative_lim = tuple((omega / omega0).m_as('') for omega in omega_lim)

    if is_3d:
        ax = next(axes_iter)
        extent = mpl_utils.calculate_extent(t_axis, x_axis)
        ax.imshow(np.sqrt(field['intensity_x'].T.magnitude), extent=extent, aspect='auto', origin='lower')
        ax.set_xlabel(f'Duration, {t_axis.units}')
        ax.set_ylabel(f'$x$, {x_axis.units}')
        ax.set_xlim(t_lim)
        ax.set_ylim(x_lim)
        remove_grid(ax)

        ax = next(axes_iter)
        extent = mpl_utils.calculate_extent(t_axis, y_axis)
        ax.imshow(np.sqrt(field['intensity_y'].T.magnitude), extent=extent, aspect='auto', origin='lower')
        ax.set_xlabel(f'Duration, {t_axis.units}')
        ax.set_ylabel(f'$y$, {y_axis.units}')
        ax.set_xlim(t_lim)
        ax.set_ylim(y_lim)
        remove_grid(ax)
    else:
        ax = next(axes_iter)
        extent = mpl_utils.calculate_extent(t_axis, r_axis)
        ax.imshow(np.sqrt(field['intensity'].T.magnitude), extent=extent, aspect='auto', origin='lower')
        ax.set_xlabel(f'Duration, {t_axis.units}')
        ax.set_ylabel(f'Radius, {r_axis.units}')
        ax.set_xlim(t_lim)
        ax.set_ylim(r_lim)
        remove_grid(ax)

    ax = next(axes_iter)
    ax.plot(t_axis, power)
    ax.set_xlabel(f'Duration, {t_axis.units}')
    ax.set_ylabel(f'Power, {power.units}')
    ax.set_xlim(t_lim)
    ax.set_ylim(ymin=0)

    if is_3d:
        ax = next(axes_iter)
        extent = mpl_utils.calculate_extent(x_axis, y_axis)
        ax.imshow(field['fluence'].magnitude, extent=extent, aspect='auto', origin='lower')
        ax.set_xlabel(f'$x$, {x_axis.units}')
        ax.set_ylabel(f'$y$, {y_axis.units}')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        remove_grid(ax)

        ax = next(axes_iter)
        ax.plot(x_axis, field['fluence_x'].to('J/cm^2'))
        ax.set_xlabel(f'$x$, {x_axis.units}')
        ax.set_ylabel('Fluence, J/cm$^2$')
        ax.set_xlim(x_lim)
        ax.set_ylim(ymin=0)

        ax = next(axes_iter)
        ax.plot(y_axis, field['fluence_y'].to('J/cm^2'))
        ax.set_xlabel(f'$y$, {y_axis.units}')
        ax.set_ylabel('Fluence, J/cm$^2$')
        ax.set_xlim(y_lim)
        ax.set_ylim(ymin=0)
    else:
        ax = next(axes_iter)
        ax.plot(r_axis, field['fluence'].to('J/cm^2'))
        ax.set_xlabel(f'Radius, {r_axis.units}')
        ax.set_ylabel('Fluence, J/cm$^2$')
        ax.set_xlim(r_lim)
        ax.set_ylim(ymin=0)

    omega_rel = (omega_axis / omega0).m_as('')
    if is_3d:
        ax = next(axes_iter)
        extent = mpl_utils.calculate_extent(omega_rel, x_axis)
        ax.imshow(np.sqrt(field['spectral_intensity_x'].T.magnitude), extent=extent, aspect='auto', origin='lower')
        ax.set_xlim(omega_relative_lim)
        ax.set_ylim(x_lim)
        ax.set_xlabel(r'$\omega/\omega_0$')
        ax.set_ylabel(f'$x$, {x_axis.units}')
        remove_grid(ax)

        ax = next(axes_iter)
        extent = mpl_utils.calculate_extent(omega_rel, y_axis)
        ax.imshow(np.sqrt(field['spectral_intensity_y'].T.magnitude), extent=extent, aspect='auto', origin='lower')
        ax.set_xlim(omega_relative_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(r'$\omega/\omega_0$')
        ax.set_ylabel(f'$y$, {y_axis.units}')
        remove_grid(ax)
    else:
        ax = next(axes_iter)
        extent = mpl_utils.calculate_extent(omega_rel, r_axis)
        ax.imshow(np.sqrt(field['spectral_intensity'].T.magnitude), extent=extent, aspect='auto', origin='lower')
        ax.set_xlim(omega_relative_lim)
        ax.set_ylim(r_lim)
        ax.set_xlabel(r'$\omega/\omega_0$')
        ax.set_ylabel(f'Radius, {r_axis.units}')
        remove_grid(ax)

    ax = next(axes_iter)
    ax.plot(lambda_axis[lambda_axis > 0], field['spectral_power_lambda'][lambda_axis > 0].to('mJ/nm'))
    ax.set_xlim(lambda_lim)
    ax.set_ylim(ymin=0)
    ax.set_xlabel('Wavelength, nm')
    ax.set_ylabel('Spectral power, mJ/nm')

    fig.suptitle(figtitle)

    if return_figure:
        return fig, axes_grid


def analyze_field(
    scl: ScalarFieldEnvelope,
    *,
    r_units='um',
    plot_field=False,
    **plot_kwargs,
):
    from pic_utils.functions import fwhm

    is_3d = is_envelope_3d(scl)

    if not plot_field and plot_kwargs:
        raise ValueError(f'Additional parameters supplied: {plot_kwargs.keys()}, but plot_field is False')

    c = ureg['speed_of_light']
    eps0 = ureg['vacuum_permittivity']
    e = ureg['elementary_charge']
    m = ureg['electron_mass']

    E = scl.Field * ureg['V/m']

    t_axis = (scl.t * ureg.s).to('fs')

    k0 = scl.k0 * ureg['1/m']
    omega0 = scl.omega0 * ureg['1/s']
    if is_3d:
        x_axis = (scl.x * ureg.m).to(r_units)
        y_axis = (scl.y * ureg.m).to(r_units)
        dx = x_axis[1] - x_axis[0]
        dy = y_axis[1] - y_axis[0]
        ix_center = len(x_axis) // 2
        iy_center = len(y_axis) // 2
    else:
        r_axis = (scl.r * ureg.m).to(r_units)
        dr = r_axis[1] - r_axis[0]
    dt = t_axis[1] - t_axis[0]

    intensity = (c * eps0 * np.abs(E) ** 2 / 2).to('W/cm^2')
    if is_3d:
        intensity_x = intensity[:, :, iy_center]
        intensity_y = intensity[:, ix_center, :]
        intensity_axis = intensity_x[:, ix_center]
    else:
        intensity_axis = intensity[:, 0]

    max_field = np.max(np.abs(E)).to('V/m')
    a0 = (e * max_field / m / c / omega0).m_as('')

    if is_3d:
        power = np.trapz(np.trapz(intensity, dx=dy, axis=2), dx=dx, axis=1).to('TW')
    else:
        power = np.trapz(intensity * r_axis * 2 * np.pi, dx=dr).to('TW')
    energy = np.trapz(power, dx=dt).to('J')
    duration = fwhm(power, t_axis)
    on_axis_duration = fwhm(intensity_axis, t_axis)
    fluence = np.trapz(intensity, dx=dt, axis=0).to('J/cm^2')
    if is_3d:
        fluence_x = fluence[:, iy_center]
        fluence_y = fluence[ix_center, :]

    del E

    k_axis = np.fft.ifftshift(scl.k_freq) / ureg.m
    omega_axis = (c * k_axis).to('1/s')
    lambda_axis = (2 * np.pi / k_axis).to('nm')
    domega = omega_axis[1] - omega_axis[0]

    spectral_intensity = (
        np.pi * c * eps0 * np.abs(np.fft.ifftshift(scl.Field_ft, axes=0) * ureg['V/m'] / domega) ** 2
    ).to('J s / cm^2')
    if is_3d:
        spectral_intensity_x = spectral_intensity[:, :, iy_center]
        spectral_intensity_y = spectral_intensity[:, ix_center, :]
        spectral_power = np.trapz(np.trapz(spectral_intensity, dx=dy, axis=2), dx=dx, axis=1).to('J s')
    else:
        spectral_power = 2 * np.pi * np.trapz(spectral_intensity * r_axis, dx=dr).to('J s')
    spectral_power_lambda = (spectral_power * 2 * np.pi * c / lambda_axis**2).to('mJ/nm')

    output = {
        'envelope': scl,
        'k0': k0,
        'omega0': omega0,
        'a0': a0,
        'intensity_axis': intensity_axis,
        'spectral_power': spectral_power,
        'k': k_axis,
        'power': power,
        't': t_axis,
        'lambda': lambda_axis,
        'omega': omega_axis,
        'spectral_power_lambda': spectral_power_lambda,
        'fluence': fluence,
        'energy': energy,
        'max_field': max_field,
        'duration_global': duration,
        'duration_on_axis': on_axis_duration,
    }

    if is_3d:
        output['x'] = x_axis
        output['y'] = y_axis
        output['intensity_x'] = intensity_x
        output['intensity_y'] = intensity_y
        output['spectral_intensity_x'] = spectral_intensity_x
        output['spectral_intensity_y'] = spectral_intensity_y
        output['fluence_x'] = fluence_x
        output['fluence_y'] = fluence_y
        output['3d'] = True
    else:
        output['r'] = r_axis
        output['intensity'] = intensity
        output['spectral_intensity'] = spectral_intensity
        output['3d'] = False

    if plot_field:
        plot_field_func = globals()['plot_field']  # hack around shadowing by the local parameter
        fig, axes = plot_field_func(output, return_figure=True, **plot_kwargs)
        output['figure'] = fig
        output['axes'] = axes

    return output


def analyze_time_series(field_array, z_axis, t_axis, r_axis, k0):
    from pic_utils.functions import fwhm, fwhm_radial
    from tqdm.auto import tqdm

    ureg = z_axis._REGISTRY
    c = ureg['speed_of_light']

    z_size = len(z_axis)
    t_size = len(t_axis)
    r_size = len(r_axis)

    intensity_axis = np.ones((z_size, t_size))
    max_intensity = np.ones(z_size)
    max_intensity_transverse = np.ones((z_size, r_size))
    a0 = np.zeros(z_size)
    fluence = np.ones((z_size, r_size))
    power = np.ones((z_size, t_size))
    duration_axis = np.ones(z_size)
    peak_position_axis = np.ones(z_size)
    duration = np.ones(z_size)
    energy = np.zeros(z_size)
    r_fwhm = np.zeros(z_size)
    w0 = np.zeros(z_size)

    for i, z in enumerate(tqdm(z_axis)):
        t_loc = (t_axis.min() + z_axis[i] / c).to('s').magnitude

        sclDiag = ScalarFieldEnvelope(k0=k0, t_axis=t_axis.m_as('s'), n_dump=4)
        sclDiag.import_field_ft(field_array[i], t_loc=t_loc, r_axis=r_axis.m_as('m'), transform=True)
        analysis = analyze_field(sclDiag)

        max_intensity_transverse[i] = np.max(analysis['intensity'], axis=0).m_as('W/cm^2')

        max_intensity[i] = np.max(analysis['intensity']).m_as('W/cm^2')
        intensity_axis[i] = analysis['intensity_axis'].m_as('W/cm^2')

        fluence[i] = analysis['fluence'].m_as('J/cm^2')
        power[i] = analysis['power'].m_as('TW')

        duration[i] = fwhm(power[i], t_axis).m_as('fs')
        duration_axis[i] = fwhm(intensity_axis[i], t_axis).m_as('fs')

        peak_position_axis[i] = t_axis[np.argmax(intensity_axis[i])].m_as('fs')

        a0[i] = analysis['a0']
        energy[i] = analysis['energy'].m_as('J')
        r_fwhm[i] = fwhm_radial(fluence[i], r_axis).m_as('um')
        w0[i] = r_fwhm[i] / np.sqrt(2 * np.log(2))

    max_intensity = max_intensity * ureg['W/cm^2']
    intensity_axis = intensity_axis * ureg['W/cm^2']
    fluence = fluence * ureg['J/cm^2']
    power = power * ureg['TW']
    duration = duration * ureg['fs']
    duration_axis = duration_axis * ureg['fs']
    peak_position_axis = peak_position_axis * ureg['fs']
    energy = energy * ureg['J']
    r_fwhm = r_fwhm * ureg['um']
    w0 = w0 * ureg['um']

    res = {
        'z': z_axis,
        'r': r_axis,
        't': t_axis,
        'k0': k0 / ureg.m,
        'field': field_array,
        'a0': a0,
        'max_intensity': max_intensity,
        'intensity_axis': intensity_axis,
        'fluence': fluence,
        'power': power,
        'energy': energy,
        'duration': duration,
        'duration_axis': duration_axis,
        'peak_position_axis': peak_position_axis,
        'r_fwhm': r_fwhm,
        'w0': w0,
    }

    res['omega0'] = (res['k0'] * c).to('1/s')
    res['lambda0'] = (2 * np.pi / res['k0']).to('um')

    return res


def create_symmetric_propagator(envelope: ScalarFieldEnvelope) -> 'PropagatorSymmetric':
    from axiprop.lib import PropagatorSymmetric

    return PropagatorSymmetric((np.max(envelope.r), envelope.r.size), envelope.k_freq)


def create_resampling_propagator(
    envelope: ScalarFieldEnvelope, R_new: pint.Quantity, Nr_new: int
) -> 'PropagatorResampling':
    from axiprop.lib import PropagatorResampling

    return PropagatorResampling(r_axis=envelope.r, kz_axis=envelope.k_freq, r_axis_new=(R_new.m_as('m'), Nr_new))


def calculate_time_series(envelope: ScalarFieldEnvelope, z_axis, prop=None):
    ureg = z_axis._REGISTRY

    if prop is None:
        prop = create_symmetric_propagator(envelope)

    field_propagated = prop.steps(envelope.Field_ft, z_axis=z_axis.m_as('m'))
    t_axis = envelope.t * ureg.s
    r_axis = prop.r_new * ureg.m

    return analyze_time_series(field_propagated, z_axis, t_axis, r_axis, envelope.k0)


def envelope_from_time_series(time_series: dict, iteration):
    ureg = time_series['z']._REGISTRY
    c = ureg['speed_of_light']

    k0 = time_series['k0'].m_as('1/m')
    t_loc = (time_series['t'].min() + time_series['z'][iteration] / c).m_as('s')
    r_axis = time_series['r'].m_as('m')
    t_axis = time_series['t'].m_as('s')

    scl = ScalarFieldEnvelope(k0=k0, t_axis=t_axis, n_dump=4)
    scl.import_field_ft(time_series['field'][iteration], t_loc=t_loc, r_axis=r_axis, transform=True)
    return scl


def apply_spectral_multiplier(envelope: ScalarFieldEnvelope, multiplier: np.ndarray) -> ScalarFieldEnvelope:
    """Apply spectral multiplier to the envelope.

    Parameters
    ----------
    envelope : ScalarFieldEnvelope
        the envelope to be modified
    multiplier : np.ndarray
        the spectral multiplier array

    Returns
    -------
    ScalarFieldEnvelope
        The new envelope with the applied multiplier
    """
    envelope_new = ScalarFieldEnvelope(k0=envelope.k0, t_axis=envelope.t, n_dump=envelope.n_dump)
    envelope_new.import_field_ft(envelope.Field_ft * multiplier, r_axis=envelope.r, transform=True)
    return envelope_new


def save_to_lasy(envelope: ScalarFieldEnvelope, filename: str | Path):
    """Save an envelope to the lasy envelope format.

    Parameters
    ----------
    envelope : ScalarFieldEnvelope
        the envelope to be saved
    filename : str | Path
        The name of the file.
        If it doesn't have an extension, .h5 will be assumed.

    Raises
    ------
    FileNotFoundError
        if lasy fails to write an expected file.
    """
    filename = Path(filename)
    write_dir = filename.parent
    if filename.suffix:
        format = filename.suffix[1:]
        prefix = filename.stem
    else:
        prefix = 'h5'
        filename = filename.name
    expected_lasy_filename = write_dir / f'{prefix}_00000.{format}'
    final_filename = write_dir / f'{prefix}.{format}'

    from lasy.laser import Laser
    from lasy.profiles.gaussian_profile import GaussianProfile
    from scipy.constants import speed_of_light

    t_peak = 0.0e-15  # s
    pol = (1, 0)
    lambda0 = 2 * np.pi * speed_of_light / envelope.omega0  # m

    gaussian = GaussianProfile(lambda0, pol, 1.0, 30e-6, 60e-15, t_peak)

    dim = 'rt'
    lo = (envelope.r.min(), envelope.t.min())
    hi = (envelope.r.max(), envelope.t.max())
    npoints = (envelope.r.size, envelope.t.size)

    laser = Laser(dim, lo, hi, npoints, gaussian)
    laser.grid.temporal_field[0] = envelope.Field.T.copy()

    laser.write_to_file(prefix, file_format=format, write_dir=write_dir.resolve())

    if expected_lasy_filename.exists():
        expected_lasy_filename.rename(final_filename)
    else:
        raise FileNotFoundError(f'File {expected_lasy_filename} was not created by the lasy writer')


def read_from_lasy(
    filename: Path | str, energy_coeff: float = 1.0, n_dump: int = 4, attenuate_boundaries: bool = True
) -> ScalarFieldEnvelope:
    """Read a scalar field envelope from a lasy file.

    Parameters
    ----------
    filename : Path | str
        path to the input file
    energy_coeff : float, optional
        factor to adjust the energy after reading, by default 1.0
    n_dump : int, optional
        number of cells to be used for attenuating boundaries, by default 4
    attenuate_boundaries : bool, optional
        whether to apply boundary attenuation after reading. Should be set to False if the envelope already has attenuation applied (e.g. it was generated by save_to_lasy), by default True

    Returns
    -------
    ScalarFieldEnvelope
        the resulting envelope
    """
    import h5py
    from scipy.constants import speed_of_light

    with h5py.File(filename, mode='r') as f:
        dset = f['/data/0/meshes/laserEnvelope']
        env = np.array(dset)
        env = env[0] * np.sqrt(energy_coeff)
        nt, nr = env.shape
        dt, dr = dset.attrs['gridSpacing'] * dset.attrs['gridUnitSI']
        omega = dset.attrs['angularFrequency']
        t_min_lasy = dset.attrs['gridGlobalOffset'][0]

    t_axis = np.linspace(t_min_lasy, t_min_lasy + dt * (nt - 1), nt)
    r_axis = np.linspace(0, dr * (nr - 1), nr)

    scl = ScalarFieldEnvelope(omega / speed_of_light, t_axis, n_dump)

    if attenuate_boundaries:
        scl.import_field(env, r_axis=r_axis)
    else:
        scl.import_field(env, r_axis=r_axis, transform=False)
        scl.time_to_frequency()

    return scl


def add_pulse_front_curvature(envelope: ScalarFieldEnvelope, pfc):
    pfc_phase = np.exp(1j * pfc.m_as('s/m^2') * envelope.r**2 * (envelope.omega - envelope.omega0)[:, np.newaxis])
    field = envelope.Field_ft * pfc_phase

    return ScalarFieldEnvelope(envelope.k0, envelope.t, envelope.n_dump).import_field_ft(
        field, r_axis=envelope.r, transform=True
    )


def propagate_envelope(
    envelope: ScalarFieldEnvelope, distance: pint.Quantity, propagator: 'PropagatorResampling'
) -> ScalarFieldEnvelope:
    """Propagate an envelope to a specific coordinate.

    Parameters
    ----------
    envelope : ScalarFieldEnvelope
        Envelope to be propagated
    distance : pint.Quantity
        Propagation distance
    propagator : PropagatorResampling
        The propagator

    Returns
    -------
    ScalarFieldEnvelope
        Envelope after the propagation.
    """
    c = ureg['speed_of_light']

    t_loc = envelope.t.min() + (distance / c).m_as('s')
    E_propagated = propagator.step(envelope.Field_ft, distance.m_as('m'))

    new_envelope = ScalarFieldEnvelope(envelope.k0, envelope.t, envelope.n_dump)
    new_envelope.import_field_ft(E_propagated, t_loc=t_loc, r_axis=propagator.r_new, transform=True)
    return new_envelope


def calculate_fluence_radius(envelope: ScalarFieldEnvelope | dict, threshold: float) -> pint.Quantity:
    """Calculate the radius at which the fluence drops below a certain threshold.

    Only works for 2D envelopes.

    Parameters
    ----------
    envelope : ScalarFieldEnvelope | dict
        The envelope or the result of `analyze_field` run on it.
    threshold : float
        The threshold for the fluence.

    Returns
    -------
    pint.Quantity
        The radius at which the fluence drops below the threshold.

    Raises
    ------
    NotImplementedError
        If the envelope is 3D.
    """
    if isinstance(envelope, ScalarFieldEnvelope):
        envelope = analyze_field(envelope)

    if envelope['3d']:
        raise NotImplementedError('Fluence radius can only be calculated for 2D envelopes')

    r = envelope['r']
    fluence = envelope['fluence']

    max_fluence = np.max(fluence)

    return r[::-1][np.argmax(fluence[::-1] > threshold * max_fluence)]
