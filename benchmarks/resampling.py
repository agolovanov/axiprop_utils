"""Benchmark Axiprop's resampling DHT for a focused Gaussian beam (default: CU)."""

import argparse
import csv
import gc
import importlib.metadata
import io
import json
import os
import platform
import random
from contextlib import redirect_stdout
from pathlib import Path
from statistics import median
from time import perf_counter


def sizes(value):
    """Parse comma-separated sizes and inclusive start:stop[:step] ranges."""
    result = []
    try:
        for part in value.split(','):
            if ':' in part:
                bounds = [int(item) for item in part.split(':')]
                if len(bounds) not in (2, 3):
                    raise ValueError
                start, stop = bounds[:2]
                step = bounds[2] if len(bounds) == 3 else 1
                if start < 2 or stop < start or step < 1:
                    raise ValueError
                result.extend(range(start, stop + 1, step))
            else:
                result.append(int(part))
        if not result or min(result) < 2:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use sizes >= 2, e.g. 2047,2048,2049 or 1920:2176:32') from exc
    return list(dict.fromkeys(result))


def positive_int(value):
    """Parse a strictly positive integer."""
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return result


def positive_float(value):
    """Parse a finite, strictly positive physical parameter."""
    import math

    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('Must be finite and positive')
    return result


def synchronize(prop):
    """Finish pending backend work, including lazy ArrayFire evaluation."""
    backend = prop.bcknd
    if backend.name == 'CU':
        backend.cp.cuda.get_current_stream().synchronize()
    elif backend.name == 'CL':
        backend.queue.finish()
    elif backend.name == 'AF':
        backend.af.eval(prop.u_ht, prop.u_iht)
        backend.af.sync()


def timings(operation, prop, repeats, iterations):
    """Return seconds per call from repeated batches after one warm-up call."""
    operation()
    synchronize(prop)
    samples = []
    for _ in range(repeats):
        synchronize(prop)
        start = perf_counter()
        for _ in range(iterations):
            result = operation()
            if prop.bcknd.name == 'AF':
                prop.bcknd.af.eval(prop.u_ht, prop.u_iht)
                if result is not None:
                    prop.bcknd.af.eval(result)
        synchronize(prop)
        samples.append((perf_counter() - start) / iterations)
    return samples


def main():
    """Run every input/output size combination and save timings with metadata."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nr', type=sizes, default=sizes('1984,2016,2047,2048,2049,2080,2112'))
    parser.add_argument('--nr-new', type=sizes, default=sizes('1024'))
    parser.add_argument('--backend', default='CU', help='Axiprop backend (CU, CL, AF, NP, NP_MKL, NP_FFTW)')
    parser.add_argument('--nkz', type=positive_int, default=1, help='Number of frequencies per propagation call')
    parser.add_argument('--repeats', type=positive_int, default=5)
    parser.add_argument('--iterations', type=positive_int, default=10, help='Calls per timing batch')
    parser.add_argument('--setup-repeats', type=positive_int, default=1)
    parser.add_argument('--radius', type=positive_float, default=3e-3, help='Input domain radius (m)')
    parser.add_argument('--radius-new', type=positive_float, default=100e-6, help='Output domain radius (m)')
    parser.add_argument('--waist', type=positive_float, default=1e-3, help='Gaussian amplitude 1/e radius (m)')
    parser.add_argument('--wavelength', type=positive_float, default=800e-9, help='Central wavelength (m)')
    parser.add_argument('--focal-length', type=positive_float, default=0.1, help='Parabolic mirror focal length (m)')
    parser.add_argument('--grid', choices=['uniform', 'bessel'], default='uniform', help='Input radial sampling')
    parser.add_argument('--seed', type=int, default=0, help='Seed for shuffled case order')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/resampling.csv'))
    args = parser.parse_args()
    if args.radius_new > args.radius:
        parser.error('--radius-new must be <= --radius (this benchmark excludes input padding)')

    import numpy as np
    from axiprop.backends import AVAILABLE_BACKENDS
    from axiprop.lib import PropagatorResampling

    if args.backend not in AVAILABLE_BACKENDS:
        parser.error(f'Backend {args.backend!r} unavailable; installed backends: {", ".join(AVAILABLE_BACKENDS)}')
    k0 = 2 * np.pi / args.wavelength
    kz = k0 * (np.linspace(0.99, 1.01, args.nkz) if args.nkz > 1 else np.ones(1))
    cases = [(nr, nr_new) for nr in args.nr for nr_new in args.nr_new]
    random.Random(args.seed).shuffle(cases)

    # Initialize lazy numerical-library routines before measuring setup.
    warmup = PropagatorResampling(
        r_axis=(args.radius, 32),
        kz_axis=kz[:1],
        r_axis_new=(args.radius_new, 32),
        r_axes_types=(args.grid, 'uniform'),
        backend=args.backend,
        dtype=np.complex128,
        verbose=False,
    )
    warmup.step(warmup.bcknd.to_device(np.ones((1, 32), dtype=np.complex128)), args.focal_length)
    synchronize(warmup)
    device_metadata = {}
    if args.backend == 'CU':
        cp = warmup.bcknd.cp
        device_id = cp.cuda.runtime.getDevice()
        device_metadata = {
            'cupy_version': cp.__version__,
            'device_id': device_id,
            'device_name': cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode(),
            'cuda_runtime_version': cp.cuda.runtime.runtimeGetVersion(),
            'cuda_driver_version': cp.cuda.runtime.driverGetVersion(),
        }
    del warmup
    gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
    metadata = {
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in ('axiprop', 'numpy', 'scipy')},
        'numpy_config': config.getvalue(),
        'device': device_metadata,
        'thread_environment': {
            name: os.environ.get(name)
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS')
        },
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'case_order': cases,
        'raw_timings_s': [],
    }
    metadata_path = args.output.with_suffix('.json')
    columns = [
        'nr',
        'nr_new',
        'nkz',
        'backend',
        'setup_median_s',
        'forward_median_ms',
        'forward_min_ms',
        'inverse_median_ms',
        'inverse_min_ms',
        'step_median_ms',
        'step_min_ms',
        'matrix_mib',
        'peak_intensity',
    ]
    print(f'Backend {args.backend}; {len(cases)} cases', flush=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for nr, nr_new in cases:
            print(f'Nr={nr}, Nr_new={nr_new}: setting up ...', flush=True)
            setup = []
            for _ in range(args.setup_repeats):
                start = perf_counter()
                prop = PropagatorResampling(
                    r_axis=(args.radius, nr),
                    kz_axis=kz,
                    r_axis_new=(args.radius_new, nr_new),
                    r_axes_types=(args.grid, 'uniform'),
                    backend=args.backend,
                    dtype=np.complex128,
                    verbose=False,
                )
                synchronize(prop)
                setup.append(perf_counter() - start)
                if len(setup) < args.setup_repeats:
                    del prop
                    gc.collect()

            # exp(-r²/w²) amplitude and the reflected phase of sag r²/(4f).
            field = np.exp(
                -((prop.r[None, :] / args.waist) ** 2)
                - 1j * kz[:, None] * prop.r[None, :] ** 2 / (2 * args.focal_length)
            ).astype(np.complex128)
            field = prop.bcknd.to_device(field)
            prop.u_loc = field[args.nkz // 2].copy()
            forward = timings(prop.TST, prop, args.repeats, args.iterations)
            inverse = timings(prop.iTST, prop, args.repeats, args.iterations)
            # Every call propagates the original beam independently to the focus.
            step = timings(lambda: prop.step(field, args.focal_length), prop, args.repeats, args.iterations)
            focused = prop.bcknd.to_host(prop.step(field, args.focal_length))
            if not np.isfinite(focused).all():
                raise RuntimeError(f'Nonfinite output for Nr={nr}, Nr_new={nr_new}')
            row = {
                'nr': nr,
                'nr_new': nr_new,
                'nkz': args.nkz,
                'backend': args.backend,
                'setup_median_s': median(setup),
                'matrix_mib': (nr * nr + nr_new * nr) * np.dtype(np.complex128).itemsize / 2**20,
                'peak_intensity': float(np.max(np.abs(focused) ** 2)),
            }
            for name, samples in [('forward', forward), ('inverse', inverse), ('step', step)]:
                row[f'{name}_median_ms'] = 1e3 * median(samples)
                row[f'{name}_min_ms'] = 1e3 * min(samples)
            writer.writerow(row)
            stream.flush()
            metadata['raw_timings_s'].append(
                {'nr': nr, 'nr_new': nr_new, 'setup': setup, 'forward': forward, 'inverse': inverse, 'step': step}
            )
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
            print(
                f'  setup {row["setup_median_s"]:.3f} s; '
                f'DHT {row["forward_median_ms"]:.3f} ms; '
                f'iDHT {row["inverse_median_ms"]:.3f} ms; '
                f'step {row["step_median_ms"]:.3f} ms',
                flush=True,
            )
            del prop, field, focused
            gc.collect()
    print(f'Saved {args.output} and {metadata_path}', flush=True)


if __name__ == '__main__':
    main()
