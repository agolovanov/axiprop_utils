# Resampling benchmark

Run on a GPU node with Axiprop and CuPy installed:

```bash
python benchmarks/resampling.py \
    --nr 1984:2112:16,2047,2049 --nr-new 1024 \
    --output benchmarks/resampling.csv
```

Both size arguments accept comma-separated integers and inclusive
`start:stop[:step]` ranges. Every input/output combination is measured. For example:

```bash
python benchmarks/resampling.py \
    --nr 2047:2049 --nr-new 1008:1040:16 \
    --nkz 8 --setup-repeats 3 --repeats 7 --iterations 20 \
    --output benchmarks/resampling_sweep.csv
```

The default sweep includes 2047, 2048, and 2049 with fixed `Nr_new=1024`.
The default backend is **CU** (CuPy/CUDA), using complex128. Select another
installed Axiprop backend with `--backend` (`CL`, `AF`, `NP`, `NP_MKL`, or
`NP_FFTW`). An unavailable backend is an error; there is no fallback.
GPU work is synchronized at timing boundaries. CUDA uses the current device;
use `CUDA_VISIBLE_DEVICES` to choose a GPU. Case order is shuffled reproducibly
to reduce ordering bias (`--seed`).

The field is a Gaussian amplitude with the quadratic reflected phase of a
parabolic mirror, propagated to its focal length. All physical parameters are
in SI units and have command-line options (`--help`). The default uniform input
grid resembles sampled envelope data; `--grid bessel` selects Bessel-root
sampling. Output sampling is uniform. The output radius must not exceed the
input radius, so padding does not change the effective transform size.

CSV results report median setup time, median and minimum forward/inverse
transform times, full propagation time, transform matrix storage, and peak
output intensity as a basic diagnostic. Transform timings are for one
frequency; full propagation timings include all `--nkz` frequencies, allocations,
copies within the backend, and propagation phases. The input field is uploaded
before timing, and propagation returns a device array on GPU backends; transfers
to and from the host are excluded. Each timing sample averages `--iterations`
calls after a warm-up. Setup includes grid creation, Bessel matrix evaluation,
inversion on the host (including for CU), device transfers, and buffer allocation;
it defaults to one measurement because it can be expensive. Increase
`--setup-repeats` to compare setup more reliably.
A small, untimed propagator initializes numerical-library routines before the
sweep. Each propagation starts from the same input field.

A sibling JSON file records raw timing samples, arguments, package versions,
hostname, CUDA device and versions for CU, CPU affinity, thread environment,
NumPy build configuration, and case order. Results are flushed after each case.
Matrix storage excludes temporary setup allocations and
is not a peak memory measurement. The finite-output check and peak intensity
are diagnostics, not a convergence or physical-accuracy test.
