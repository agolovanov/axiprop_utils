# Repository Guidelines

## Project Structure & Module Organization

`axiprop_utils/` contains the importable Python package. General field-analysis and
plotting helpers live in `axiprop_utils/__init__.py`; pulse construction and
manipulation belong in `pulse.py`, axiparabola optics in `axiparabola.py`, and
small shared helpers in `utils.py`. Tests are under `test/`, grouped by geometry
or feature (`test_2d.py`, `test_3d.py`, `test_lasy.py`, and
`test_axiparabola.py`). Packaging metadata is in `pyproject.toml`, while Ruff
configuration is kept in `ruff.toml`. The repository has no separate static
asset directory.

## Build, Test, and Development Commands

Use Python 3.10 or newer in a virtual environment. Runtime dependencies are
installed with the package; use the `dev` extra for contributor tooling.

- `python -m pip install -e ".[dev]"` installs the package and development tools
  in editable mode.
- `python -m pytest` runs the complete test suite.
- `python -m pytest test/test_axiparabola.py -q` runs one focused module.
- `ruff check .` checks NumPy-style docstrings and configured lint rules.
- `ruff format --check .` verifies formatting; `ruff format .` applies it.
- `python -m build` creates source and wheel distributions when `build` is
  installed.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` for
functions and variables, and `PascalCase` for classes. Ruff enforces single
quotes, a 120-character line length, and NumPy docstring style. Add type hints
where they improve public API clarity. Preserve physical units with
`pint.Quantity`; convert to dimensionless or explicit units only at numerical
library boundaries. Keep public imports intentional when exposing helpers from
`axiprop_utils`.

## Testing Guidelines

Tests use pytest and NumPy assertions. Name files `test_*.py` and test functions
`test_<behavior>`. Cover both returned values and physical invariants such as
energy conservation, units, symmetry, or limiting cases. For floating-point
results, use `numpy.testing.assert_allclose` with a justified tolerance. Run the
full suite before submitting changes; no formal coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, sentence-case subjects such as `Add
apply_radial_delay function` and `Fix time series for length of 1`. Keep each
commit focused. Pull requests should explain the motivation, summarize API or
numerical changes, list tests run, and link relevant issues. Include plots or
screenshots only when visualization behavior changes, and call out compatibility
or dependency changes explicitly.
