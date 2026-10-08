# Third-party software

The project code is licensed under GNU GPL version 3 only (`GPL-3.0-only`);
see `LICENSE`. Third-party code retains its own copyright and license notices.

## Vendored Eigen

`vendor/eigen/` contains Eigen 5.0.0 headers, used to build the C++ extension.
Upstream: <https://gitlab.com/libeigen/eigen>.

Eigen is licensed under the Mozilla Public License 2.0. The full license is
included in `vendor/eigen/COPYING.MPL2`. Individual headers retain their
upstream copyright and any additional license notices. These headers are
included in source distributions; the MPL license is also included in wheels.

## Separately installed dependencies

Python dependencies are installed separately and are not copied into this
repository. They retain their upstream licenses, including:

- TRexSelectorNeo: GPL-3.0-or-later, <https://github.com/G4Lactus/TRexSelector>.
- knockpy: MIT, <https://github.com/amspector100/knockpy>.
- NumPy and SciPy: BSD-style licenses, with additional notices for bundled code.
- Matplotlib: the Matplotlib license, based on the Python Software Foundation license.
- joblib: BSD-3-Clause.
- tqdm: MPL-2.0 and MIT, according to the relevant files.
- pybind11 (build dependency): BSD-3-Clause.

Consult the installed distributions for their complete notices, including
transitive dependencies. BLAS and optional OpenMP libraries are supplied by
the build/runtime environment and retain their respective licenses.
