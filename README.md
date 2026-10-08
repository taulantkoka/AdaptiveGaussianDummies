# ICASSP 2027 reproducibility package

This archive reproduces the simulation results in **Scalable Variable
Selection under Predictor Dependence with Adaptive Virtual Dummies**. It
contains the Sphere AG and shared lazy Gram C++ implementation, the manuscript
experiment drivers, the minimal helper modules they import, and the numerical
summaries used in the submitted paper.

## Contents

- `src/`, `python/vd_selectors/`: virtual-dummy T-Rex, Sphere AG, OMP, LARS, and the
  shared lazy Gram cache.
- `experiments/paper_benchmark.py`: SNR and dependence sweeps for Fig. 2 and
  the compact Sphere AG versus spherical T-Rex comparison.
- `experiments/combine_selector_benchmarks.py`: assembles the two manuscript
  figures from component Monte Carlo runs.
- `experiments/paper_method_timing.py`: end-to-end method timing in Table 1.
- `experiments/paper_shared_gram_scaling.py`: direct versus shared Gram timing
  over joint `(n,p)` and random-experiment-count sweeps in Table 1.
- `experiments/paper_large_scale_synthetic.py`: the `p=100000` feasibility
  study in Table 2.
- `reference_results/`: compact CSV summaries underlying the reported figures
  and tables. Raw Monte Carlo checkpoints are intentionally omitted.
- `reproduce.py`: resumable entry point for smoke tests and full experiments.

Historical tail-trimming studies, NHG diagnostics, HAPNEST experiments,
notebooks, compiled binaries, and unrelated project code are not included.

## Installation

The package requires Python 3.12 or newer (the pinned SciPy version requires
3.12+) and is tested with Python 3.13 on macOS. The local extension requires
a C++17 compiler, CMake 3.20 or newer, and a BLAS implementation. The T-Rex+DA
dependency additionally requires Git, a C++20 compiler, and CMake 3.24 or newer.
OpenMP is optional for
correctness but required to reproduce the ten-thread timing protocol. On
Apple Silicon, install `libomp` with Homebrew before building if multithreaded
timings are desired. The Xcode command-line tools must be installed and their
license accepted before compiling on macOS.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

The install builds the local `vd_selectors` extension and installs the pinned
direct Python dependencies, including `knockpy==1.3.5`. `TRexSelectorNeo`
is built from the public [upstream repository](https://github.com/G4Lactus/TRexSelector)
at commit `c1edbfbea83ffb44eb4fcebc87b2888d3e5abb70`; version 0.2.0 is not
available from PyPI. Building that dependency also downloads its upstream
C++ dependencies, so the initial installation requires internet access.
Eigen headers for the local extension are included under `vendor/`.
Transitive Python dependencies are resolved by pip and are not fully locked.

## Quick validation

```bash
python reproduce.py --stage smoke

# Additional numerical and regression checks
python -m pip install '.[dev]'
python -m pytest -q
python -m pip check
```

The smoke stage performs small paired runs involving Sphere AG, spherical
T-Rex, T-Rex+DA, Model-X, and Model-X+, exercises the timing driver with OMP
and LARS, verifies direct/shared selected-set and occurrence-matrix equality,
exercises the large-scale driver at a tiny dimension with both solvers, and
regenerates both manuscript figures from the included reference summary.
Recorded method failures cause the smoke stage to fail. Outputs are written
under `results/generated/` and `figures/generated/`; these, the virtual
environment, and build/cache files are excluded by `.gitignore`.
The smoke stage checks functionality, not the paper's full statistical claims.

To replot the submitted figure without running a simulation:

```bash
python reproduce.py --stage reference
```

## Full manuscript experiments

The stages are independent and resumable. The drivers checkpoint raw rows after
completed Monte Carlo cells. Omit `--recompute` to continue an interrupted run.

```bash
# Fig. 1 and Fig. 2: three covariance structures, OMP/LARS,
# T-Rex+DA, Model-X, and Model-X+.
python reproduce.py --stage figures --trials 100 --jobs -1

# Table 1, end-to-end selector timing, one thread.
python reproduce.py --stage timing --trials 100

# Table 1, direct versus shared Gram timing, ten selector threads.
python reproduce.py --stage scaling --trials 100

# Table 2, n=1000 and p=100000, ten selector threads.
python reproduce.py --stage large --trials 100
```

Run all four stages sequentially with:

```bash
python reproduce.py --stage all --trials 100 --jobs -1
```

The figure benchmark uses paired datasets and seeds at every operating point:

- block-Toeplitz: SNR sweep at `rho=0.9`, dependence sweep through `rho=0.9`;
- Student-`t_3` loaded blocks: SNR sweep at `rho=0.4`, dependence sweep through
  `rho=0.5`;
- grouped factor: SNR sweep at `rho=0.3`, dependence sweep through `rho=0.5`.

All use `n=500`, `p=1000`, `p1=10`, `B=20`, `L=3p`, target FDR `0.2`, and
100 paired trials. Model-X and Model-X+ share one Gaussian knockoff draw and
Lasso statistic. T-Rex+DA uses the block-tree method from
`TRexSelectorNeo==0.2.0`.

The cache-scaling experiment uses fixed `T=5`, as stated in the manuscript.
The selector benchmarks and the large-scale study use the standard T-Rex
calibrated-`T` search. The statistical results are seed reproducible; wall-clock
times are machine dependent. The manuscript timing reference machine was a
64 GB, 10-core Apple M1 Max. BLAS thread counts are fixed to one by
`reproduce.py`; the C++ selector thread count is set separately by each stage.

## Reference outputs

The files in `reference_results/` contain exactly the aggregate values used in
the manuscript:

- `final_selector_comparison.summary.csv`
- `paper_method_timing.summary.csv`
- `shared_gram_scaling.summary.csv`
- `large_scale.summary.csv`

Small floating-point differences across compilers and BLAS libraries are
possible. Timing values should only be compared under a matched machine and
thread configuration.

## License and third-party software

The project source is distributed under the **GNU General Public License,
version 3 only** (`GPL-3.0-only`); see [LICENSE](LICENSE). This permits use,
modification, and redistribution, including commercial use; distributed
derivative works must comply with GPLv3's source and licensing requirements.
See the [GNU GPLv3 text](https://www.gnu.org/licenses/gpl-3.0.html).

The vendored Eigen headers retain their MPL-2.0 license and upstream notices;
see `vendor/eigen/COPYING.MPL2`. T-Rex+DA and Model-X are separately installed
dependencies. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## Source archive checksums

`MANIFEST.sha256` covers the repository's source and reference files, excluding
itself and ignored/generated files. Verify it with `shasum -a 256 -c
MANIFEST.sha256` on macOS or `sha256sum -c MANIFEST.sha256` on Linux. After
intentional source changes, regenerate it with `python tools/update_manifest.py`.
