#!/usr/bin/env python3
"""Run the ICASSP experiments in resumable, independently selectable stages."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
GENERATED_RESULTS = ROOT / "results" / "generated"
GENERATED_FIGURES = ROOT / "figures" / "generated"


def _run(script: str, arguments: list[str], env: dict[str, str]) -> None:
    command = [sys.executable, str(EXPERIMENTS / script), *arguments]
    print("\n+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _latest_raw(prefix: Path) -> Path:
    matches = list(prefix.parent.glob(prefix.name + ".*.raw.csv"))
    if not matches:
        raise FileNotFoundError(f"no raw result found for {prefix}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _validate_raw(prefix: Path, expected_rows: int) -> None:
    path = _latest_raw(prefix)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path}: expected {expected_rows} rows, found {len(rows)}")
    failures = [row["error"] for row in rows if row.get("error")]
    if failures:
        raise RuntimeError(f"{path}: {len(failures)} failed calls: {failures}")


def _benchmark_arguments(
    args: argparse.Namespace,
    structure: str,
    solver: str,
    output: Path,
) -> list[str]:
    settings = {
        "block_toeplitz": {
            "rhos": ["0", "0.3", "0.5", "0.7", "0.8", "0.9"],
            "snr_rho": "0.9",
            "extra": [],
        },
        "block": {
            "rhos": ["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
            "snr_rho": "0.4",
            "extra": ["--block-loading-law", "student_t", "--block-loading-df", "3"],
        },
        "factor": {
            "rhos": ["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
            "snr_rho": "0.3",
            "extra": [],
        },
    }[structure]
    methods = ["Sphere AG", "Spherical T-Rex", "T-Rex+DA"]
    if solver == "omp":
        methods += ["Model-X", "Model-X+"]
    result = [
        "--n", "500", "--p", "1000", "--p1", "10",
        "--alpha", "0.2", "--K", "20", "--L", "3",
        "--solver", solver,
        "--sweeps", "rho", "snr",
        "--structures", structure,
        "--rhos", *settings["rhos"],
        "--snrs", "0.25", "0.5", "1", "2", "4", "8",
        "--rho-snr", "1", "--snr-rho", settings["snr_rho"],
        "--block-size", "100",
        "--factor-groups", "10", "--factor-global-share", "0.25",
        "--methods", *methods,
        "--da-method", "trex+DA+BT", "--mx-fstat", "lasso",
        "--n-mc", str(args.trials), "--n-jobs", str(args.jobs),
        "--seed", "72000",
        "--out", str(output),
        "--plot-prefix", str(GENERATED_FIGURES / output.name),
        *settings["extra"],
    ]
    if args.recompute:
        result.append("--recompute")
    return result


def figures(args: argparse.Namespace, env: dict[str, str]) -> None:
    sources: dict[tuple[str, str], Path] = {}
    for structure in ("block_toeplitz", "block", "factor"):
        for solver in ("omp", "lars"):
            prefix = GENERATED_RESULTS / f"{args.tag}_{structure}_{solver}"
            _run(
                "paper_benchmark.py",
                _benchmark_arguments(args, structure, solver, prefix),
                env,
            )
            sources[(structure, solver)] = _latest_raw(prefix)

    combine_arguments: list[str] = []
    for structure in ("block_toeplitz", "block", "factor"):
        combine_arguments += [
            "--raw-source", f"omp={sources[(structure, 'omp')]}",
            "--raw-source", f"lars={sources[(structure, 'lars')]}",
            "--raw-source", f"knockoff={sources[(structure, 'omp')]}",
        ]
    combine_arguments += [
        "--plot", str(GENERATED_FIGURES / "final_selector_comparison.pdf"),
        "--comparison-plot", str(GENERATED_FIGURES / "sphere_uniform_comparison.pdf"),
        "--summary", str(GENERATED_RESULTS / "final_selector_comparison.summary.csv"),
    ]
    _run("combine_selector_benchmarks.py", combine_arguments, env)


def timing(args: argparse.Namespace, env: dict[str, str]) -> None:
    command = [
        "--n", "1000", "--p", "2500", "--p1", "10",
        "--structure", "block_toeplitz", "--rho", "0.9", "--snr", "1",
        "--alpha", "0.2", "--K", "20", "--L", "3",
        "--methods", "Sphere AG", "Spherical T-Rex", "T-Rex+DA", "Model-X pair",
        "--solvers", "omp", "lars",
        "--repeats", str(args.trials), "--seed", "98000", "--n-threads", "1",
        "--out", str(GENERATED_RESULTS / f"{args.tag}_method_timing"),
        "--plot", str(GENERATED_FIGURES / "method_timing.pdf"),
    ]
    if args.recompute:
        command.append("--recompute")
    _run("paper_method_timing.py", command, env)


def scaling(args: argparse.Namespace, env: dict[str, str]) -> None:
    command = [
        "--sweeps", "joint", "K",
        "--joint-pairs", "2500:2500", "5000:5000", "7500:7500", "10000:10000",
        "--K-values", "5", "10", "20", "40", "80",
        "--K-sweep-n", "5000", "--K-sweep-p", "5000", "--K-fixed", "20",
        "--rho", "0.9", "--block-size", "100",
        "--repeats", str(args.trials), "--n-threads", "10",
        "--dummy-law", "sphere-ag", "--solver", "omp",
        "--out", str(GENERATED_RESULTS / f"{args.tag}_shared_gram"),
        "--plot", str(GENERATED_FIGURES / "shared_gram"),
    ]
    if args.recompute:
        command.append("--recompute")
    _run("paper_shared_gram_scaling.py", command, env)


def large(args: argparse.Namespace, env: dict[str, str]) -> None:
    command = [
        "--n", "1000", "--p", "100000", "--p1", "10",
        "--repeats", str(args.trials), "--K", "20", "--L", "3",
        "--structure", "factor", "--factor-groups", "10",
        "--factor-global-share", "0.25", "--rho", "0.3",
        "--snrs", "1", "2", "4",
        "--methods", "Sphere AG", "Spherical T-Rex",
        "--solvers", "omp", "lars",
        "--n-threads", "10", "--cache-mb", "2048",
        "--out", str(GENERATED_RESULTS / f"{args.tag}_large_scale"),
    ]
    if args.recompute:
        command.append("--recompute")
    _run("paper_large_scale_synthetic.py", command, env)


def reference(env: dict[str, str]) -> None:
    _run(
        "combine_selector_benchmarks.py",
        [
            "--replot-summary", str(ROOT / "reference_results" / "final_selector_comparison.summary.csv"),
            "--plot", str(GENERATED_FIGURES / "reference_selector_comparison.pdf"),
            "--comparison-plot", str(GENERATED_FIGURES / "reference_sphere_uniform.pdf"),
        ],
        env,
    )


def smoke(args: argparse.Namespace, env: dict[str, str]) -> None:
    prefix = GENERATED_RESULTS / f"{args.tag}_smoke_benchmark"
    command = [
        "--n", "60", "--p", "40", "--p1", "4",
        "--alpha", "0.2", "--K", "2", "--L", "1",
        "--solver", "omp", "--sweeps", "rho", "snr",
        "--structures", "factor", "--rhos", "0.1", "--snrs", "1",
        "--rho-snr", "1", "--snr-rho", "0.1",
        "--factor-groups", "4", "--methods",
        "Sphere AG", "Spherical T-Rex", "T-Rex+DA", "Model-X", "Model-X+",
        "--n-mc", "1", "--n-jobs", "1", "--seed", "72000",
        "--out", str(prefix),
        "--plot-prefix", str(GENERATED_FIGURES / f"{args.tag}_smoke_benchmark"),
        "--recompute",
    ]
    _run("paper_benchmark.py", command, env)
    _validate_raw(prefix, expected_rows=10)
    timing_prefix = GENERATED_RESULTS / f"{args.tag}_smoke_timing"
    _run(
        "paper_method_timing.py",
        [
            "--n", "60", "--p", "40", "--p1", "4", "--K", "2", "--L", "1",
            "--structure", "factor", "--factor-groups", "4", "--rho", "0.1",
            "--methods", "Sphere AG", "Spherical T-Rex", "T-Rex+DA", "Model-X pair",
            "--solvers", "omp", "lars", "--repeats", "1", "--n-threads", "1",
            "--out", str(timing_prefix),
            "--plot", str(GENERATED_FIGURES / f"{args.tag}_smoke_timing.pdf"),
            "--recompute",
        ],
        env,
    )
    _validate_raw(timing_prefix, expected_rows=7)
    _run(
        "paper_shared_gram_scaling.py",
        [
            "--sweeps", "joint", "K", "--joint-pairs", "40:40",
            "--K-values", "2", "--K-sweep-n", "40", "--K-sweep-p", "40",
            "--K-fixed", "2", "--p1", "3", "--L", "1", "--T", "1",
            "--repeats", "1", "--n-threads", "1", "--cache-mb", "16",
            "--out", str(GENERATED_RESULTS / f"{args.tag}_smoke_cache"),
            "--plot", str(GENERATED_FIGURES / f"{args.tag}_smoke_cache"),
            "--recompute",
        ],
        env,
    )
    _run(
        "paper_large_scale_synthetic.py",
        [
            "--n", "40", "--p", "60", "--p1", "3", "--repeats", "1",
            "--K", "2", "--L", "1", "--structure", "factor",
            "--factor-groups", "3", "--rho", "0.2", "--snrs", "1",
            "--methods", "Sphere AG", "Spherical T-Rex", "--solvers", "omp", "lars",
            "--n-threads", "1", "--cache-mb", "16",
            "--out", str(GENERATED_RESULTS / f"{args.tag}_smoke_large"),
            "--recompute",
        ],
        env,
    )
    reference(env)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=("smoke", "reference", "figures", "timing", "scaling", "large", "all"),
        default=["smoke"],
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--tag", default="paper")
    parser.add_argument("--recompute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.trials < 1:
        raise ValueError("trials must be positive")
    GENERATED_RESULTS.mkdir(parents=True, exist_ok=True)
    GENERATED_FIGURES.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[variable] = "1"
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(EXPERIMENTS), str(ROOT), env.get("PYTHONPATH", "")]
    )

    stages = args.stage
    if "all" in stages:
        stages = ["figures", "timing", "scaling", "large"]
    actions = {
        "smoke": lambda: smoke(args, env),
        "reference": lambda: reference(env),
        "figures": lambda: figures(args, env),
        "timing": lambda: timing(args, env),
        "scaling": lambda: scaling(args, env),
        "large": lambda: large(args, env),
    }
    for stage in stages:
        actions[stage]()


if __name__ == "__main__":
    main()
