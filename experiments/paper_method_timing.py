"""Paired wall-clock comparison of the paper's statistical methods.

The separate scaling driver isolates the direct versus shared real-product
engines at large scale. This driver times complete end-to-end method calls on
one representative problem, including calibrated-T selection for all T-Rex methods. Model-X and
Model-X+ share their knockoff draw and feature statistics, so they are reported
as one timing entry; changing the final threshold has negligible cost.

Methods are executed sequentially in a shuffled order within every paired
repeat.  BLAS and selector thread counts default to one so the comparison does
not hide method cost behind different internal parallelism.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from trex_da_adapter import _run_da_trex  # noqa: E402
from paper_benchmark import (  # noqa: E402
    _implementation_fingerprint,
    _metrics,
    _modelx_pair,
    _standardized_data,
    _trex_select,
)


METHODS = (
    "Sphere-tail AG",
    "Sphere AG",
    "Adaptive Gaussian",
    "Spherical T-Rex",
    "T-Rex+DA",
    "Model-X pair",
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    rows = []
    text = {"method", "base_method", "solver", "error"}
    integers = {"repeat", "seed", "selected_size"}
    with path.open(newline="") as handle:
        for source in csv.DictReader(handle):
            row = {}
            for key, value in source.items():
                if key in text:
                    row[key] = value
                elif key in integers:
                    row[key] = int(float(value))
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def _specs(args):
    specs = []
    for method in args.methods:
        if method == "Model-X pair":
            specs.append((method, "none"))
        else:
            specs.extend((method, solver) for solver in args.solvers)
    return specs


def _label(method, solver):
    return method if solver == "none" else f"{method} ({solver.upper()})"


def _run_method(method, solver, X, Xf, y, support, args, seed):
    local = copy.copy(args)
    local.solver = solver
    started = time.perf_counter()
    if method == "Model-X pair":
        pair = _modelx_pair(X, y, local, seed)
        elapsed = time.perf_counter() - started
        # Runtime is shared.  Use Model-X+ for a conservative diagnostic row.
        metrics = _metrics(pair["Model-X+"], support)
    elif method == "T-Rex+DA":
        selected = _run_da_trex(
            Xf, y, local.alpha, local.K, local.L_factor,
            local.da_method, seed, solver=solver,
            parallel_rnd_experiments=local.parallel_da)
        elapsed = time.perf_counter() - started
        metrics = _metrics(selected, support)
    else:
        selected = _trex_select(Xf, y, method, local, seed)
        elapsed = time.perf_counter() - started
        metrics = _metrics(selected, support)
    return elapsed, metrics


def run_all(args, initial_rows=None, checkpoint_path=None):
    rows = list(initial_rows or [])
    specs = _specs(args)
    completed = {
        (int(row["repeat"]), row["method"])
        for row in rows
    }
    for repeat in range(args.repeats):
        X, Xf, y, support, data_seed = _standardized_data(
            args.structure, args.rho, args.snr, repeat, args)
        order_rng = np.random.default_rng(args.seed + 900_001 + repeat)
        order = list(range(len(specs)))
        order_rng.shuffle(order)
        print(f"repeat {repeat + 1}/{args.repeats}", flush=True)
        for order_index in order:
            method, solver = specs[order_index]
            label = _label(method, solver)
            if (repeat, label) in completed:
                print(f"  {label:32s} checkpointed", flush=True)
                continue
            solver_offset = {"none": 70_001, "omp": 101, "lars": 10_101}[
                solver]
            method_seed = int(data_seed * 1009 + 700_001 + solver_offset)
            try:
                runtime, metrics = _run_method(
                    method, solver, X, Xf, y, support, args, method_seed)
                error = ""
            except Exception as exc:
                runtime = np.nan
                metrics = {"fdp": np.nan, "tpp": np.nan, "selected_size": 0}
                error = f"{type(exc).__name__}: {exc}"
            rows.append({
                "repeat": repeat,
                "seed": data_seed,
                "method": label,
                "base_method": method,
                "solver": solver,
                "runtime_s": runtime,
                "fdp": metrics["fdp"],
                "tpp": metrics["tpp"],
                "selected_size": metrics["selected_size"],
                "error": error,
            })
            suffix = f" FAILED: {error}" if error else ""
            print(f"  {label:32s} {runtime:9.3f} s{suffix}", flush=True)
            if checkpoint_path is not None:
                _write_csv(checkpoint_path, rows)
            gc.collect()
        del X, Xf, y, support
        gc.collect()
    return rows


def _mean_se(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan, 0
    error = (values.std(ddof=1) / np.sqrt(values.size)
             if values.size > 1 else 0.0)
    return float(values.mean()), float(error), int(values.size)


def summarize(rows):
    summary = []
    labels = []
    for row in rows:
        if row["method"] not in labels:
            labels.append(row["method"])
    for label in labels:
        subset = [row for row in rows if row["method"] == label]
        runtime, runtime_se, n_valid = _mean_se(
            [row["runtime_s"] for row in subset])
        fdr, fdr_se, _ = _mean_se([row["fdp"] for row in subset])
        tpp, tpp_se, _ = _mean_se([row["tpp"] for row in subset])
        summary.append({
            "method": label,
            "runtime_mean_s": runtime,
            "runtime_se_s": runtime_se,
            "fdr": fdr,
            "fdr_se": fdr_se,
            "tpp": tpp,
            "tpp_se": tpp_se,
            "n_valid": n_valid,
        })
    summary.sort(key=lambda row: (not np.isfinite(row["runtime_mean_s"]),
                                  row["runtime_mean_s"]))
    return summary


def plot_summary(summary, args, path):
    valid = [row for row in summary if np.isfinite(row["runtime_mean_s"])]
    figure, axis = plt.subplots(figsize=(7.1, max(2.2, 0.34 * len(valid) + 0.7)))
    positions = np.arange(len(valid))
    runtime = np.asarray([row["runtime_mean_s"] for row in valid])
    error = np.asarray([row["runtime_se_s"] for row in valid])
    axis.barh(positions, runtime, xerr=error, color="#0072B2", alpha=0.86,
              capsize=2)
    axis.set_yticks(positions, [row["method"] for row in valid], fontsize=7)
    axis.set_xscale("log")
    axis.set_xlabel("end-to-end runtime (s, log scale)", fontsize=8)
    axis.grid(axis="x", color="0.88", lw=0.5)
    axis.invert_yaxis()
    axis.set_title(
        f"{args.structure}, n={args.n}, p={args.p}, K={args.K}, "
        f"{args.repeats} paired repeats",
        fontsize=9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--p", type=int, default=1000)
    parser.add_argument("--p1", type=int, default=10)
    parser.add_argument(
        "--structure", choices=["toeplitz", "block_toeplitz", "factor"],
        default="factor")
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--snr", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--L", type=int, default=3, dest="L_factor")
    parser.add_argument(
        "--n-threads", type=int, default=1,
        help="Threads for T-Rex random experiments; set the BLAS thread "
             "environment consistently for Model-X")
    parser.add_argument(
        "--parallel-da", action="store_true",
        help="Enable the vendored DA-T-Rex parallel-experiment backend")
    parser.add_argument("--max-vd-proj", type=int, default=200)
    parser.add_argument("--tail-percentage", type=float, default=60.0)
    parser.add_argument("--real-product-engine", choices=["shared", "direct"],
                        default="shared")
    parser.add_argument("--gram-cache-mb", type=float, default=512.0)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--factor-groups", type=int, default=10)
    parser.add_argument("--factor-global-share", type=float, default=0.25)
    parser.add_argument(
        "--da-method",
        choices=["trex+DA+BT", "trex+DA+NN", "trex+DA+AR1", "trex+DA+equi"],
        default="trex+DA+BT")
    parser.add_argument("--mx-fstat", default="lasso")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--solvers", nargs="+", choices=["omp", "lars"],
                        default=["omp"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=96_000)
    parser.add_argument("--out", default="ag_icassp/results/paper_method_timing")
    parser.add_argument("--plot", default="ag_icassp/figs/paper_method_timing.pdf")
    parser.add_argument("--recompute", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.n_threads < 1:
        raise ValueError("n-threads must be positive")
    if args.p1 > args.p:
        raise ValueError("p1 cannot exceed p")
    if not 0.0 < args.tail_percentage <= 100.0:
        raise ValueError("tail-percentage must lie in (0, 100]")
    if not 0.0 <= args.factor_global_share <= 1.0:
        raise ValueError("factor-global-share must lie in [0, 1]")

    config = {key: value for key, value in vars(args).items()
              if key not in {"out", "plot", "recompute"}}
    config["cache_schema_version"] = 1
    config["selector_fingerprint"] = _implementation_fingerprint()
    config["timing_driver_hash"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()[:16]
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    stem = Path(f"{args.out}.{digest}")
    raw_path = Path(f"{stem}.raw.csv")
    summary_path = Path(f"{stem}.summary.csv")
    config_path = Path(f"{stem}.json")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    expected_rows = args.repeats * len(_specs(args))
    if raw_path.exists() and not args.recompute:
        rows = _read_csv(raw_path)
        if len(rows) < expected_rows:
            print(f"resuming {raw_path}: {len(rows)}/{expected_rows} rows")
            rows = run_all(
                args, initial_rows=rows, checkpoint_path=raw_path)
        else:
            print(f"loaded {raw_path}")
    else:
        rows = run_all(args, checkpoint_path=raw_path)
    _write_csv(raw_path, rows)
    print(f"wrote {raw_path}")

    summary = summarize(rows)
    _write_csv(summary_path, summary)
    plot_summary(summary, args, Path(args.plot))
    failures = [row for row in rows if row["error"]]
    if failures:
        print(f"warning: {len(failures)} timed calls failed; inspect {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
