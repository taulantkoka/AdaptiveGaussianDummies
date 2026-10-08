"""Large-p synthetic feasibility benchmark for virtual-dummy T-Rex.

The benchmark crosses adaptive and spherical dummy laws with OMP and LARS. It
intentionally excludes Gaussian Model-X at dimensions where the dense
covariance matrix used by the manuscript comparator exceeds physical memory.
It records this resource calculation instead of treating an aborted or
out-of-memory run as a timing observation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

import vd_selectors as vd  # noqa: E402
METHODS = {
    "Sphere AG": vd.VDDummyLaw.SphericalAdaptiveGaussian,
    "Spherical T-Rex": vd.VDDummyLaw.Spherical,
}

SOLVERS = {
    "omp": vd.SolverType.OMP,
    "lars": vd.SolverType.LARS,
}


def _mean_se(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = (
        float(values.std(ddof=1) / np.sqrt(values.size))
        if values.size > 1 else 0.0
    )
    return mean, se


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _prepare_design(args, seed):
    """Generate one large design shared across the requested SNR values."""
    rng = np.random.default_rng(seed)
    X = np.empty((args.n, args.p), dtype=np.float64, order="F")

    if args.structure == "block_toeplitz":
        innovation_scale = np.sqrt(1.0 - args.rho * args.rho)
        for start in range(0, args.p, args.block_size):
            stop = min(start + args.block_size, args.p)
            innovations = rng.standard_normal((args.n, stop - start))
            block = X[:, start:stop]
            block[:, 0] = innovations[:, 0]
            for offset in range(1, stop - start):
                block[:, offset] = (
                    args.rho * block[:, offset - 1]
                    + innovation_scale * innovations[:, offset]
                )
    else:
        # Match the grouped-factor model in the main benchmark while filling
        # the large Fortran-order matrix in bounded temporary blocks.
        chunk_size = 5000
        noise_scale = np.sqrt(1.0 - args.rho)
        for start in range(0, args.p, chunk_size):
            stop = min(start + chunk_size, args.p)
            X[:, start:stop] = (
                noise_scale
                * rng.standard_normal((args.n, stop - start))
            )

        global_scale = np.sqrt(args.rho * args.factor_global_share)
        if global_scale > 0.0:
            X += global_scale * rng.standard_normal(args.n)[:, None]

        group_scale = np.sqrt(args.rho * (1.0 - args.factor_global_share))
        if group_scale > 0.0:
            factors = rng.standard_normal((args.n, args.factor_groups))
            for group in range(args.factor_groups):
                start = group * args.p // args.factor_groups
                stop = (group + 1) * args.p // args.factor_groups
                X[:, start:stop] += group_scale * factors[:, group, None]

    X -= X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0)
    if np.any(norms <= 1e-12):
        raise RuntimeError("synthetic design produced a constant column")
    X /= norms

    support = np.sort(
        rng.choice(args.p, size=args.p1, replace=False)
    ).astype(int)
    signal = np.sum(X[:, support], axis=1) / np.sqrt(args.p1)
    noise = rng.standard_normal(args.n)
    return X, signal, noise, support


def _response(signal, noise, snr):
    noise_sd = np.sqrt(np.var(signal) / max(float(snr), 1e-12))
    y = signal + noise_sd * noise
    y = np.ascontiguousarray(y - y.mean())
    return y


def _select(X, y, support, args, method, solver, seed):
    options = vd.TRexOptions()
    options.solver = SOLVERS[solver]
    options.calib = vd.CalibMode.CalibrateT
    options.K = args.K
    options.L_factor = args.L
    options.T_stop = args.T_max
    options.tFDR = args.alpha
    options.seed = seed
    options.n_threads = args.n_threads
    options.max_vd_proj = args.max_vd_proj
    options.trim_q = 1.0
    options.verbose = False
    options.use_shared_gram_cache = True
    options.gram_cache_mb = args.cache_mb
    options.dummy_law = METHODS[method]

    started = time.perf_counter()
    result = vd.TRexSelector(options).run(X, y)
    elapsed = time.perf_counter() - started
    selected = set(map(int, np.asarray(result.selected_var).ravel()))
    truth = set(map(int, support))
    true_positives = len(selected & truth)
    false_positives = len(selected - truth)
    return {
        "runtime_s": elapsed,
        "fdp": false_positives / max(len(selected), 1),
        "tpp": true_positives / len(truth),
        "selected_size": len(selected),
        "cache_hits": int(result.gram_cache_hits),
        "cache_misses": int(result.gram_cache_misses),
    }


def _summarize(rows, snrs, methods, solvers):
    output = []
    for snr in snrs:
        for method in methods:
            for solver in solvers:
                cells = [
                    row for row in rows
                    if float(row["snr"]) == float(snr)
                    and row["method"] == method
                    and row.get("solver", "omp") == solver
                ]
                row = {
                    "snr": snr,
                    "method": method,
                    "solver": solver,
                    "n_valid": len(cells),
                }
                for metric in (
                    "runtime_s", "fdp", "tpp", "selected_size"
                ):
                    mean, se = _mean_se(
                        [float(cell[metric]) for cell in cells]
                    )
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_se"] = se
                output.append(row)
    return output


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=100_000)
    parser.add_argument("--p1", type=int, default=10)
    parser.add_argument(
        "--snrs", "--snr", type=float, nargs="+", default=[1.0, 2.0, 4.0],
        help="SNR values; --snr is retained as a backward-compatible alias")
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument(
        "--structure", choices=("factor", "block_toeplitz"),
        default="factor")
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--factor-groups", type=int, default=10)
    parser.add_argument("--factor-global-share", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument(
        "--T-max", type=int, default=-1,
        help="Maximum calibrated depth; -1 uses min(L, ceil(n/2))")
    parser.add_argument("--max-vd-proj", type=int, default=200)
    parser.add_argument("--cache-mb", type=float, default=2048.0)
    parser.add_argument("--n-threads", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS),
        default=list(METHODS))
    parser.add_argument(
        "--solvers", nargs="+", choices=list(SOLVERS),
        default=list(SOLVERS),
        help="Forward solvers crossed with every selected dummy law")
    parser.add_argument(
        "--out", default="ag_icassp/results/paper_large_scale_synthetic")
    parser.add_argument("--recompute", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.n < 2 or args.p < 2 or args.p1 < 1 or args.p1 > args.p:
        raise ValueError("require n,p >= 2 and 1 <= p1 <= p")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if any(snr <= 0.0 for snr in args.snrs):
        raise ValueError("all SNR values must be positive")
    if not 0.0 <= args.rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if args.block_size < 1:
        raise ValueError("block-size must be positive")
    if args.factor_groups < 1:
        raise ValueError("factor-groups must be positive")
    if not 0.0 <= args.factor_global_share <= 1.0:
        raise ValueError("factor-global-share must lie in [0, 1]")

    config = {
        key: value for key, value in vars(args).items()
        if key not in {"out", "recompute"}
    }
    config.update({
        "calibration": "CalibrateT",
        "design": (
            "centered unit-normalized grouped-factor Gaussian"
            if args.structure == "factor"
            else "centered unit-normalized block-Toeplitz Gaussian"
        ),
        "dense_covariance_bytes": 8 * args.p * args.p,
        "dense_covariance_gib": 8 * args.p * args.p / 2**30,
        "machine": platform.platform(),
    })
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    stem = Path(f"{args.out}.{digest}")
    raw_path = Path(f"{stem}.raw.csv")
    summary_path = Path(f"{stem}.summary.csv")
    config_path = Path(f"{stem}.json")

    rows = [] if args.recompute or not raw_path.exists() else _read_csv(raw_path)
    completed = {
        (
            int(float(row["repeat"])),
            float(row["snr"]),
            row["method"],
            row.get("solver", "omp"),
        )
        for row in rows
    }
    for repeat in range(args.repeats):
        missing_by_snr = {
            float(snr): [
                (method, solver)
                for method in args.methods
                for solver in args.solvers
                if (repeat, float(snr), method, solver) not in completed
            ]
            for snr in args.snrs
        }
        if not any(missing_by_snr.values()):
            print(f"[{repeat + 1}/{args.repeats}] checkpointed", flush=True)
            continue
        problem_seed = args.seed + repeat * 100_003
        X, signal, noise, support = _prepare_design(args, problem_seed)
        snr_order = (
            list(args.snrs)
            if repeat % 2 == 0 else list(reversed(args.snrs))
        )
        for snr_index, snr in enumerate(snr_order):
            missing = missing_by_snr[float(snr)]
            if not missing:
                continue
            y = _response(signal, noise, snr)
            order = (
                missing
                if (repeat + snr_index) % 2 == 0
                else list(reversed(missing))
            )
            for method, solver in order:
                metrics = _select(
                    X, y, support, args, method, solver, problem_seed + 1)
                rows.append({
                    "repeat": repeat,
                    "problem_seed": problem_seed,
                    "snr": snr,
                    "method": method,
                    "solver": solver,
                    **metrics,
                })
                _write_csv(raw_path, rows)
                print(
                    f"[SNR {snr:g} | {repeat + 1}/{args.repeats}] "
                    f"{method} ({solver.upper()}): "
                    f"{metrics['runtime_s']:.3f}s, "
                    f"FDP={metrics['fdp']:.3f}, "
                    f"TPP={metrics['tpp']:.3f}", flush=True)

    summary = _summarize(rows, args.snrs, args.methods, args.solvers)
    _write_csv(summary_path, summary)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")
    print(f"dense covariance: {config['dense_covariance_gib']:.1f} GiB")


if __name__ == "__main__":
    main()
