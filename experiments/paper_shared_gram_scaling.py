"""Controlled synthetic scaling for direct and shared real Gram products.

The benchmark uses centered, unit-normalized block-Toeplitz Gaussian designs.
It compares the direct observation-space score updates with the shared lazy
real-Gram cache under identical data, response, selector seed, and virtual
dummy law.  Joint-size sweeps increase n and p together.  K sweeps reuse one
fixed design, response, support, and base selector seed for every K value.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from shared_gram_helpers import (
    _fdr_tpp,
    _machine_description,
    _parse_pair,
    _plot,
    _read_csv,
    _run_trex,
    _summarize,
    _write_csv,
)


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    extension = importlib.import_module("vd_selectors.vd_selectors")
    for path in (
        Path(__file__).resolve(),
        (Path(__file__).resolve().parent / "shared_gram_helpers.py"),
        Path(extension.__file__).resolve(),
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _prepare_problem(
        n: int, p: int, p1: int, snr: float, rho: float,
        block_size: int, seed: int):
    if p1 > p:
        raise ValueError("p1 cannot exceed p")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if block_size < 1:
        raise ValueError("block-size must be positive")

    rng = np.random.default_rng(seed)
    X = np.empty((n, p), dtype=np.float64, order="F")
    innovation_scale = np.sqrt(1.0 - rho * rho)
    for start in range(0, p, block_size):
        stop = min(start + block_size, p)
        width = stop - start
        innovations = rng.standard_normal((n, width))
        block = X[:, start:stop]
        block[:, 0] = innovations[:, 0]
        for offset in range(1, width):
            block[:, offset] = (
                rho * block[:, offset - 1]
                + innovation_scale * innovations[:, offset])
        block -= block.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(block, axis=0)
        if np.any(norms <= 1e-12):
            raise RuntimeError("synthetic design produced a constant column")
        block /= norms
        del innovations

    support = np.sort(rng.choice(p, size=p1, replace=False)).astype(int)
    signal = np.sum(X[:, support], axis=1) / np.sqrt(p1)
    noise_sd = np.sqrt(np.var(signal) / max(float(snr), 1e-12))
    y = signal + noise_sd * rng.standard_normal(n)
    y = np.ascontiguousarray(y - y.mean())
    return X, y, support


def _points(args):
    points = []
    if "joint" in args.sweeps:
        points.extend(
            ("joint", p, n, p, args.K_fixed)
            for n, p in args.joint_pairs)
    if "K" in args.sweeps:
        points.extend(
            ("K", K, args.K_sweep_n, args.K_sweep_p, K)
            for K in args.K_values)
    return points


def _row_key(row):
    return row["sweep"], float(row["x"]), int(float(row["repeat"]))


def _run_all(args, initial_rows=None, checkpoint_path=None):
    initial_rows = list(initial_rows or [])
    engines_by_key = {}
    for row in initial_rows:
        engines_by_key.setdefault(_row_key(row), set()).add(row["engine"])
    completed = {
        key for key, engines in engines_by_key.items()
        if engines == {"direct", "shared"}
    }
    rows = [row for row in initial_rows if _row_key(row) in completed]
    points = _points(args)

    for point_index, (sweep, x, n, p, K) in enumerate(points):
        problem_seed = int(
            args.seed + 9_000_003 if sweep == "K"
            else args.seed + point_index * 100_003)
        print(
            f"[{point_index + 1}/{len(points)}] {sweep}={x}: "
            f"n={n}, p={p}, K={K}",
            flush=True)
        X, y, support = _prepare_problem(
            n=n, p=p, p1=args.p1, snr=args.snr, rho=args.rho,
            block_size=args.block_size, seed=problem_seed)

        for repeat in range(args.repeats):
            key = (sweep, float(x), repeat)
            if key in completed:
                print(
                    f"  repeat {repeat + 1}/{args.repeats} checkpointed",
                    flush=True)
                continue
            engines = (
                ("direct", "shared") if repeat % 2 == 0
                else ("shared", "direct"))
            paired = {}
            for engine in engines:
                elapsed, result = _run_trex(
                    X, y, args, K=K, shared=(engine == "shared"),
                    seed=problem_seed + 1)
                selected = np.sort(
                    np.asarray(result.selected_var, dtype=int))
                fdp, tpp = _fdr_tpp(selected, support)
                paired[engine] = {
                    "elapsed": elapsed,
                    "selected": selected,
                    "phi": np.asarray(result.Phi_mat),
                    "fdp": fdp,
                    "tpp": tpp,
                    "hits": int(result.gram_cache_hits),
                    "misses": int(result.gram_cache_misses),
                    "waits": int(result.gram_cache_waits),
                    "retained": int(result.gram_cache_columns),
                }
                print(
                    f"  repeat {repeat + 1}/{args.repeats} "
                    f"{engine:6s}: {elapsed:.3f} s",
                    flush=True)

            selected_equal = np.array_equal(
                paired["direct"]["selected"], paired["shared"]["selected"])
            phi_error = float(np.max(np.abs(
                paired["direct"]["phi"] - paired["shared"]["phi"])))
            for engine in ("direct", "shared"):
                result = paired[engine]
                rows.append({
                    "sweep": sweep,
                    "x": x,
                    "n": n,
                    "p": p,
                    "K": K,
                    "rho": args.rho,
                    "block_size": args.block_size,
                    "problem_seed": problem_seed,
                    "selector_seed": problem_seed + 1,
                    "repeat": repeat,
                    "engine": engine,
                    "runtime_s": result["elapsed"],
                    "fdp": result["fdp"],
                    "tpp": result["tpp"],
                    "selected_size": result["selected"].size,
                    "selected_equal": selected_equal,
                    "max_phi_error": phi_error,
                    "cache_hits": result["hits"],
                    "cache_misses": result["misses"],
                    "cache_waits": result["waits"],
                    "cache_columns": result["retained"],
                })
            if checkpoint_path is not None:
                _write_csv(checkpoint_path, rows)

        del X, y, support
        gc.collect()
    return rows


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweeps", nargs="+", choices=("joint", "K"),
        default=("joint", "K"))
    parser.add_argument(
        "--joint-pairs", nargs="+", type=_parse_pair,
        default=((2500, 2500), (5000, 5000),
                 (7500, 7500), (10000, 10000)),
        metavar="N:P")
    parser.add_argument(
        "--K-values", type=int, nargs="+", default=(5, 10, 20, 40, 80))
    parser.add_argument("--K-sweep-n", type=int, default=5000)
    parser.add_argument("--K-sweep-p", type=int, default=5000)
    parser.add_argument("--K-fixed", type=int, default=20)
    parser.add_argument("--p1", type=int, default=10)
    parser.add_argument("--snr", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--L", type=int, default=3,
        help="Number of virtual dummies as a multiple of p")
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--solver", choices=("omp", "lars"), default="omp")
    parser.add_argument(
        "--dummy-law",
        choices=("sphere-ag", "sphere-tail-ag", "ag", "spherical"),
        default="sphere-ag")
    parser.add_argument("--tail-percentage", type=float, default=60.0)
    parser.add_argument("--max-vd-proj", type=int, default=200)
    parser.add_argument("--n-threads", type=int, default=10)
    parser.add_argument("--cache-mb", type=float, default=2048.0)
    parser.add_argument("--mmap-block-cols", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--out", default="ag_icassp/results/shared_gram_synthetic")
    parser.add_argument(
        "--plot", default="ag_icassp/figs/shared_gram_synthetic")
    parser.add_argument("--recompute", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.p1 < 1:
        raise ValueError("p1 must be positive")
    if len(set(args.joint_pairs)) != len(args.joint_pairs):
        raise ValueError("joint pairs must be unique")
    if len(set(args.K_values)) != len(args.K_values):
        raise ValueError("K values must be unique")

    config = {
        key: value for key, value in vars(args).items()
        if key not in {"out", "plot", "recompute"}
    }
    config.update({
        "cache_schema_version": 1,
        "implementation_fingerprint": _implementation_fingerprint(),
        "machine": _machine_description(),
        "design": (
            "centered unit-normalized block-Toeplitz Gaussian columns"),
    })
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    stem = Path(f"{args.out}.{digest}")
    raw_path = Path(f"{stem}.raw.csv")
    summary_path = Path(f"{stem}.summary.csv")
    config_path = Path(f"{stem}.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    expected_rows = len(_points(args)) * args.repeats * 2
    initial_rows = []
    if raw_path.exists() and not args.recompute:
        initial_rows = _read_csv(raw_path)
    if len(initial_rows) >= expected_rows:
        print(f"loading {raw_path}")
        rows = initial_rows
    else:
        if initial_rows:
            print(
                f"resuming {raw_path}: "
                f"{len(initial_rows)}/{expected_rows} rows")
        rows = _run_all(
            args, initial_rows=initial_rows, checkpoint_path=raw_path)
        _write_csv(raw_path, rows)
        print(f"wrote {raw_path}")

    summary = _summarize(rows)
    _write_csv(summary_path, summary)
    _plot(summary, args)
    _validate_results(rows)
    print(f"wrote {summary_path}")


def _validate_results(rows):
    if not rows:
        raise RuntimeError("no direct/shared results to validate")
    if not all(
            str(row["selected_equal"]).lower() == "true"
            for row in rows):
        raise RuntimeError("direct and shared engines selected different sets")
    errors = np.asarray([float(row["max_phi_error"]) for row in rows])
    if not np.all(np.isfinite(errors)) or np.any(errors != 0.0):
        raise RuntimeError("direct and shared engines produced different occurrence matrices")
    max_error = float(np.max(errors))
    print(f"maximum occurrence-matrix difference: {max_error:.3g}")


if __name__ == "__main__":
    main()
