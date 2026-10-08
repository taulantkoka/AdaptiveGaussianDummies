"""Paper benchmark across covariance strength and SNR.

The same Monte Carlo dataset is shared by all methods at each design point.
Two sweeps are run for every requested covariance structure:

* dependence strength at fixed SNR;
* SNR at fixed dependence strength.

The default structures are Toeplitz, block-Toeplitz, and a grouped factor
model with dense global dependence.
Model-X and Model-X+ share one estimated-covariance knockoff draw and differ
only in their threshold offsets.  No method receives the population covariance.
All T-Rex variants, including T-Rex+DA, use the solver selected by ``--solver``.

Drive through import so joblib workers import a named module::

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -c "
  import sys; sys.path[:0]=['ag_icassp/sims','ag_icassp'];
  import paper_benchmark as m; m.main()" \
    --n-mc 200 --n-jobs -1 --backend loky --recompute
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
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
from joblib import Parallel, delayed  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

_HERE = Path(__file__).resolve().parent

import vd_selectors as vd  # noqa: E402
from trex_da_adapter import _run_da_trex  # noqa: E402
from designs import generate_structured_design  # noqa: E402


def _implementation_fingerprint():
    """Hash the benchmark driver, design helpers, and loaded selector binary.

    Simulation caches must be invalidated when statistical or solver code
    changes, even if all command-line arguments remain the same.
    """
    digest = hashlib.sha256()
    selector_extension = importlib.import_module("vd_selectors.vd_selectors")
    paths = (
        Path(__file__).resolve(),
        (_HERE / "trex_da_adapter.py").resolve(),
        (_HERE / "designs.py").resolve(),
        Path(selector_extension.__file__).resolve(),
    )
    for path in paths:
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


METHODS = (
    "Sphere AG",
    "Adaptive Gaussian",
    "Spherical T-Rex",
    "T-Rex+DA",
    "Model-X",
    "Model-X+",
)

METHOD_STYLE = {
    "Sphere AG": dict(color="#000000", marker="o", ls="-", lw=1.8),
    "Adaptive Gaussian": dict(color="#e377c2", marker="x", ls="-.", lw=1.2),
    "Spherical T-Rex": dict(color="#1f77b4", marker="^", ls=":", lw=1.35),
    "T-Rex+DA": dict(color="#2ca02c", marker="D", ls="-.", lw=1.25),
    "Model-X": dict(color="#ff7f0e", marker="v", ls=(0, (5, 2)), lw=1.25),
    "Model-X+": dict(color="#9467bd", marker="P", ls=(0, (2, 2)), lw=1.25),
}

STRUCTURE_LABELS = {
    "toeplitz": "Toeplitz",
    "block_toeplitz": "Block-Toeplitz",
    "banded": "Banded",
    "ma": "Moving average",
    "factor": "Grouped factor",
    "hierarchical": "Hierarchical",
    "block": "Block",
}


def _standardized_data(structure, rho, snr, rep, args):
    structure_index = [
        "toeplitz", "block_toeplitz", "banded", "ma", "factor",
        "hierarchical", "block",
    ].index(structure)
    seed = int(args.seed + rep + 10_000_000 * structure_index
               + round(100_000 * rho) + round(10_000 * snr))
    rng = np.random.default_rng(seed)
    X = generate_structured_design(
        args.n, args.p, structure, rho, args, rng)
    X = np.asarray(X, float)
    X -= X.mean(axis=0, keepdims=True)
    scale = X.std(axis=0, ddof=1)
    X /= np.where(scale > 1e-12, scale, 1.0)

    support = np.sort(rng.choice(args.p, args.p1, replace=False))
    coefficients = np.zeros(args.p)
    coefficients[support] = 1.0 / np.sqrt(args.p1)
    signal = X @ coefficients
    noise_sd = np.sqrt(np.var(signal) / max(float(snr), 1e-12))
    y = signal + noise_sd * rng.standard_normal(args.n)
    y = np.ascontiguousarray(y - y.mean())

    Xf = X / np.maximum(np.linalg.norm(X, axis=0), 1e-12)
    return np.ascontiguousarray(X), np.asfortranarray(Xf), y, support, seed


def _trex_select(Xf, y, method, args, seed):
    options = vd.TRexOptions()
    options.solver = (vd.SolverType.LARS if args.solver == "lars"
                      else vd.SolverType.OMP)
    options.calib = vd.CalibMode.CalibrateT
    options.K = int(args.K)
    options.L_factor = int(args.L_factor)
    options.max_vd_proj = int(args.max_vd_proj)
    options.trim_q = 1.0
    options.tFDR = float(args.alpha)
    options.seed = int(seed)
    options.n_threads = int(getattr(args, "n_threads", 1))
    options.verbose = False
    options.use_shared_gram_cache = args.real_product_engine == "shared"
    options.gram_cache_mb = float(args.gram_cache_mb)
    if method == "Sphere AG":
        options.dummy_law = vd.VDDummyLaw.SphericalAdaptiveGaussian
        options.ag_tail_m = 0
    elif method == "Adaptive Gaussian":
        options.dummy_law = vd.VDDummyLaw.AdaptiveGaussian
    elif method == "Spherical T-Rex":
        options.dummy_law = vd.VDDummyLaw.Spherical
    else:
        raise ValueError(method)
    result = vd.TRexSelector(options).run(Xf, y)
    return np.sort(np.asarray(result.selected_var, dtype=int))


def _modelx_pair(X, y, args, seed):
    """One estimated-covariance knockoff draw, thresholded two ways."""
    from knockpy.knockoff_filter import KnockoffFilter
    from knockpy.knockoff_stats import data_dependent_threshhold

    np.random.seed(int(seed) & 0xFFFFFFFF)
    knockoff_filter = KnockoffFilter(ksampler="gaussian", fstat=args.mx_fstat)
    knockoff_filter.forward(
        X=np.ascontiguousarray(X, dtype=float),
        y=np.ascontiguousarray(y, dtype=float),
        fdr=float(args.alpha),
    )
    statistics = np.asarray(knockoff_filter.W, float)
    selected = {}
    for method, offset in (("Model-X", 0), ("Model-X+", 1)):
        threshold = data_dependent_threshhold(
            W=statistics, fdr=float(args.alpha), offset=offset)
        selected[method] = np.flatnonzero(statistics >= threshold)
    return selected


def _metrics(selected, support):
    selected = set(int(value) for value in selected)
    support = set(int(value) for value in support)
    true_positives = len(selected & support)
    return {
        "fdp": (len(selected) - true_positives) / max(len(selected), 1),
        "tpp": true_positives / max(len(support), 1),
        "selected_size": len(selected),
    }


def _row(base, method, selected, support, error=""):
    row = dict(base, method=method, error=error)
    if error:
        row.update(fdp=np.nan, tpp=np.nan, selected_size=0)
    else:
        row.update(_metrics(selected, support))
    return row


def trial(sweep, structure, x_value, rep, args):
    rho = float(x_value if sweep == "rho" else args.snr_rho)
    snr = float(args.rho_snr if sweep == "rho" else x_value)
    X, Xf, y, support, data_seed = _standardized_data(
        structure, rho, snr, rep, args)
    base = {
        "sweep": sweep,
        "structure": structure,
        "x": float(x_value),
        "rho": rho,
        "snr": snr,
        "rep": int(rep),
        "seed": int(data_seed),
    }
    final_seed = int(data_seed * 1009 + 700_001)
    rows = []

    trex_methods = {
        "Sphere AG", "Adaptive Gaussian",
        "Spherical T-Rex",
    }
    for method in args.methods:
        if method not in trex_methods:
            continue
        try:
            selected = _trex_select(
                Xf, y, method, args, final_seed + 101)
            rows.append(_row(base, method, selected, support))
        except Exception as exc:
            rows.append(_row(
                base, method, [], support,
                error=f"{type(exc).__name__}: {exc}"))

    if "T-Rex+DA" in args.methods:
        try:
            selected = _run_da_trex(
                Xf, y, args.alpha, args.K, args.L_factor,
                args.da_method, final_seed + 50_003, solver=args.solver)
            rows.append(_row(base, "T-Rex+DA", selected, support))
        except Exception as exc:
            rows.append(_row(
                base, "T-Rex+DA", [], support,
                error=f"{type(exc).__name__}: {exc}"))

    if {"Model-X", "Model-X+"} & set(args.methods):
        try:
            pair = _modelx_pair(X, y, args, final_seed + 70_001)
            for method in ("Model-X", "Model-X+"):
                if method in args.methods:
                    rows.append(_row(base, method, pair[method], support))
        except Exception as exc:
            for method in ("Model-X", "Model-X+"):
                if method in args.methods:
                    rows.append(_row(
                        base, method, [], support,
                        error=f"{type(exc).__name__}: {exc}"))
    return rows


def _retry_trial(job, methods, args):
    retry_args = copy.copy(args)
    retry_args.methods = list(methods)
    return trial(*job, retry_args)


def _row_key(row):
    return (
        row["sweep"], row["structure"], float(row["x"]),
        int(row["rep"]), row["method"],
    )


def _cache_compatibility_key(config):
    """Return the statistical configuration shared by extendable caches.

    ``n_mc`` is deliberately omitted so a shorter deterministic run can seed a
    longer run.  The implementation fingerprint is also omitted here because
    this compatibility path is opt-in and is intended for orchestration-only
    driver changes.  All statistical arguments, the cache schema, and the
    solver-dispatch version must still agree.
    """
    ignored = {"implementation_fingerprint", "methods_present", "n_mc"}
    return {
        key: value for key, value in config.items()
        if key not in ignored
    }


def _bootstrap_compatible_rows(args, config, current_config_path, current_rows):
    """Merge deterministic rows from compatible shorter Monte Carlo caches."""
    merged = {_row_key(row): row for row in current_rows}
    target_key = _cache_compatibility_key(config)
    out = Path(args.out)
    # Accept both the current shared prefix (``final_omp.<hash>``) and legacy
    # per-structure prefixes (``final_omp_<structure>.<hash>``).
    pattern = f"{out.name}*.json"
    imported = 0
    sources = 0
    for config_path in sorted(out.parent.glob(pattern)):
        if config_path.resolve() == current_config_path.resolve():
            continue
        try:
            source_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _cache_compatibility_key(source_config) != target_key:
            continue
        raw_path = Path(str(config_path)[:-5] + ".raw.csv")
        if not raw_path.exists():
            continue
        source_rows = _read_csv(raw_path)
        accepted = 0
        for row in source_rows:
            if int(row["rep"]) >= int(args.n_mc):
                continue
            key = _row_key(row)
            if key not in merged:
                merged[key] = row
                imported += 1
                accepted += 1
        if accepted:
            sources += 1
            print(f"bootstrapped {accepted} rows from {raw_path}")
    if sources:
        print(f"bootstrapped {imported} rows from {sources} compatible caches")
    return sorted(merged.values(), key=_row_key)


def _retry_failed_rows(rows, args):
    failures = {}
    for row in rows:
        if not row["error"]:
            continue
        job = (
            row["sweep"], row["structure"], float(row["x"]), int(row["rep"]),
        )
        failures.setdefault(job, set()).add(row["method"])
    if not failures:
        print("no failed rows to retry")
        return rows

    jobs = [(job, tuple(sorted(methods)))
            for job, methods in sorted(failures.items())]
    nested = list(tqdm(
        Parallel(
            n_jobs=args.n_jobs, backend=args.backend, batch_size=1,
            return_as="generator")(
                delayed(_retry_trial)(job, methods, args)
                for job, methods in jobs),
        total=len(jobs), desc="retry failed benchmark rows", dynamic_ncols=True,
    ))
    replacements = {
        _row_key(row): row for group in nested for row in group
    }
    repaired = 0
    output = []
    for row in rows:
        replacement = replacements.get(_row_key(row)) if row["error"] else None
        if replacement is not None:
            output.append(replacement)
            repaired += int(not replacement["error"])
        else:
            output.append(row)
    print(f"repaired {repaired} of {sum(bool(row['error']) for row in rows)} "
          "failed rows")
    return output


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    rows = []
    text = {"sweep", "structure", "method", "error"}
    integers = {"rep", "seed", "selected_size"}
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


def _mean_se(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan, 0
    error = (values.std(ddof=1) / np.sqrt(values.size)
             if values.size > 1 else 0.0)
    return float(values.mean()), float(error), int(values.size)


def summarize(rows, args):
    summary = []
    sweep_values = {"rho": args.rhos, "snr": args.snrs}
    for sweep in args.sweeps:
        for structure in args.structures:
            for x_value in sweep_values[sweep]:
                for method in args.methods:
                    subset = [
                        row for row in rows
                        if row["sweep"] == sweep
                        and row["structure"] == structure
                        and np.isclose(row["x"], x_value)
                        and row["method"] == method
                    ]
                    fdr, fdr_se, n_valid = _mean_se(
                        [row["fdp"] for row in subset])
                    tpp, tpp_se, _ = _mean_se(
                        [row["tpp"] for row in subset])
                    size, size_se, _ = _mean_se(
                        [row["selected_size"] for row in subset])
                    summary.append({
                        "sweep": sweep,
                        "structure": structure,
                        "x": float(x_value),
                        "method": method,
                        "fdr": fdr,
                        "fdr_se": fdr_se,
                        "tpp": tpp,
                        "tpp_se": tpp_se,
                        "selected_size": size,
                        "selected_size_se": size_se,
                        "n_valid": n_valid,
                    })
    return summary


def _series(summary, sweep, structure, method, metric):
    cells = [
        row for row in summary
        if row["sweep"] == sweep and row["structure"] == structure
        and row["method"] == method
    ]
    cells.sort(key=lambda row: row["x"])
    return (np.asarray([row["x"] for row in cells], float),
            np.asarray([row[metric] for row in cells], float))


def plot_sweep(summary, sweep, args, path):
    columns = len(args.structures)
    figure, axes = plt.subplots(
        2, columns, figsize=(7.1, 4.05), squeeze=False, sharey="row")
    for column, structure in enumerate(args.structures):
        for row_index, metric in enumerate(("fdr", "tpp")):
            axis = axes[row_index, column]
            for method in args.methods:
                x, y = _series(summary, sweep, structure, method, metric)
                _, error = _series(
                    summary, sweep, structure, method, f"{metric}_se")
                axis.errorbar(
                    x, y, yerr=error, label=method, markersize=3.0,
                    capsize=1.8, elinewidth=0.65, **METHOD_STYLE[method])
                if metric == "fdr" and np.any(y > 0.25):
                    clipped = y > 0.25
                    axis.scatter(
                        x[clipped], np.full(np.count_nonzero(clipped), 0.247),
                        marker="^", s=13, color=METHOD_STYLE[method]["color"],
                        clip_on=False, zorder=4)
            if metric == "fdr":
                axis.axhline(args.alpha, color="0.35", ls="--", lw=0.8)
                axis.set_ylim(0.0, 0.25)
                axis.set_title(STRUCTURE_LABELS[structure], fontsize=8)
            else:
                axis.set_ylim(0.0, 1.02)
                axis.set_xlabel(
                    r"dependence parameter $\rho$" if sweep == "rho"
                    else "SNR", fontsize=7)
            if column == 0:
                axis.set_ylabel("FDR" if metric == "fdr" else "TPP",
                                fontsize=7)
            axis.tick_params(labelsize=6, length=2)
            axis.grid(color="0.9", lw=0.45)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        fontsize=6.0, handlelength=2.2, columnspacing=0.9,
        bbox_to_anchor=(0.5, 1.005))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=0.65, h_pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--p", type=int, default=1000)
    parser.add_argument("--p1", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--L", type=int, default=3, dest="L_factor")
    parser.add_argument("--solver", choices=["omp", "lars"], default="omp")
    parser.add_argument(
        "--real-product-engine", choices=["shared", "direct"], default="shared",
        help="Use the shared lazy Gram cache or the original observation-space "
             "correlation updates for T-Rex variants.")
    parser.add_argument("--gram-cache-mb", type=float, default=512.0)
    parser.add_argument("--max-vd-proj", type=int, default=200)
    parser.add_argument("--sweeps", nargs="+", choices=["rho", "snr"],
                        default=["rho", "snr"])
    parser.add_argument(
        "--structures", nargs="+",
        choices=list(STRUCTURE_LABELS),
        default=["toeplitz", "block_toeplitz", "factor"])
    parser.add_argument("--rhos", type=float, nargs="+",
                        default=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--snrs", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--rho-snr", type=float, default=1.0,
                        help="SNR held fixed during the rho sweep.")
    parser.add_argument("--snr-rho", type=float, default=0.9,
                        help="Dependence held fixed during the SNR sweep.")
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--bandwidth", type=int, default=20)
    parser.add_argument("--ma-order", type=int, default=20)
    parser.add_argument("--ma-type", choices=["exp_decay", "equal", "hanning",
                                               "custom"], default="exp_decay")
    parser.add_argument("--ma-thetas", default=None)
    parser.add_argument("--factor-groups", type=int, default=10)
    parser.add_argument("--factor-global-share", type=float, default=0.25)
    parser.add_argument("--n-superblocks", type=int, default=5)
    parser.add_argument("--subblock-size", type=int, default=None)
    parser.add_argument("--hier-within-ratio", type=float, default=0.5)
    parser.add_argument("--hier-between-ratio", type=float, default=0.1)
    parser.add_argument("--block-rho-low-ratio", type=float, default=0.5)
    parser.add_argument(
        "--block-loading-law", choices=["uniform", "student_t"],
        default="uniform",
        help="Distribution used for heterogeneous block-factor loadings.")
    parser.add_argument(
        "--block-loading-df", type=float, default=3.0,
        help="Degrees of freedom for Student-t block loadings.")
    parser.add_argument("--block-positive-loadings", action="store_false",
                        dest="block_random_signs", default=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument(
        "--da-method",
        choices=["trex+DA+BT", "trex+DA+NN", "trex+DA+AR1", "trex+DA+equi"],
        default="trex+DA+BT")
    parser.add_argument("--mx-fstat", default="lasso")
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--seed", type=int, default=72_000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--backend", choices=["loky", "threading"],
                        default="loky")
    parser.add_argument("--out", default="ag_icassp/results/paper_benchmark")
    parser.add_argument("--plot-prefix", default="ag_icassp/figs/paper_benchmark")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="Reload the matching cache and rerun only failed method rows.")
    parser.add_argument(
        "--bootstrap-compatible", action="store_true",
        help="Opt in to importing deterministic rows from shorter caches with "
             "the same statistical configuration. Implementation fingerprints "
             "may differ, so use this only after verifying that intervening "
             "changes are orchestration-only.")
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Write the raw cache after this many completed Monte Carlo cells.")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.p1 > args.p:
        raise ValueError("p1 cannot exceed p")
    if args.block_loading_law == "student_t" and args.block_loading_df <= 2.0:
        raise ValueError("block-loading-df must exceed two")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    if args.recompute and args.retry_errors:
        raise ValueError("choose either --recompute or --retry-errors")
    excluded = {
        "n_jobs", "backend", "recompute", "retry_errors", "out",
        "plot_prefix", "bootstrap_compatible", "checkpoint_every",
    }
    requested_methods = list(args.methods)
    config = {
        key: value for key, value in vars(args).items()
        if key not in excluded and key != "methods"
    }
    config["cache_schema_version"] = 4
    config["implementation_fingerprint"] = _implementation_fingerprint()
    config["solver_dispatch_version"] = 2
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    stem = Path(f"{args.out}.{digest}")
    raw_path = Path(f"{stem}.raw.csv")
    summary_path = Path(f"{stem}.summary.csv")
    config_path = Path(f"{stem}.json")
    # Persist the configuration before launching workers so an interrupted raw
    # checkpoint remains discoverable even if orchestration code changes before
    # it is resumed.
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n")

    if args.retry_errors:
        if not raw_path.exists():
            raise FileNotFoundError(
                f"no matching raw cache to repair: {raw_path}")
        rows = _retry_failed_rows(_read_csv(raw_path), args)
        _write_csv(raw_path, rows)
        print(f"updated {raw_path}")
    else:
        rows = _read_csv(raw_path) if raw_path.exists() else []
        if raw_path.exists():
            print(f"loaded {raw_path}")
        if args.bootstrap_compatible and not args.recompute:
            rows = _bootstrap_compatible_rows(
                args, config, config_path, rows)
            if rows:
                _write_csv(raw_path, rows)
                print(f"updated {raw_path} with compatible cached rows")
        if args.recompute:
            requested = set(requested_methods)
            rows = [row for row in rows if row["method"] not in requested]

        values = {"rho": args.rhos, "snr": args.snrs}
        existing = {_row_key(row) for row in rows}
        jobs = []
        for sweep in args.sweeps:
            for structure in args.structures:
                for x_value in values[sweep]:
                    for rep in range(args.n_mc):
                        job = (sweep, structure, float(x_value), rep)
                        missing = tuple(
                            method for method in requested_methods
                            if (*job, method) not in existing)
                        if missing:
                            jobs.append((job, missing))
        if jobs:
            started = time.time()
            row_map = {_row_key(row): row for row in rows}
            additions = 0
            completed = 0
            generator = Parallel(
                n_jobs=args.n_jobs, backend=args.backend, batch_size=1,
                return_as="generator")(
                delayed(_retry_trial)(job, methods, args)
                for job, methods in jobs)
            try:
                for group in tqdm(
                        generator, total=len(jobs), desc="paper benchmark",
                        dynamic_ncols=True):
                    for row in group:
                        row_map[_row_key(row)] = row
                        additions += 1
                    completed += 1
                    if completed % args.checkpoint_every == 0:
                        rows = sorted(row_map.values(), key=_row_key)
                        _write_csv(raw_path, rows)
            finally:
                rows = sorted(row_map.values(), key=_row_key)
                if completed:
                    _write_csv(raw_path, rows)
                    print(f"checkpointed {completed}/{len(jobs)} completed "
                          f"Monte Carlo cells to {raw_path}")
            print(f"computed {additions} method rows "
                  f"({(time.time() - started) / 60.0:.1f} min)")
        else:
            print("all requested method rows are already cached")
        _write_csv(raw_path, rows)
        print(f"updated {raw_path}")

    failed = sum(bool(row["error"]) for row in rows
                 if row["method"] in set(requested_methods))
    if failed:
        print(f"warning: requested cache contains {failed} failed rows; rerun "
              "with --retry-errors to repair them")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    stored_config = dict(config)
    stored_config["methods_present"] = sorted(
        {row["method"] for row in rows})
    config_path.write_text(
        json.dumps(stored_config, indent=2, sort_keys=True) + "\n")

    args.methods = requested_methods
    summary = summarize(rows, args)
    _write_csv(summary_path, summary)
    for sweep in args.sweeps:
        path = Path(f"{args.plot_prefix}_{sweep}.pdf")
        plot_sweep(summary, sweep, args, path)
        print(f"wrote {path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
