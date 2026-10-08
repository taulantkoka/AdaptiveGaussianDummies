"""Utilities shared by the synthetic direct-versus-cached Gram benchmark."""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vd_selectors as vd


METHOD_STYLES = {
    "direct": dict(color="#6b7280", marker="o", ls="--"),
    "shared": dict(color="#0072B2", marker="s", ls="-"),
}


def _parse_pair(value):
    try:
        n_text, p_text = value.split(":", 1)
        n_value, p_value = int(n_text), int(p_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "joint pairs must have the form n:p") from exc
    if n_value < 2 or p_value < 2:
        raise argparse.ArgumentTypeError("n and p must both be at least two")
    return n_value, p_value


def _machine_description():
    description = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        output = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], check=True,
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return description
    for line in output.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key in {
                "Model Name", "Chip", "Memory", "Total Number of Cores"}:
            description[key.lower().replace(" ", "_")] = value.strip()
    return description


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_se(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(values.size))
        if values.size > 1 else 0.0)
    return mean, standard_error


def _fdr_tpp(selected, support):
    selected = set(map(int, np.asarray(selected, dtype=int).ravel()))
    support = set(map(int, np.asarray(support, dtype=int).ravel()))
    true_positives = len(selected & support)
    false_positives = len(selected - support)
    return (
        float(false_positives / max(len(selected), 1)),
        float(true_positives / max(len(support), 1)),
    )


def _run_trex(X, y, args, K: int, shared: bool, seed: int, mmap_fd: int = -1):
    options = vd.TRexOptions()
    options.solver = {
        "omp": vd.SolverType.OMP,
        "lars": vd.SolverType.LARS,
    }[args.solver]
    options.calib = vd.CalibMode.FixedTL
    options.K = int(K)
    options.L_factor = int(args.L)
    options.T_stop = int(args.T)
    options.tFDR = float(args.alpha)
    options.seed = int(seed)
    options.n_threads = int(args.n_threads)
    options.max_vd_proj = int(args.max_vd_proj)
    options.trim_q = 1.0
    options.verbose = False
    options.use_shared_gram_cache = bool(shared)
    options.gram_cache_mb = float(args.cache_mb)
    options.mmap_fd = int(mmap_fd)
    options.mmap_block_cols = int(args.mmap_block_cols if mmap_fd >= 0 else 0)

    if args.dummy_law == "sphere-ag":
        options.dummy_law = vd.VDDummyLaw.SphericalAdaptiveGaussian
    elif args.dummy_law == "sphere-tail-ag":
        options.dummy_law = vd.VDDummyLaw.SphericalAdaptiveGaussian
        options.ag_tail_m = max(
            1, min(X.shape[1],
                   int(round(args.tail_percentage * X.shape[1] / 100))))
    elif args.dummy_law == "ag":
        options.dummy_law = vd.VDDummyLaw.AdaptiveGaussian
    elif args.dummy_law == "spherical":
        options.dummy_law = vd.VDDummyLaw.Spherical
    else:
        raise ValueError(args.dummy_law)

    started = time.perf_counter()
    result = vd.TRexSelector(options).run(X, y)
    return time.perf_counter() - started, result


def _summarize(rows):
    summary = []
    keys = sorted(
        {(row["sweep"], float(row["x"])) for row in rows},
        key=lambda item: (item[0], item[1]))
    for sweep, x_value in keys:
        point_rows = [
            row for row in rows
            if row["sweep"] == sweep and float(row["x"]) == x_value]
        direct = [row for row in point_rows if row["engine"] == "direct"]
        shared = [row for row in point_rows if row["engine"] == "shared"]
        direct_times = {
            int(row["repeat"]): float(row["runtime_s"]) for row in direct}
        shared_times = {
            int(row["repeat"]): float(row["runtime_s"]) for row in shared}
        speedups = [
            direct_times[index] / shared_times[index]
            for index in sorted(direct_times)]
        speedup_mean, speedup_se = _mean_se(speedups)
        for engine, subset in (("direct", direct), ("shared", shared)):
            runtime_mean, runtime_se = _mean_se(
                [float(row["runtime_s"]) for row in subset])
            summary.append({
                "sweep": sweep,
                "x": x_value,
                "n": int(float(subset[0]["n"])),
                "p": int(float(subset[0]["p"])),
                "K": int(float(subset[0]["K"])),
                "engine": engine,
                "runtime_mean_s": runtime_mean,
                "runtime_se_s": runtime_se,
                "speedup_mean": speedup_mean,
                "speedup_se": speedup_se,
                "all_selected_equal": all(
                    str(row["selected_equal"]).lower() == "true"
                    for row in subset),
                "max_phi_error": max(
                    float(row["max_phi_error"]) for row in subset),
                "cache_hits_mean": float(np.mean(
                    [float(row["cache_hits"]) for row in subset])),
                "cache_misses_mean": float(np.mean(
                    [float(row["cache_misses"]) for row in subset])),
            })
    return summary


def _plot(summary, args):
    sweeps = [
        sweep for sweep in ("joint", "n", "p", "K")
        if sweep in args.sweeps]
    figure, axes = plt.subplots(
        2, len(sweeps), figsize=(4.0 * len(sweeps), 5.6), squeeze=False,
        constrained_layout=True)
    labels = {
        "joint": "joint problem size $(n,p)$",
        "n": "sample size $n$",
        "p": "predictor dimension $p$",
        "K": "random experiments $K$",
    }
    for column, sweep in enumerate(sweeps):
        subset = [row for row in summary if row["sweep"] == sweep]
        for engine in ("direct", "shared"):
            data = sorted(
                (row for row in subset if row["engine"] == engine),
                key=lambda row: float(row["x"]))
            x_values = np.array([float(row["x"]) for row in data])
            means = np.array([float(row["runtime_mean_s"]) for row in data])
            errors = np.array([float(row["runtime_se_s"]) for row in data])
            axes[0, column].errorbar(
                x_values, means, yerr=errors, capsize=2,
                label=engine.capitalize(), **METHOD_STYLES[engine])
        direct = sorted(
            (row for row in subset if row["engine"] == "direct"),
            key=lambda row: float(row["x"]))
        x_values = np.array([float(row["x"]) for row in direct])
        means = np.array([float(row["speedup_mean"]) for row in direct])
        errors = np.array([float(row["speedup_se"]) for row in direct])
        axes[1, column].errorbar(
            x_values, means, yerr=errors, capsize=2, color="#009E73",
            marker="o", lw=1.7)
        axes[1, column].axhline(1.0, color="0.45", lw=0.9, ls="--")
        if sweep == "joint":
            pairs = [
                (int(float(row["n"])), int(float(row["p"])))
                for row in direct]
            for axis in axes[:, column]:
                axis.set_xticks(
                    x_values,
                    [f"({n_value},{p_value})" for n_value, p_value in pairs],
                    rotation=20)
        axes[0, column].set_yscale("log")
        axes[0, column].set_title(labels[sweep])
        axes[1, column].set_xlabel(labels[sweep])
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("runtime (s, log scale)")
    axes[1, 0].set_ylabel("paired speedup")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        f"Synthetic block-Toeplitz, {args.dummy_law}, "
        f"{args.solver.upper()}, $T={args.T}$", fontsize=11)
    output = Path(args.plot)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output.with_suffix('.pdf')}")
