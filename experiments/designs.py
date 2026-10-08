"""Synthetic covariance structures used in the ICASSP experiments."""

from __future__ import annotations

import numpy as np


def generate_design(n, p, structure, rho, block_size, rng):
    """Sample Toeplitz designs without forming a dense covariance matrix."""
    rho = float(rho)
    innovation_scale = np.sqrt(max(1.0 - rho * rho, 0.0))
    X = np.empty((n, p), dtype=float)
    if structure == "toeplitz":
        X[:, 0] = rng.standard_normal(n)
        for j in range(1, p):
            X[:, j] = (
                rho * X[:, j - 1]
                + innovation_scale * rng.standard_normal(n))
    elif structure == "block_toeplitz":
        for start in range(0, p, int(block_size)):
            end = min(start + int(block_size), p)
            X[:, start] = rng.standard_normal(n)
            for j in range(start + 1, end):
                X[:, j] = (
                    rho * X[:, j - 1]
                    + innovation_scale * rng.standard_normal(n))
    else:
        raise ValueError(f"unknown structure {structure!r}")
    return X


def _ma_weights(order, ma_type, alpha, custom):
    order = int(order)
    if ma_type == "custom":
        if custom is None:
            raise ValueError("--ma-type custom requires --ma-thetas")
        weights = np.asarray([float(value) for value in custom.split(",")])
        if weights.size != order + 1:
            raise ValueError("--ma-thetas must contain ma_order+1 values")
        return weights
    if ma_type == "equal":
        return np.ones(order + 1)
    if ma_type == "hanning":
        weights = 0.5 * (
            1.0 - np.cos(2.0 * np.pi * np.arange(order + 1) / (order + 1)))
        weights[0] = max(weights[0], 1e-6)
        return weights
    return float(alpha) ** np.arange(order + 1)


def _sample_ma(n, p, weights, rng):
    weights = np.asarray(weights, dtype=float)
    order = weights.size - 1
    innovations = rng.standard_normal((n, p + order))
    X = np.zeros((n, p), dtype=float)
    for lag, weight in enumerate(weights):
        X += weight * innovations[:, order - lag:order - lag + p]
    return X / max(float(np.linalg.norm(weights)), 1e-12)


def generate_structured_design(n, p, structure, rho, args, rng):
    """Dispatch to the structured Gaussian designs used by the paper."""
    if structure in {"toeplitz", "block_toeplitz"}:
        return generate_design(n, p, structure, rho, args.block_size, rng)

    rho = float(rho)
    if not 0.0 <= rho < 1.0:
        raise ValueError("structured dependence values must lie in [0, 1)")

    if structure == "banded":
        base = _sample_ma(n, p, np.ones(args.bandwidth + 1), rng)
        noise = rng.standard_normal((n, p))
        return np.sqrt(rho) * base + np.sqrt(1.0 - rho) * noise

    if structure == "ma":
        weights = _ma_weights(
            args.ma_order, args.ma_type, rho, args.ma_thetas)
        base = _sample_ma(n, p, weights, rng)
        if args.ma_type in {"equal", "hanning", "custom"}:
            noise = rng.standard_normal((n, p))
            base = np.sqrt(rho) * base + np.sqrt(1.0 - rho) * noise
        return base

    if structure == "factor":
        n_groups = max(1, int(args.factor_groups))
        global_share = float(args.factor_global_share)
        if not 0.0 <= global_share <= 1.0:
            raise ValueError("factor-global-share must lie in [0, 1]")
        group_id = np.minimum(
            np.arange(p, dtype=int) * n_groups // max(p, 1), n_groups - 1)
        X = np.sqrt(max(1.0 - rho, 0.0)) * rng.standard_normal((n, p))
        if global_share > 0.0:
            X += np.sqrt(rho * global_share) * rng.standard_normal(n)[:, None]
        group_share = rho * (1.0 - global_share)
        if group_share > 0.0:
            factors = rng.standard_normal((n, n_groups))
            X += np.sqrt(group_share) * factors[:, group_id]
        return X

    if structure == "block":
        low = np.clip(args.block_rho_low_ratio * rho, 0.0, rho)
        loading_law = getattr(args, "block_loading_law", "uniform")
        if loading_law == "student_t":
            degrees = float(getattr(args, "block_loading_df", 3.0))
            if degrees <= 2.0:
                raise ValueError("block-loading-df must exceed two")
            raw = rng.standard_t(degrees, size=p)
            standardized = raw / np.sqrt(degrees / (degrees - 2.0))
            strength = standardized**2 / (1.0 + standardized**2)
            loading_sq = low + (rho - low) * strength
            loading = np.sqrt(loading_sq)
            if args.block_random_signs:
                loading *= np.sign(raw)
        elif loading_law == "uniform":
            loading_sq = rng.uniform(low, rho, size=p)
            loading = np.sqrt(loading_sq)
            if args.block_random_signs:
                loading *= rng.choice([-1.0, 1.0], size=p)
        else:
            raise ValueError(
                "block-loading-law must be 'uniform' or 'student_t'")
        X = (rng.standard_normal((n, p))
             * np.sqrt(np.maximum(1.0 - loading_sq, 0.0))[None, :])
        for start in range(0, p, int(args.block_size)):
            end = min(start + int(args.block_size), p)
            X[:, start:end] += (
                rng.standard_normal(n)[:, None]
                * loading[None, start:end])
        return X

    if structure == "hierarchical":
        rho_sub = rho
        rho_within = args.hier_within_ratio * rho
        rho_between = args.hier_between_ratio * rho
        if not 0.0 <= rho_between <= rho_within <= rho_sub < 1.0:
            raise ValueError(
                "hierarchical ratios must give 0 <= between <= within <= rho")
        n_super = max(1, int(args.n_superblocks))
        super_size = int(np.ceil(p / n_super))
        sub_size = (int(args.subblock_size)
                    if args.subblock_size is not None
                    else max(5, int(args.block_size) // 5))
        X = np.sqrt(1.0 - rho_sub) * rng.standard_normal((n, p))
        X += np.sqrt(rho_between) * rng.standard_normal(n)[:, None]
        for super_start in range(0, p, super_size):
            super_end = min(super_start + super_size, p)
            X[:, super_start:super_end] += (
                np.sqrt(rho_within - rho_between)
                * rng.standard_normal(n)[:, None])
            for sub_start in range(super_start, super_end, sub_size):
                sub_end = min(sub_start + sub_size, super_end)
                X[:, sub_start:sub_end] += (
                    np.sqrt(rho_sub - rho_within)
                    * rng.standard_normal(n)[:, None])
        return X

    raise ValueError(f"unknown structure {structure!r}")
