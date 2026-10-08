"""Small adapter for the published dependency-aware T-Rex implementation."""

from __future__ import annotations

import numpy as np
import trex_selector_neo as tsn


_DA_METHOD_MAP = {
    "trex+DA+BT": tsn.DAMethod.BT,
    "trex+DA+NN": tsn.DAMethod.NN,
    "trex+DA+AR1": tsn.DAMethod.AR1,
    "trex+DA+equi": tsn.DAMethod.EQUI,
}


def run_da_trex(
        X, y, tFDR, K, L_factor, method, seed, solver="lars",
        parallel_rnd_experiments=False):
    """Run ordinary or dependency-aware T-Rex through TRexSelectorNeo."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    cpp_seed = int(seed) % np.iinfo(np.int32).max

    control = tsn.TRexControlParameter()
    solver_types = {
        "lars": tsn.SolverTypeForTRex.TLARS,
        "omp": tsn.SolverTypeForTRex.TOMP,
    }
    solver_key = str(solver).lower()
    if solver_key not in solver_types:
        raise ValueError(
            f"Unsupported T-Rex+DA solver {solver!r}; choose 'omp' or 'lars'")
    control.solver_type = solver_types[solver_key]
    control.K = int(K)
    control.max_dummy_multiplier = int(L_factor)
    control.parallel_rnd_experiments = bool(parallel_rnd_experiments)

    if method == "trex":
        selector = tsn.TRexSelector(
            X, y, tFDR=float(tFDR), trex_control=control,
            seed=cpp_seed, verbose=False)
    else:
        da_control = tsn.TRexDAControlParameter()
        da_control.method = _DA_METHOD_MAP[method]
        selector = tsn.TRexDASelector(
            X, y, tFDR=float(tFDR), da_control=da_control,
            trex_control=control, seed=cpp_seed, verbose=False)
    selector.select()
    return sorted(int(index) for index in selector.selected_indices)


# Preserve the private name used by the original experiment drivers.
_run_da_trex = run_da_trex
