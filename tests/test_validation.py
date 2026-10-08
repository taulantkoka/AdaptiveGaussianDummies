"""Regression checks for errors that must fail reproducibility validation."""

import csv

import pytest

from paper_shared_gram_scaling import _validate_results
from reproduce import _validate_raw


@pytest.mark.parametrize("error", [0.01, float("nan"), float("inf")])
def test_occurrence_mismatch_fails_even_with_identical_selections(error):
    with pytest.raises(RuntimeError, match="occurrence matrices"):
        _validate_results([{"selected_equal": True, "max_phi_error": error}])


def test_selected_set_mismatch_fails():
    with pytest.raises(RuntimeError, match="different sets"):
        _validate_results([{"selected_equal": False, "max_phi_error": 0.0}])


def test_empty_cache_results_fail():
    with pytest.raises(RuntimeError, match="no direct/shared results"):
        _validate_results([])


def test_matching_results_pass():
    _validate_results([{"selected_equal": "True", "max_phi_error": "0.0"}])


def test_smoke_rejects_recorded_method_failures(tmp_path):
    prefix = tmp_path / "benchmark"
    with (tmp_path / "benchmark.test.raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "error"])
        writer.writeheader()
        writer.writerow({"method": "T-Rex+DA", "error": "dependency failed"})
    with pytest.raises(RuntimeError, match="failed calls"):
        _validate_raw(prefix, expected_rows=1)
    with pytest.raises(RuntimeError, match="expected 2 rows"):
        _validate_raw(prefix, expected_rows=2)
