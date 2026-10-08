"""Small numerical and native-extension checks, independent of paper timings."""

import numpy as np
import pytest
from scipy.stats import beta

import vd_selectors as vd


def problem(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 40))
    X -= X.mean(axis=0)
    X /= np.linalg.norm(X, axis=0)
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.1, size=60)
    return np.asfortranarray(X), y - y.mean()


@pytest.mark.parametrize("solver", [vd.SolverType.OMP, vd.SolverType.LARS])
@pytest.mark.parametrize("law", [vd.VDDummyLaw.Spherical,
                                vd.VDDummyLaw.SphericalAdaptiveGaussian])
@pytest.mark.parametrize("threads", [1, 2])
def test_shared_cache_matches_direct_and_is_repeatable(solver, law, threads):
    X, y = problem()
    options = vd.TRexOptions()
    options.solver = solver
    options.dummy_law = law
    options.calib = vd.CalibMode.FixedTL
    options.K = 4
    options.L_factor = 1
    options.T_stop = 2
    options.seed = 123
    options.n_threads = threads
    options.verbose = False
    options.use_shared_gram_cache = False
    direct = vd.TRexSelector(options).run(X, y)
    options.use_shared_gram_cache = True
    for _ in range(2):
        cached = vd.TRexSelector(options).run(X, y)
        np.testing.assert_array_equal(direct.selected_var, cached.selected_var)
        np.testing.assert_array_equal(direct.Phi_mat, cached.Phi_mat)
        assert np.all(np.isfinite(cached.Phi_mat))
        assert np.all((cached.Phi_mat >= 0) & (cached.Phi_mat <= 1))


@pytest.mark.parametrize("solver", [vd.VD_OMP, vd.VD_LARS, vd.VD_AFS])
def test_forward_solvers_run(solver):
    X, y = problem()
    options = vd.VDOptions()
    options.seed = 42
    model = solver(X, y, 40, options)
    model.run(2)
    assert model.n_samples() == 60
    assert model.n_features() == 40
    assert np.all(np.isfinite(model.beta_real()))


@pytest.mark.parametrize("n", [3, 10, 100])
def test_spherical_quantile_matches_scipy(n):
    for probability in [0.01, 0.5, 0.99]:
        expected = np.sqrt(beta.ppf(probability, 0.5, 0.5 * (n - 2)))
        assert vd._centered_spherical_abs_quantile(probability, n) == pytest.approx(
            expected, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("writable", [False, True])
def test_mmap_roundtrip_and_permissions(tmp_path, writable):
    X, _ = problem()
    path = str(tmp_path / "matrix.bin")
    vd.MMapMatrix.create_from_array(path, X)
    mapped = vd.MMapMatrix(path, *X.shape, writable=writable)
    array = mapped.as_array()
    np.testing.assert_array_equal(array, X)
    assert array.flags.f_contiguous
    assert array.flags.writeable == writable
    if writable:
        array[0, 0] = 99.0
        assert mapped.as_array()[0, 0] == 99.0
    else:
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 99.0


def test_mmap_rejects_truncated_files(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(b"\0" * 8)
    with pytest.raises(RuntimeError, match="too small"):
        vd.MMapMatrix(str(path), 10, 10)


def test_mmap_rejects_empty_dimensions(tmp_path):
    with pytest.raises(ValueError, match="dimensions"):
        vd.MMapMatrix(str(tmp_path / "empty.bin"), 0, 10, writable=True)


def test_invalid_shapes_raise():
    X, y = problem()
    with pytest.raises(RuntimeError, match="X.rows != y.size"):
        vd.TRexSelector(vd.TRexOptions()).run(X, y[:-1])
