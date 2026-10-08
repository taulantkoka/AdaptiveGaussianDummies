// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vd_lars.hpp"
#include "vd_omp.hpp"
#include "vd_afs.hpp"
#include "trex.hpp"
#include "memory_mapped_eigen_matrix.hpp"

namespace py = pybind11;

// Helper: construct any VD solver from numpy arrays
template <typename Solver>
static std::unique_ptr<Solver> make_solver(
    py::array_t<double, py::array::f_style> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    int num_dummies, const VDOptions& opt)
{
  auto bx = X.request();
  auto by = y.request();
  if (bx.ndim != 2) throw std::runtime_error("X must be 2D.");
  if (by.ndim != 1) throw std::runtime_error("y must be 1D.");
  if (bx.shape[0] != by.shape[0]) throw std::runtime_error("X.rows != y.size");
  return std::unique_ptr<Solver>(new Solver(
      static_cast<double*>(bx.ptr), (int)bx.shape[0], (int)bx.shape[1],
      static_cast<double*>(by.ptr), (int)by.shape[0],
      num_dummies, opt));
}

// Helper: bind common accessors on any VD solver (they all inherit from VD_Base)
template <typename Cls, typename PyClass>
static void bind_common(PyClass& c) {
  c.def("basis_size",           &Cls::basis_size)
   .def("num_dummies", [](const Cls& self) {
       return self.num_dummies();
   })
   .def("num_realized_dummies", [](const Cls& self) {
       return self.num_realized_dummies();
   })
   .def("num_realized_calibration_dummies",
        [](const Cls& self) {
          return self.num_realized_calibration_dummies();
        })
   .def("n_features",           &Cls::n_features)
   .def("n_samples",            &Cls::n_samples)
   .def("sg_beta_path",         &Cls::sg_beta_path_copy)
   .def("dummy_radius_path",     &Cls::dummy_radius_path_copy)
   .def("X_realized_copy", &Cls::X_realized_copy)
   .def("realized_dummy_column_copy", &Cls::realized_dummy_column_copy, py::arg("j"))
   .def("normx",    &Cls::normx_view,   py::return_value_policy::reference_internal)
   .def("vd_proj",  &Cls::vd_proj_view, py::return_value_policy::reference_internal)
   .def("vd_corr",  &Cls::vd_corr_view, py::return_value_policy::reference_internal)
   .def("vd_stick", &Cls::vd_stick_view,py::return_value_policy::reference_internal)

   .def("beta_view_copy",     &Cls::beta_view_copy)
   .def("beta_real",          &Cls::beta_real)
   .def("corr_view_copy",     &Cls::corr_view_copy)
   .def("corr_realized_view", &Cls::corr_realized_view_copy)
   .def("active_indices",     &Cls::active_indices)
   .def("is_dummy_realized",  &Cls::is_dummy_realized_view)

   .def("active_features", [](const Cls& self) {
       py::list out;
       for (const auto& af : self.active_features_copy()) {
         const char* kind = "dummy";
         if (af.kind == ActiveFeature::Kind::Real) kind = "real";
         else if (af.kind == ActiveFeature::Kind::Persistent)
           kind = "persistent";
         else if (af.kind == ActiveFeature::Kind::Calibration)
           kind = "calibration";
         out.append(py::make_tuple(kind, af.index, af.label));
       }
       return out;
   });
}

PYBIND11_MODULE(vd_selectors, m) {

  m.def("_centered_spherical_abs_quantile",
        [](double probability, int n) {
          if (n < 3) throw std::invalid_argument("n must be at least 3");
          const double squared = vd_detail::beta_quantile(
              probability, 0.5, 0.5 * double(n - 2));
          return std::sqrt(std::min(std::max(squared, 0.0), 1.0));
        },
        py::arg("probability"), py::arg("n"));

  m.def("_conditional_spherical_pool_radius",
        [](double threshold, double target_probability,
           const Eigen::VectorXd& means,
           const Eigen::VectorXd& scales,
           int remaining_dimension) {
          if (means.size() != scales.size())
            throw std::invalid_argument(
                "means and scales must have the same length");
          std::vector<double> mean_values(
              means.data(), means.data() + means.size());
          std::vector<double> scale_values(
              scales.data(), scales.data() + scales.size());
          return vd_detail::conditional_spherical_pool_radius(
              threshold, target_probability, mean_values, scale_values,
              remaining_dimension);
        },
        py::arg("threshold"), py::arg("target_probability"),
        py::arg("means"), py::arg("scales"),
        py::arg("remaining_dimension"));

  m.def("_covariance_envelope_theta",
        &vd_detail::covariance_envelope_theta,
        py::arg("model"), py::arg("rho"),
        py::arg("remaining_active_upper_bound"),
        py::arg("selected_real"));

  // ==================================================================
  // Shared types
  // ==================================================================
  py::enum_<VDDummyLaw>(m, "VDDummyLaw", py::arithmetic())
    .value("Spherical", VDDummyLaw::Spherical)
    .value("Gaussian",  VDDummyLaw::Gaussian)
    .value("AdaptiveGaussian", VDDummyLaw::AdaptiveGaussian)
    .value("AdaptiveBeta", VDDummyLaw::AdaptiveBeta)
    .value("SubGaussian", VDDummyLaw::SubGaussian)
    .value("SphericalAdaptiveGaussian",
           VDDummyLaw::SphericalAdaptiveGaussian)
    .export_values();

  py::enum_<DummyRadiusCovarianceModel>(
      m, "DummyRadiusCovarianceModel")
    .value("None_", DummyRadiusCovarianceModel::None)
    .value("CompoundSymmetric",
           DummyRadiusCovarianceModel::CompoundSymmetric)
    .value("AR1", DummyRadiusCovarianceModel::AR1);

  py::class_<VDOptions>(m, "VDOptions")
    .def(py::init<>())
    .def_readwrite("T_stop",           &VDOptions::T_stop)
    .def_readwrite("consecutive_dummy_stop",
                   &VDOptions::consecutive_dummy_stop)
    .def_readwrite("max_vd_proj",     &VDOptions::max_vd_proj)
    .def_readwrite("eps",             &VDOptions::eps)
    .def_readwrite("standardize",     &VDOptions::standardize)
    .def_readwrite("debug",           &VDOptions::debug)
    .def_readwrite("seed",            &VDOptions::seed)
    .def_readwrite("dummy_law",       &VDOptions::dummy_law)
    .def_readwrite("rho",             &VDOptions::rho)
    .def_readwrite("trim_q",          &VDOptions::trim_q)
    .def_readwrite("ag_tail_m",       &VDOptions::ag_tail_m)
    .def_readwrite("sg_beta",         &VDOptions::sg_beta)
    .def_readwrite("sg_adaptive",     &VDOptions::sg_adaptive)
    .def_readwrite("sg_fit_q_lo",     &VDOptions::sg_fit_q_lo)
    .def_readwrite("sg_fit_q_hi",     &VDOptions::sg_fit_q_hi)
    .def_readwrite("sg_beta_min",     &VDOptions::sg_beta_min)
    .def_readwrite("sg_beta_max",     &VDOptions::sg_beta_max)
    .def_readwrite("sg_beta_smooth",  &VDOptions::sg_beta_smooth)
    .def_readwrite("sg_tau",          &VDOptions::sg_tau)
    .def_readwrite("vd_seed_corr",    &VDOptions::vd_seed_corr)
    .def_readwrite("sigma_reg",       &VDOptions::sigma_reg)
    .def_readwrite("groups",          &VDOptions::groups)
    .def_readwrite("mmap_fd",         &VDOptions::mmap_fd)
    .def_readwrite("mmap_block_cols", &VDOptions::mmap_block_cols)
    .def_readwrite("fresh_calibration_dummies",
                   &VDOptions::fresh_calibration_dummies)
    .def_readwrite("dummy_priority_radius",
                   &VDOptions::dummy_priority_radius);

  py::class_<ActiveFeature>(m, "ActiveFeature")
    .def_property_readonly("kind", [](const ActiveFeature& af) {
        return static_cast<int>(af.kind);
    })
    .def_readonly("index", &ActiveFeature::index)
    .def_readonly("label", &ActiveFeature::label)
    .def("__repr__", [](const ActiveFeature& af) {
        const char* kind = "Dummy";
        if (af.kind == ActiveFeature::Kind::Real) kind = "Real";
        else if (af.kind == ActiveFeature::Kind::Persistent)
          kind = "Persistent";
        else if (af.kind == ActiveFeature::Kind::Calibration)
          kind = "Calibration";
        return std::string("<ActiveFeature kind=") +
               kind +
               " index=" + std::to_string(af.index) +
               " label=" + std::to_string(af.label) + ">";
    });

  // ==================================================================
  // VD_LARS
  // ==================================================================
  auto lars = py::class_<VD_LARS>(m, "VD_LARS")
    .def(py::init(&make_solver<VD_LARS>),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())
    .def("run", [](VD_LARS& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1);
  bind_common<VD_LARS>(lars);

  // ==================================================================
  // VD_OMP
  // ==================================================================
  auto omp = py::class_<VD_OMP>(m, "VD_OMP")
    .def(py::init(&make_solver<VD_OMP>),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())
    .def("run", [](VD_OMP& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1);
  bind_common<VD_OMP>(omp);

  // ==================================================================
  // VD_AFS
  // ==================================================================
  auto afs = py::class_<VD_AFS>(m, "VD_AFS")
    .def(py::init(&make_solver<VD_AFS>),
        py::arg("X"), py::arg("y"), py::arg("num_dummies"), py::arg("options"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>())
    .def("run", [](VD_AFS& self, int T) {
          py::gil_scoped_release release;
          return self.run(T);
        }, py::arg("T") = 1)
    .def("rho", &VD_AFS::rho);
  bind_common<VD_AFS>(afs);

  // ==================================================================
  // MMapMatrix
  // ==================================================================
  using MMap = MemoryMappedEigenMatrix<double>;

  py::class_<MMap>(m, "MMapMatrix")
    .def(py::init([](const std::string& filename, int nrows, int ncols, bool writable) {
          auto mode = writable ? MMap::Mode::ReadWrite : MMap::Mode::ReadOnly;
          return std::unique_ptr<MMap>(new MMap(filename, nrows, ncols, mode));
        }),
        py::arg("filename"), py::arg("nrows"), py::arg("ncols"),
        py::arg("writable") = false)

    .def_property_readonly("nrows", &MMap::nrows)
    .def_property_readonly("ncols", &MMap::ncols)
    .def_property_readonly("nbytes", &MMap::size_bytes)
    .def_property_readonly("shape", [](const MMap& self) {
        return py::make_tuple(self.nrows(), self.ncols());
    })

    .def("as_array", [](py::object self_obj) {
      auto& self = self_obj.cast<MMap&>();
      py::ssize_t n = self.nrows(), p = self.ncols();
      py::ssize_t itemsize = sizeof(double);
      auto array = py::array(py::dtype::of<double>(),
          {n, p}, {itemsize, n*itemsize}, self.data(), self_obj);
      if (self.mode() == MMap::Mode::ReadOnly)
        array.attr("setflags")(py::arg("write") = false);
      return array;
    }, "Zero-copy NumPy view onto the mapped matrix.")

    .def("fileno", &MMap::fileno,
        "File descriptor (Unix) or -1 (Windows).")

    .def_static("create_from_array",
        [](const std::string& filename,
           py::array_t<double, py::array::f_style | py::array::forcecast> arr) {
          auto buf = arr.request();
          if (buf.ndim != 2) throw std::runtime_error("Array must be 2D.");
          MMap::create_from_ptr(filename,
              static_cast<const double*>(buf.ptr),
              buf.shape[0], buf.shape[1], true, MMap::Mode::ReadOnly);
        },
        py::arg("filename"), py::arg("array"),
        "Write a col-major numpy array to a binary file.")

    .def("__repr__", [](const MMap& self) {
        return "<MMapMatrix " + std::to_string(self.nrows()) + "x" +
               std::to_string(self.ncols()) + " (" +
               std::to_string(self.size_bytes()/(1024*1024)) + " MB) " +
               (self.mode() == MMap::Mode::ReadOnly ? "ro" : "rw") + ">";
    });

  // ==================================================================
  // TRex
  // ==================================================================
  py::enum_<SolverType>(m, "SolverType")
    .value("LARS", SolverType::LARS)
    .value("OMP",  SolverType::OMP)
    .value("AFS",  SolverType::AFS)
    .value("NNLS", SolverType::NNLS)
    .export_values();

  py::enum_<CalibMode>(m, "CalibMode")
    .value("FixedTL",       CalibMode::FixedTL)
    .value("CalibrateT",    CalibMode::CalibrateT)
    .value("CalibrateL",    CalibMode::CalibrateL)
    .value("CalibrateBoth", CalibMode::CalibrateBoth)
    .export_values();

  py::class_<TRexOptions>(m, "TRexOptions")
    .def(py::init<>())
    .def_readwrite("tFDR",           &TRexOptions::tFDR)
    .def_readwrite("K",              &TRexOptions::K)
    .def_readwrite("L_factor",       &TRexOptions::L_factor)
    .def_readwrite("L",              &TRexOptions::L)
    .def_readwrite("T_stop",          &TRexOptions::T_stop)
    .def_readwrite("consecutive_dummy_stop",
                   &TRexOptions::consecutive_dummy_stop)
    .def_readwrite("max_L_factor",   &TRexOptions::max_L_factor)
    .def_readwrite("stride_width",   &TRexOptions::stride_width)
    .def_readwrite("groups",         &TRexOptions::groups)
    .def_readwrite("posthoc_mode",   &TRexOptions::posthoc_mode)
    .def_readwrite("max_stale_strides", &TRexOptions::max_stale_strides)
    .def_readwrite("max_vd_proj",    &TRexOptions::max_vd_proj)
    .def_readwrite("eps",            &TRexOptions::eps)
    .def_readwrite("verbose",        &TRexOptions::verbose)
    .def_readwrite("seed",           &TRexOptions::seed)
    .def_readwrite("solver",         &TRexOptions::solver)
    .def_readwrite("calib",          &TRexOptions::calib)
    .def_readwrite("dummy_law",      &TRexOptions::dummy_law)
    .def_readwrite("rho",            &TRexOptions::rho)
    .def_readwrite("trim_q",         &TRexOptions::trim_q)
    .def_readwrite("ag_tail_m",      &TRexOptions::ag_tail_m)
    .def_readwrite("deflation_v",    &TRexOptions::deflation_v)
    .def_readwrite("sg_beta",        &TRexOptions::sg_beta)
    .def_readwrite("sg_adaptive",    &TRexOptions::sg_adaptive)
    .def_readwrite("sg_fit_q_lo",    &TRexOptions::sg_fit_q_lo)
    .def_readwrite("sg_fit_q_hi",    &TRexOptions::sg_fit_q_hi)
    .def_readwrite("sg_beta_min",    &TRexOptions::sg_beta_min)
    .def_readwrite("sg_beta_max",    &TRexOptions::sg_beta_max)
    .def_readwrite("sg_beta_smooth", &TRexOptions::sg_beta_smooth)
    .def_readwrite("sg_tau",         &TRexOptions::sg_tau)
    .def_readwrite("vd_seed_corr",   &TRexOptions::vd_seed_corr)
    .def_readwrite("sigma_reg",      &TRexOptions::sigma_reg)
    .def_readwrite("mmap_fd",        &TRexOptions::mmap_fd)
    .def_readwrite("mmap_block_cols",&TRexOptions::mmap_block_cols)
    .def_readwrite("n_threads",      &TRexOptions::n_threads)
    .def_readwrite("use_shared_gram_cache", &TRexOptions::use_shared_gram_cache)
    .def_readwrite("gram_cache_mb",  &TRexOptions::gram_cache_mb)
    .def_readwrite("persistent_spherical_controls",
                   &TRexOptions::persistent_spherical_controls)
    .def_readwrite("fresh_calibration_dummies",
                   &TRexOptions::fresh_calibration_dummies)
    .def_readwrite("fresh_calibration_priority_sigma",
                   &TRexOptions::fresh_calibration_priority_sigma)
    .def_readwrite("dummy_priority_radius",
                   &TRexOptions::dummy_priority_radius)
    .def_readwrite("adaptive_dummy_radius",
                   &TRexOptions::adaptive_dummy_radius)
    .def_readwrite("dummy_radius_tail_m",
                   &TRexOptions::dummy_radius_tail_m)
    .def_readwrite("dummy_radius_tail_fraction",
                   &TRexOptions::dummy_radius_tail_fraction)
    .def_readwrite("dummy_radius_target_win_prob",
                   &TRexOptions::dummy_radius_target_win_prob)
    .def_readwrite("dummy_radius_urn_target",
                   &TRexOptions::dummy_radius_urn_target)
    .def_readwrite("dummy_radius_account_for_stopping",
                   &TRexOptions::dummy_radius_account_for_stopping)
    .def_readwrite("dummy_radius_active_upper_bound",
                   &TRexOptions::dummy_radius_active_upper_bound)
    .def_readwrite("dummy_radius_real_score_quantile",
                   &TRexOptions::dummy_radius_real_score_quantile)
    .def_readwrite("dummy_radius_real_score_rank",
                   &TRexOptions::dummy_radius_real_score_rank)
    .def_readwrite("dummy_radius_envelope_theta",
                   &TRexOptions::dummy_radius_envelope_theta)
    .def_readwrite("dummy_radius_noise_score_bound",
                   &TRexOptions::dummy_radius_noise_score_bound)
    .def_readwrite("dummy_radius_covariance_model",
                   &TRexOptions::dummy_radius_covariance_model)
    .def_readwrite("dummy_radius_covariance_rho",
                   &TRexOptions::dummy_radius_covariance_rho)
    .def_readwrite("persistent_score_ar1",
                   &TRexOptions::persistent_score_ar1)
    .def_readwrite("dummy_radius_scale_stopping",
                   &TRexOptions::dummy_radius_scale_stopping)
    .def_readwrite("dummy_radius_smooth",
                   &TRexOptions::dummy_radius_smooth)
    .def_readwrite("dummy_radius_min",
                   &TRexOptions::dummy_radius_min)
    .def_readwrite("dummy_radius_max",
                   &TRexOptions::dummy_radius_max);

  py::class_<TRexResult>(m, "TRexResult")
    .def(py::init<>())
    .def_readonly("selected_var",  &TRexResult::selected_var)
    .def_readonly("v_thresh",      &TRexResult::v_thresh)
    .def_readonly("T_stop",        &TRexResult::T_stop)
    .def_readonly("num_dummies",   &TRexResult::num_dummies)
    .def_readonly("L_calibrated",  &TRexResult::L_calibrated)
    .def_readonly("V",             &TRexResult::V)
    .def_readonly("FDP_hat_mat",   &TRexResult::FDP_hat_mat)
    .def_readonly("Phi_mat",       &TRexResult::Phi_mat)
    .def_readonly("Phi_prime",     &TRexResult::Phi_prime)
    .def_readonly("coefficient_mean", &TRexResult::coefficient_mean)
    .def_readonly("K",             &TRexResult::K)
    .def_readonly("gram_cache_hits", &TRexResult::gram_cache_hits)
    .def_readonly("gram_cache_misses", &TRexResult::gram_cache_misses)
    .def_readonly("gram_cache_waits", &TRexResult::gram_cache_waits)
    .def_readonly("gram_cache_columns", &TRexResult::gram_cache_columns)
    .def_readonly("persistent_master_rank",
                  &TRexResult::persistent_master_rank)
    .def_readonly("persistent_materialized",
                  &TRexResult::persistent_materialized)
    .def_readonly("dummy_radius_mean", &TRexResult::dummy_radius_mean)
    .def_readonly("dummy_radius_min", &TRexResult::dummy_radius_min)
    .def_readonly("dummy_radius_max", &TRexResult::dummy_radius_max)
    .def("to_dict", [](const TRexResult& r) {
        py::dict d;
        d["selected_var"]  = r.selected_var;
        d["v_thresh"]      = r.v_thresh;
        d["T_stop"]        = r.T_stop;
        d["num_dummies"]   = r.num_dummies;
        d["L_calibrated"]  = r.L_calibrated;
        d["V"]             = r.V;
        d["FDP_hat_mat"]   = r.FDP_hat_mat;
        d["Phi_mat"]       = r.Phi_mat;
        d["Phi_prime"]     = r.Phi_prime;
        d["coefficient_mean"] = r.coefficient_mean;
        d["K"]             = r.K;
        d["gram_cache_hits"] = r.gram_cache_hits;
        d["gram_cache_misses"] = r.gram_cache_misses;
        d["gram_cache_waits"] = r.gram_cache_waits;
        d["gram_cache_columns"] = r.gram_cache_columns;
        d["persistent_master_rank"] = r.persistent_master_rank;
        d["persistent_materialized"] = r.persistent_materialized;
        d["dummy_radius_mean"] = r.dummy_radius_mean;
        d["dummy_radius_min"] = r.dummy_radius_min;
        d["dummy_radius_max"] = r.dummy_radius_max;
        return d;
    });

  py::class_<TRexSelector>(m, "TRexSelector")
    .def(py::init<const TRexOptions&>(), py::arg("options"))
    .def("run",
        [](TRexSelector& self,
           py::array_t<double, py::array::f_style | py::array::forcecast> X,
           py::array_t<double, py::array::c_style | py::array::forcecast> y) {
          auto bx = X.request();
          auto by = y.request();
          if (bx.ndim != 2) throw std::runtime_error("X must be 2D.");
          if (by.ndim != 1) throw std::runtime_error("y must be 1D.");
          if (bx.shape[0] != by.shape[0]) throw std::runtime_error("X.rows != y.size");
          Eigen::Map<const MatC> Xm(static_cast<double*>(bx.ptr),
              (int)bx.shape[0], (int)bx.shape[1]);
          Eigen::Map<const Vec> ym(static_cast<double*>(by.ptr), (int)by.shape[0]);
          py::gil_scoped_release release;
          return self.run(Xm, ym);
        },
        py::arg("X"), py::arg("y"),
        py::keep_alive<1,2>(), py::keep_alive<1,3>());
}
