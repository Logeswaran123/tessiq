// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The exception boundary (ADR 0001). Result<T> is the currency inside the
// core; Python callers expect exceptions. Translation happens here and nowhere else.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

#include "tessiq/ir/recipe.h"
#include "tessiq/ir/text.h"
#include "tessiq/passes/pass.h"
#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/device.h"
#include "tessiq/support/error.h"
#include "tessiq/support/logging.h"
#include "tessiq/torch/backend.h"
#include "tessiq/torch/importer.h"
#include "tessiq/torch/partition.h"
#include "tessiq/version.h"

namespace py = pybind11;

namespace {

/// Maps an Error onto the Python exception type a caller would expect. Collapsing these into
/// RuntimeError would force callers to match on message text instead of using `except`.
[[noreturn]] void raise_from_error(const tessiq::Error& error) {
  const std::string message = error.to_string();
  switch (error.code()) {
  case tessiq::ErrorCode::kInvalidArgument:
    throw py::value_error(message);
  case tessiq::ErrorCode::kUnsupported:
    throw py::type_error(message);
  case tessiq::ErrorCode::kNotFound:
    throw py::key_error(message);
  case tessiq::ErrorCode::kResourceExhausted:
    throw std::bad_alloc();
  case tessiq::ErrorCode::kExternal:
  case tessiq::ErrorCode::kInternal:
    throw std::runtime_error(message);
  }
  throw std::runtime_error(message);
}

void set_log_level_from_string(const std::string& level_name) {
  tessiq::LogLevel level = tessiq::LogLevel::kWarn;
  if (!tessiq::parse_log_level(level_name, &level)) {
    raise_from_error(
        tessiq::invalid_argument("unknown log level '" + level_name +
                                 "'; expected one of: error, warn, info, debug, trace"));
  }
  tessiq::set_log_level(level);
}

} // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "tessiq native core";

  m.attr("__version__") = tessiq::kVersion;

  m.def(
      "version",
      []() { return std::string(tessiq::kVersion); },
      "Returns the native core version, which must match the Python package version.");

  m.def("set_log_level",
        &set_log_level_from_string,
        py::arg("level"),
        "Sets the native log level. One of: error, warn, info, debug, trace.");

  m.def(
      "get_log_level",
      []() { return std::string(tessiq::log_level_name(tessiq::get_log_level())); },
      "Returns the current native log level as a string.");

  m.def("_init_backend",
        &tessiq::torch_backend::register_allocator,
        "Internal: makes the native backend known to PyTorch. Called during import.");

  m.def(
      "device_count",
      []() { return tessiq::device_count(); },
      "Number of tessiq devices: the CPU reference plus every usable CUDA device.");

  m.def(
      "backend_name",
      [](int index) {
        auto device = tessiq::device_at(index);
        if (!device) {
          raise_from_error(device.error());
        }
        return std::string(tessiq::device_kind_name((*device)->kind()));
      },
      py::arg("index"),
      "Backend behind a device index: 'cpu' for the reference, 'cuda' for a GPU.");

  m.def("empty_cache",
        &tessiq::torch_backend::empty_cache,
        py::arg("index"),
        "Releases cached blocks on a device. Returns the number of bytes handed back.");

  py::class_<tessiq::AllocatorStats>(m, "AllocatorStats")
      .def_readonly("allocated_bytes", &tessiq::AllocatorStats::allocated_bytes)
      .def_readonly("reserved_bytes", &tessiq::AllocatorStats::reserved_bytes)
      .def_readonly("peak_allocated_bytes", &tessiq::AllocatorStats::peak_allocated_bytes)
      .def_readonly("peak_reserved_bytes", &tessiq::AllocatorStats::peak_reserved_bytes)
      .def_readonly("allocation_count", &tessiq::AllocatorStats::allocation_count)
      .def_readonly("deallocation_count", &tessiq::AllocatorStats::deallocation_count)
      .def_readonly("cache_hits", &tessiq::AllocatorStats::cache_hits)
      .def_readonly("cache_misses", &tessiq::AllocatorStats::cache_misses);

  m.def("memory_stats",
        &tessiq::torch_backend::allocator_stats,
        py::arg("index"),
        "Allocator counters for one device.");

  m.def(
      "import_fx",
      [](const py::object& graph_module, const std::string& name) {
        auto graph = tessiq::torch_backend::import_fx(graph_module, name);
        if (!graph) {
          raise_from_error(graph.error());
        }
        // Returns the printed IR rather than the graph itself. Nothing downstream consumes a
        // graph yet, and text is what the tests compare and what tessiq-opt reads.
        return tessiq::ir::print(**graph);
      },
      py::arg("graph_module"),
      py::arg("name") = "forward",
      "Converts a torch.fx.GraphModule into tessiq IR and returns its textual form.");

  m.def(
      "partition_fx",
      [](const py::object& graph_module, const std::string& name) {
        auto result = tessiq::torch_backend::partition_fx(graph_module, name);
        if (!result) {
          raise_from_error(result.error());
        }
        return py::make_tuple(result->ir, result->clusters);
      },
      py::arg("graph_module"),
      py::arg("name") = "forward",
      "Imports and partitions an FX graph. Returns the IR and a map from node name to cluster; "
      "a node absent from the map runs on the host.");

  py::class_<tessiq::ir::Step>(m, "Step")
      .def_readonly("op", &tessiq::ir::Step::op)
      .def_readonly("results", &tessiq::ir::Step::results)
      .def_readonly("cluster", &tessiq::ir::Step::cluster);

  py::class_<tessiq::ir::Buffer>(m, "Buffer")
      .def_readonly("name", &tessiq::ir::Buffer::name)
      .def_readonly("offset", &tessiq::ir::Buffer::offset)
      .def_readonly("size", &tessiq::ir::Buffer::size);

  py::class_<tessiq::ir::Recipe>(m, "Recipe")
      .def_readonly("steps", &tessiq::ir::Recipe::steps)
      .def_readonly("buffers", &tessiq::ir::Recipe::buffers)
      .def_readonly("arena_bytes", &tessiq::ir::Recipe::arena_bytes)
      .def_readonly("naive_bytes", &tessiq::ir::Recipe::naive_bytes)
      .def("__str__", &tessiq::ir::Recipe::to_string);

  m.def(
      "build_recipe_fx",
      [](const py::object& graph_module, const std::string& name, std::size_t alignment) {
        auto result = tessiq::torch_backend::build_recipe_fx(graph_module, name, alignment);
        if (!result) {
          raise_from_error(result.error());
        }
        return std::move(*result);
      },
      py::arg("graph_module"),
      py::arg("name") = "forward",
      py::arg("alignment") = 512,
      "Imports, partitions and lowers an FX graph into a recipe.");

  m.def(
      "run_passes",
      [](const std::string& ir, const std::string& pipeline) {
        auto graph = tessiq::ir::parse(ir);
        if (!graph) {
          raise_from_error(graph.error());
        }
        tessiq::passes::PassManager manager;
        if (const auto added = manager.add_from_pipeline(pipeline); !added) {
          raise_from_error(added.error());
        }
        if (auto reports = manager.run(**graph); !reports) {
          raise_from_error(reports.error());
        }
        return tessiq::ir::print(**graph);
      },
      py::arg("ir"),
      py::arg("pipeline"),
      "Runs a named pass pipeline over textual IR and returns the result. Text in and text out, "
      "for tests and debugging; the compiler keeps the graph in memory instead.");

  m.def(
      "pass_names",
      []() { return tessiq::passes::PassRegistry::instance().names(); },
      "Names of the registered passes.");

  // Exercises the Result -> exception mapping from tests; a mapping nothing tests rots.
  m.def(
      "_raise_test_error",
      [](const std::string& code) {
        if (code == "invalid_argument") {
          raise_from_error(tessiq::invalid_argument("test error"));
        } else if (code == "unsupported") {
          raise_from_error(tessiq::unsupported("test error"));
        } else if (code == "not_found") {
          raise_from_error(tessiq::not_found("test error"));
        }
        raise_from_error(tessiq::internal_error("test error"));
      },
      py::arg("code"),
      "Internal: raises the Python exception mapped from the named ErrorCode. Tests only.");
}
