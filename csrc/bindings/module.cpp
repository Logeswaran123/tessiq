// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The exception boundary (ADR 0001). Result<T> is the currency inside the
// core; Python callers expect exceptions. Translation happens here and nowhere else.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Teaches pybind11 how to convert a Python torch.Tensor to an at::Tensor. Without it a
// binding taking at::Tensor rejects every tensor it is given.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/csrc/utils/pybind.h>

#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/codegen/timing.h"
#include "tessiq/ir/recipe.h"
#include "tessiq/ir/text.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_registry.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/passes/pass.h"
#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/device.h"
#include "tessiq/support/error.h"
#include "tessiq/support/logging.h"
#include "tessiq/torch/backend.h"
#include "tessiq/torch/custom.h"
#include "tessiq/torch/fused.h"
#include "tessiq/torch/importer.h"
#include "tessiq/torch/kernel.h"
#include "tessiq/torch/partition.h"
#include "tessiq/version.h"

namespace py = pybind11;

using tessiq::codegen::Stream;

namespace {

/// What Python passes for the legacy default stream, which is where every launch goes unless the
/// caller names another. Zero because that is the stream's handle.
constexpr std::uintptr_t kLegacyDefaultStream = 0;

/// The stream whose handle Python holds as `address`, which is what `Stream.cuda_stream` gives.
///
/// The driver hands streams out as pointers and Python can only carry them as integers, so the
/// conversion has to happen somewhere, and this is the boundary where every other one happens.
Stream stream_at(std::uintptr_t address) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return Stream{reinterpret_cast<void*>(address)};
}

} // namespace

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

/// CUDA source for every fusion group in a graph.
///
/// A free function rather than a lambda in the module initialiser, which is one function and gets
/// long enough to trip the complexity limit if everything is written inline.
std::string emit_cuda_source(const std::string& ir) {
  auto graph = tessiq::ir::parse(ir);
  if (!graph) {
    raise_from_error(graph.error());
  }
  const auto groups = tessiq::codegen::fusion_groups(**graph);
  if (groups.empty()) {
    return "// no fusion groups in this graph\n";
  }

  std::string out;
  for (const std::int64_t group : groups) {
    auto kernel = tessiq::codegen::extract_kernel(**graph, group);
    if (!kernel) {
      raise_from_error(kernel.error());
    }
    auto source = tessiq::codegen::cuda_emitter().emit(*kernel);
    if (!source) {
      raise_from_error(source.error());
    }
    out += *source;
    out += "\n";
  }
  return out;
}

} // namespace

/// Raises the Python exception for a failed result, or returns what it holds.
template <typename T>
T value_or_raise(tessiq::Result<T> result) {
  if (!result) {
    raise_from_error(result.error());
  }
  return std::move(result).value();
}

/// Raises the Python exception for a failed result that holds nothing.
void value_or_raise(const tessiq::Result<void>& result) {
  if (!result) {
    raise_from_error(result.error());
  }
}

/// Binds compiled work: the interface every kernel implements, and what each kind adds to it.
///
/// The executor only uses the base class. The kinds are bound as subclasses so that a test or a
/// report can still ask a matmul whether it reads an operand transposed, and pybind11 hands back
/// the most derived type it knows, so `isinstance` works on what a compiled graph returns.
void bind_kernels(py::module_& m) {
  using tessiq::torch_backend::ConvKernel;
  using tessiq::torch_backend::FusedKernel;
  using tessiq::torch_backend::GemmKernel;
  using tessiq::torch_backend::Kernel;
  using tessiq::torch_backend::ReductionKernel;

  py::class_<Kernel, std::shared_ptr<Kernel>>(m, "Kernel")
      .def_property_readonly("kind",
                             [](const Kernel& kernel) { return std::string(kernel.kind()); })
      .def_property_readonly("inputs", &Kernel::inputs)
      .def_property_readonly("output", &Kernel::output)
      .def_property_readonly("members", &Kernel::members)
      .def_property_readonly("packs_inputs", &Kernel::packs_inputs)
      .def_property_readonly("has_dynamic_extent", &Kernel::has_dynamic_extent)
      .def(
          "output_shape",
          [](const Kernel& kernel, const std::vector<at::Tensor>& tensors) {
            return value_or_raise(kernel.output_shape(tensors));
          },
          py::arg("inputs"))
      .def(
          "launch",
          [](const Kernel& kernel,
             const std::vector<at::Tensor>& inputs,
             const at::Tensor& out,
             std::uintptr_t stream) {
            value_or_raise(kernel.launch(inputs, out, stream_at(stream)));
          },
          py::arg("inputs"),
          py::arg("out"),
          py::arg("stream") = kLegacyDefaultStream,
          "Runs the kernel on `stream`. Asynchronous, so the caller orders the result against "
          "whatever reads it.");

  py::class_<FusedKernel, Kernel, std::shared_ptr<FusedKernel>>(m, "FusedKernel")
      .def_property_readonly("group",
                             [](const FusedKernel& kernel) { return kernel.parts().group; })
      .def_property_readonly("elements",
                             [](const FusedKernel& kernel) { return kernel.parts().elements; });

  py::class_<ReductionKernel, Kernel, std::shared_ptr<ReductionKernel>>(m, "ReductionKernel")
      .def_property_readonly("outer",
                             [](const ReductionKernel& kernel) { return kernel.parts().outer; })
      .def_property_readonly("reduce",
                             [](const ReductionKernel& kernel) { return kernel.parts().reduce; })
      .def_property_readonly("inner",
                             [](const ReductionKernel& kernel) { return kernel.parts().inner; })
      .def_property_readonly("blocks", &ReductionKernel::blocks)
      .def_property_readonly("elements", &ReductionKernel::input_elements);

  py::class_<GemmKernel, Kernel, std::shared_ptr<GemmKernel>>(m, "GemmKernel")
      .def_property_readonly("transposed",
                             [](const GemmKernel& kernel) {
                               const auto& description = kernel.parts().description;
                               return std::make_pair(description.lhs.transposed,
                                                     description.rhs.transposed);
                             })
      .def_property_readonly("extents", [](const GemmKernel& kernel) {
        const auto& description = kernel.parts().description;
        return std::make_tuple(description.batch, description.m, description.n, description.k);
      });

  py::class_<ConvKernel, Kernel, std::shared_ptr<ConvKernel>>(m, "ConvKernel")
      .def_property_readonly("algorithm",
                             [](const ConvKernel& kernel) {
                               const auto& plan = kernel.parts().plan;
                               return plan == nullptr ? -1 : plan->algorithm();
                             })
      .def_property_readonly("workspace_bytes", [](const ConvKernel& kernel) {
        const auto& plan = kernel.parts().plan;
        return plan == nullptr ? std::size_t{0} : plan->workspace_bytes();
      });
}

/// Binds adding an operation from Python, and compiling it for the eager path.
void bind_registry(py::module_& m) {
  m.def(
      "register_elementwise",
      [](const std::string& name,
         const std::string& canonical,
         const std::string& expression,
         int arity,
         const std::vector<std::string>& dtypes) {
        tessiq::passes::OpDefinition definition;
        definition.name = name;
        definition.canonical = canonical;
        definition.category = tessiq::passes::OpCategory::kElementwise;
        definition.arity = arity;
        definition.expression = expression;
        for (const std::string& dtype : dtypes) {
          definition.dtypes.push_back(value_or_raise(tessiq::ir::parse_dtype(dtype)));
        }
        value_or_raise(tessiq::passes::register_op(definition));
      },
      py::arg("name"),
      py::arg("canonical"),
      py::arg("expression"),
      py::arg("arity"),
      py::arg("dtypes"),
      "Adds an elementwise operation the compiler claims, fuses and emits like a built-in one.");
  m.def("registered_ops",
        &tessiq::passes::registered_op_names,
        "Every operation added with register_elementwise, sorted.");
  m.def(
      "compile_elementwise",
      [](const std::string& name, int arity, const std::string& dtype, int ordinal) {
        return value_or_raise(tessiq::torch_backend::compile_elementwise(
            name, arity, value_or_raise(tessiq::ir::parse_dtype(dtype)), ordinal));
      },
      py::arg("name"),
      py::arg("arity"),
      py::arg("dtype"),
      py::arg("ordinal"),
      "A kernel running one elementwise operation over operands of the result's shape.");
}

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

  bind_kernels(m);
  bind_registry(m);

  py::class_<tessiq::torch_backend::CompiledGraph>(m, "CompiledGraph")
      .def_readonly("recipe", &tessiq::torch_backend::CompiledGraph::recipe)
      .def_readonly("kernels", &tessiq::torch_backend::CompiledGraph::kernels);

  m.def(
      "build_recipe_fx",
      [](const py::object& graph_module,
         const std::string& name,
         std::size_t alignment,
         int ordinal) {
        auto result =
            tessiq::torch_backend::build_recipe_fx(graph_module, name, alignment, ordinal);
        if (!result) {
          raise_from_error(result.error());
        }
        return std::move(*result);
      },
      py::arg("graph_module"),
      py::arg("name") = "forward",
      py::arg("alignment") = 512,
      py::arg("ordinal") = -1,
      "Imports, partitions, fuses and lowers an FX graph. A negative ordinal skips compiling "
      "kernels.");

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

  m.def("emit_cuda",
        &emit_cuda_source,
        py::arg("ir"),
        "CUDA source for every fusion group in a graph, which must have been through "
        "fuse-elementwise.");

  py::class_<tessiq::codegen::DeviceTimer>(m, "DeviceTimer")
      .def(py::init<int>(), py::arg("ordinal"))
      .def_property_readonly("ready", &tessiq::codegen::DeviceTimer::ready)
      .def("start",
           [](tessiq::codegen::DeviceTimer& timer) {
             if (auto started = timer.start(); !started) {
               raise_from_error(started.error());
             }
           })
      .def("stop",
           [](tessiq::codegen::DeviceTimer& timer) {
             if (auto stopped = timer.stop(); !stopped) {
               raise_from_error(stopped.error());
             }
           })
      .def("microseconds", [](const tessiq::codegen::DeviceTimer& timer) {
        auto elapsed = timer.microseconds();
        if (!elapsed) {
          raise_from_error(elapsed.error());
        }
        return *elapsed;
      });

  m.def("device_timing_available",
        &tessiq::codegen::device_timing_available,
        "Whether this build can time device work with the device's own clock.");

  m.def(
      "default_pipeline",
      []() { return std::string(tessiq::torch_backend::default_pipeline()); },
      "The passes the backend runs to decide what it takes, in order, comma separated.");

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
