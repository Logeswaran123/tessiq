// Copyright 2026 The Tessera Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The exception boundary (PLAN.md section 7.2, ADR 0001). Result<T> is the currency inside the
// core; Python callers expect exceptions. Translation happens here and nowhere else.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

#include "tessera/support/error.h"
#include "tessera/support/logging.h"
#include "tessera/version.h"

namespace py = pybind11;

namespace {

/// Maps an Error onto the Python exception type a caller would expect. Collapsing these into
/// RuntimeError would force callers to match on message text instead of using `except`.
[[noreturn]] void raise_from_error(const tessera::Error& error) {
  const std::string message = error.to_string();
  switch (error.code()) {
  case tessera::ErrorCode::kInvalidArgument:
    throw py::value_error(message);
  case tessera::ErrorCode::kUnsupported:
    throw py::type_error(message);
  case tessera::ErrorCode::kNotFound:
    throw py::key_error(message);
  case tessera::ErrorCode::kResourceExhausted:
    throw std::bad_alloc();
  case tessera::ErrorCode::kExternal:
  case tessera::ErrorCode::kInternal:
    throw std::runtime_error(message);
  }
  throw std::runtime_error(message);
}

void set_log_level_from_string(const std::string& level_name) {
  tessera::LogLevel level = tessera::LogLevel::kWarn;
  if (!tessera::parse_log_level(level_name, &level)) {
    raise_from_error(
        tessera::invalid_argument("unknown log level '" + level_name +
                                  "'; expected one of: error, warn, info, debug, trace"));
  }
  tessera::set_log_level(level);
}

} // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "tessera native core (see PLAN.md for architecture)";

  m.attr("__version__") = tessera::kVersion;

  m.def(
      "version",
      []() { return std::string(tessera::kVersion); },
      "Returns the native core version, which must match the Python package version.");

  m.def("set_log_level",
        &set_log_level_from_string,
        py::arg("level"),
        "Sets the native log level. One of: error, warn, info, debug, trace.");

  m.def(
      "get_log_level",
      []() { return std::string(tessera::log_level_name(tessera::get_log_level())); },
      "Returns the current native log level as a string.");

  // Exercises the Result -> exception mapping from tests; a mapping nothing tests rots.
  m.def(
      "_raise_test_error",
      [](const std::string& code) {
        if (code == "invalid_argument") {
          raise_from_error(tessera::invalid_argument("test error"));
        } else if (code == "unsupported") {
          raise_from_error(tessera::unsupported("test error"));
        } else if (code == "not_found") {
          raise_from_error(tessera::not_found("test error"));
        }
        raise_from_error(tessera::internal_error("test error"));
      },
      py::arg("code"),
      "Internal: raises the Python exception mapped from the named ErrorCode. Tests only.");
}
