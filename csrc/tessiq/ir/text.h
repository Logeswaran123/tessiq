// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The textual IR: the form passes are tested in.
//
//   graph @forward(%a: f32[4,8], %b: f32[4,8]) -> f32[4,8] {
//     %c = add(%a, %b) : f32[4,8]
//     %d = mul(%c, %a) {alpha = 2.0} : f32[4,8]
//     return %d
//   }
//
// Pass tests are written in it: a file in, a file out, and the difference is the test.

#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "tessiq/ir/graph.h"
#include "tessiq/support/result.h"

namespace tessiq::ir {

/// Renders `graph` in the textual form. Round-trips through parse().
std::string print(const Graph& graph);

/// Parses the textual form. Reports the line number on failure.
Result<std::unique_ptr<Graph>> parse(std::string_view text);

} // namespace tessiq::ir
