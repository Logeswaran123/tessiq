// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace tessiq::passes {

/// Which operations the backend can run itself.
///
/// Partitioning asks this, and everything it says no to is left on the host. The set is data
/// rather than a hardcoded list so that a test can narrow it to two operations and check the
/// clustering, and so that the real answer can come from the kernel registry once one exists.
class OpSupport {
 public:
  OpSupport() = default;

  explicit OpSupport(std::vector<std::string> ops);

  [[nodiscard]] bool supports(std::string_view op) const;

  void add(std::string op);

  void remove(std::string_view op);

  [[nodiscard]] std::size_t size() const { return ops_.size(); }

  /// Supported operations, sorted, for diagnostics and for `tessiq-opt --list-ops`.
  [[nodiscard]] std::vector<std::string> names() const;

 private:
  std::set<std::string, std::less<>> ops_;
};

/// The set partitioning uses when nothing else is supplied.
///
/// Hand-listed for now. Nothing has a native kernel yet, so asking the kernel registry would say
/// no to everything and produce no clusters at all, leaving the partitioner untested until phase 7.
/// A small honest list keeps the clustering exercised in the meantime, and this function is the one
/// place that changes when the registry arrives.
OpSupport& default_op_support();

} // namespace tessiq::passes
