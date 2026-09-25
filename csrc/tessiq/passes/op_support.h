// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace tessiq::passes {

/// Which operations the backend can run itself.
///
/// `mark-supported` asks this, and leaves everything it says no to on the host. A set rather than
/// a hard-coded list, so a test can narrow it.
class OpSupport {
 public:
  OpSupport() = default;

  explicit OpSupport(std::vector<std::string> ops);

  [[nodiscard]] bool supports(std::string_view op) const;

  void add(std::string op);

  void remove(std::string_view op);

  [[nodiscard]] std::size_t size() const { return ops_.size(); }

  /// Supported operations, sorted, for diagnostics.
  [[nodiscard]] std::vector<std::string> names() const;

 private:
  std::set<std::string, std::less<>> ops_;
};

/// The set partitioning uses when nothing else is supplied.
///
/// Seeded from the generated table with every operation ops.toml gives a kernel, plus each
/// registered one. Mutable and global because `mark-supported` reads it without being handed
/// one; a test that changes it changes it back.
OpSupport& default_op_support();

} // namespace tessiq::passes
