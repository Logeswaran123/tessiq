// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/support/result.h"

namespace tessiq::passes {

/// One transformation over one graph.
///
/// Every pass documents what it assumes on entry, what it guarantees on exit, and why its rewrite
/// preserves meaning. The last is the part that cannot be recovered from the code.
class Pass {
 public:
  virtual ~Pass() = default;

  Pass(const Pass&) = delete;
  Pass& operator=(const Pass&) = delete;
  Pass(Pass&&) = delete;
  Pass& operator=(Pass&&) = delete;

  /// The name the pass is selected by on the command line and in a pipeline string.
  [[nodiscard]] virtual std::string_view name() const = 0;

  /// One line, for `tessiq-opt --list`.
  [[nodiscard]] virtual std::string_view description() const = 0;

  /// Runs the pass. Returns whether it changed anything.
  virtual Result<bool> run(ir::Graph& graph) = 0;

 protected:
  Pass() = default;
};

/// Passes by name, so a pipeline is a string rather than code. Adding a pass does not mean editing
/// a dispatcher, and `tessiq-opt --list` shows what exists.
class PassRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Pass>()>;

  static PassRegistry& instance();

  void add(std::string name, Factory factory);

  /// Builds the pass called `name`, or kNotFound naming what is available.
  [[nodiscard]] Result<std::unique_ptr<Pass>> create(std::string_view name) const;

  /// Registered names, sorted.
  [[nodiscard]] std::vector<std::string> names() const;

 private:
  PassRegistry() = default;

  std::vector<std::pair<std::string, Factory>> factories_;
};

/// Registers a pass at static initialisation.
///
///   TESSIQ_REGISTER_PASS(DeadNodeElimination);
// NOLINTBEGIN(bugprone-throwing-static-initialization): registering allocates; failing here means
// the process is out of memory at load, which is not worth recovering from.
#define TESSIQ_REGISTER_PASS(PassType)                                                             \
  namespace {                                                                                      \
  const bool k_registered_##PassType = [] {                                                        \
    ::tessiq::passes::PassRegistry::instance().add(std::string(PassType().name()),                 \
                                                   [] { return std::make_unique<PassType>(); });   \
    return true;                                                                                   \
  }();                                                                                             \
  }
// NOLINTEND(bugprone-throwing-static-initialization)

/// What a pipeline did, for reporting and for tests that assert a pass fired.
struct PassReport {
  std::string pass;
  bool changed = false;
  /// Microseconds, so a slow pass is visible without a profiler.
  std::int64_t elapsed_us = 0;
};

/// Runs passes in order over one graph.
///
/// Verifies the graph after each pass in debug builds, so a pass that corrupts the IR fails in its
/// own test.
class PassManager {
 public:
  void add(std::unique_ptr<Pass> pass);

  /// Builds the named passes and adds them. Accepts "cse,dead-node-elimination" or a single name.
  Result<void> add_from_pipeline(std::string_view pipeline);

  /// Runs every pass once, in order.
  Result<std::vector<PassReport>> run(ir::Graph& graph);

  [[nodiscard]] std::size_t size() const { return passes_.size(); }

 private:
  std::vector<std::unique_ptr<Pass>> passes_;
};

} // namespace tessiq::passes
