// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
/// Every pass states three things in its documentation: what it assumes on entry, what it
/// guarantees on exit, and why the rewrite preserves meaning. A fusion or rewrite pass without a
/// written legality argument is a latent wrong-results bug, and the argument is the part that
/// cannot be recovered from the code.
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

  /// Runs the pass. Returns whether it changed anything, which is what drives fixpoint iteration.
  virtual Result<bool> run(ir::Graph& graph) = 0;

 protected:
  Pass() = default;
};

/// Passes by name, so a pipeline can be named on the command line rather than compiled in.
///
/// A registry rather than a switch: adding a pass should not mean editing a dispatcher, and
/// `tessiq-opt --list` should not be able to drift from what exists.
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
///   TESSIQ_REGISTER_PASS(DeadCodeElimination);
// NOLINTBEGIN(bugprone-throwing-static-initialization): registering allocates, and a failure
// here means the binary has no passes, which is not a state worth recovering into.
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
/// Verifies the graph after each pass in debug builds. A pass that corrupts the IR should fail in
/// its own test rather than three passes later, where the trail has gone cold.
class PassManager {
 public:
  void add(std::unique_ptr<Pass> pass);

  /// Builds the named passes and adds them. Accepts "dce,cse" or a single name.
  Result<void> add_from_pipeline(std::string_view pipeline);

  /// Runs every pass once, in order.
  Result<std::vector<PassReport>> run(ir::Graph& graph);

  [[nodiscard]] std::size_t size() const { return passes_.size(); }

 private:
  std::vector<std::unique_ptr<Pass>> passes_;
};

} // namespace tessiq::passes
