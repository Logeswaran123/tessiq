// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/pass.h"

#include <algorithm>
#include <chrono>

#include "tessiq/ir/text.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq::passes {

namespace {

constexpr std::string_view kLogModule = "passes";

std::vector<std::string_view> split(std::string_view text, char separator) {
  std::vector<std::string_view> parts;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t next = text.find(separator, start);
    const std::size_t end = next == std::string_view::npos ? text.size() : next;
    std::string_view part = text.substr(start, end - start);
    while (!part.empty() && part.front() == ' ') {
      part.remove_prefix(1);
    }
    while (!part.empty() && part.back() == ' ') {
      part.remove_suffix(1);
    }
    if (!part.empty()) {
      parts.push_back(part);
    }
    if (next == std::string_view::npos) {
      break;
    }
    start = next + 1;
  }
  return parts;
}

} // namespace

PassRegistry& PassRegistry::instance() {
  static PassRegistry registry;
  return registry;
}

void PassRegistry::add(std::string name, Factory factory) {
  TESSIQ_CHECK(factory != nullptr, "registering pass '" + name + "' with a null factory");
  factories_.emplace_back(std::move(name), std::move(factory));
}

Result<std::unique_ptr<Pass>> PassRegistry::create(std::string_view name) const {
  for (const auto& [registered, factory] : factories_) {
    if (registered == name) {
      return factory();
    }
  }

  std::string available;
  for (const std::string& known : names()) {
    available += available.empty() ? "" : ", ";
    available += known;
  }
  return not_found("no pass called '" + std::string(name) + "'; available: " + available);
}

std::vector<std::string> PassRegistry::names() const {
  std::vector<std::string> out;
  out.reserve(factories_.size());
  for (const auto& [name, factory] : factories_) {
    out.push_back(name);
  }
  std::sort(out.begin(), out.end());
  return out;
}

void PassManager::add(std::unique_ptr<Pass> pass) {
  TESSIQ_CHECK(pass != nullptr, "adding a null pass");
  passes_.push_back(std::move(pass));
}

Result<void> PassManager::add_from_pipeline(std::string_view pipeline) {
  for (const std::string_view name : split(pipeline, ',')) {
    TESSIQ_ASSIGN_OR_RETURN(auto pass, PassRegistry::instance().create(name));
    passes_.push_back(std::move(pass));
  }
  return ok();
}

Result<std::vector<PassReport>> PassManager::run(ir::Graph& graph) {
  std::vector<PassReport> reports;
  reports.reserve(passes_.size());

  for (const auto& pass : passes_) {
    const auto started = std::chrono::steady_clock::now();
    auto changed = pass->run(graph);
    const auto elapsed = std::chrono::steady_clock::now() - started;

    if (!changed) {
      return std::move(changed).error().with_context(std::string(pass->name()));
    }

    // Debug builds only: verification walks the whole graph, which would dominate a release
    // pipeline. A pass that corrupts the IR is caught here, next to the pass that did it.
#ifndef NDEBUG
    if (const auto valid = ir::verify(graph); !valid) {
      return valid.error().with_context(std::string(pass->name()) + " left the graph invalid");
    }
#endif

    reports.push_back({std::string(pass->name()),
                       *changed,
                       std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()});

    TESSIQ_LOG(kDebug, kLogModule) << pass->name() << (*changed ? ": changed" : ": no change")
                                   << ", " << reports.back().elapsed_us << "us";
  }

  return reports;
}

} // namespace tessiq::passes
