// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Pass tests are golden files: textual IR in, textual IR out, with the pipeline named in the file
// itself. Adding a case means adding a file, and reviewing one means reading a diff.

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

#include "tessiq/ir/text.h"
#include "tessiq/passes/pass.h"

namespace tessiq::passes {
namespace {

/// A golden file: a pipeline, an input graph, and the expected output.
///
///   // RUN: cse,dce
///   graph @g(...) { ... }
///   // ---
///   graph @g(...) { ... }
struct GoldenCase {
  std::string pipeline;
  std::string input;
  std::string expected;
};

Result<GoldenCase> load(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return not_found("cannot open " + path.string());
  }

  GoldenCase golden;
  std::string line;
  std::ostringstream input;
  std::ostringstream expected;
  bool after_separator = false;

  while (std::getline(file, line)) {
    if (line.rfind("// RUN:", 0) == 0) {
      golden.pipeline = line.substr(7);
      golden.pipeline.erase(0, golden.pipeline.find_first_not_of(" \t"));
      continue;
    }
    if (line == "// ---") {
      after_separator = true;
      continue;
    }
    (after_separator ? expected : input) << line << "\n";
  }

  if (!after_separator) {
    return invalid_argument(path.string() + " has no '// ---' separator");
  }
  golden.input = input.str();
  golden.expected = expected.str();
  return golden;
}

std::vector<std::filesystem::path> golden_files() {
  std::vector<std::filesystem::path> paths;
  for (const auto& entry : std::filesystem::directory_iterator(TESSIQ_GOLDEN_DIR)) {
    if (entry.path().extension() == ".tsr") {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

class GoldenTest : public ::testing::TestWithParam<std::filesystem::path> {};

TEST_P(GoldenTest, ProducesTheExpectedGraph) {
  auto golden = load(GetParam());
  ASSERT_TRUE(golden.has_value()) << golden.error().to_string();

  auto graph = ir::parse(golden->input);
  ASSERT_TRUE(graph.has_value()) << golden.error().to_string() << graph.error().to_string();

  PassManager manager;
  ASSERT_TRUE(manager.add_from_pipeline(golden->pipeline).has_value())
      << "pipeline '" << golden->pipeline << "'";

  auto reports = manager.run(**graph);
  ASSERT_TRUE(reports.has_value()) << reports.error().to_string();

  EXPECT_EQ(ir::print(**graph), golden->expected);
}

INSTANTIATE_TEST_SUITE_P(Golden,
                         GoldenTest,
                         ::testing::ValuesIn(golden_files()),
                         [](const auto& parameter) { return parameter.param.stem().string(); });

// --- the registry and the manager ----------------------------------------------------------------

TEST(RegistryTest, KnowsTheBuiltInPasses) {
  const std::vector<std::string> names = PassRegistry::instance().names();
  EXPECT_NE(std::find(names.begin(), names.end(), "dce"), names.end());
  EXPECT_NE(std::find(names.begin(), names.end(), "cse"), names.end());
  EXPECT_NE(std::find(names.begin(), names.end(), "const-fold"), names.end());
}

TEST(RegistryTest, NamesAreSorted) {
  const std::vector<std::string> names = PassRegistry::instance().names();
  EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
}

TEST(RegistryTest, UnknownPassListsWhatIsAvailable) {
  auto pass = PassRegistry::instance().create("does-not-exist");
  ASSERT_FALSE(pass.has_value());
  EXPECT_EQ(pass.error().code(), ErrorCode::kNotFound);
  EXPECT_NE(pass.error().message().find("dce"), std::string::npos) << pass.error().message();
}

TEST(PassManagerTest, BuildsAPipelineFromNames) {
  PassManager manager;
  ASSERT_TRUE(manager.add_from_pipeline("cse,dce").has_value());
  EXPECT_EQ(manager.size(), 2U);
}

TEST(PassManagerTest, RejectsAnUnknownPassInThePipeline) {
  PassManager manager;
  EXPECT_FALSE(manager.add_from_pipeline("cse,nope").has_value());
}

TEST(PassManagerTest, ReportsWhetherEachPassChangedAnything) {
  auto graph = ir::parse(R"(graph @g(%a: f32[4]) -> f32[4] {
  %x = neg(%a) : f32[4]
  return %x
}
)");
  ASSERT_TRUE(graph.has_value());

  PassManager manager;
  ASSERT_TRUE(manager.add_from_pipeline("dce").has_value());
  auto reports = manager.run(**graph);
  ASSERT_TRUE(reports.has_value());

  ASSERT_EQ(reports->size(), 1U);
  EXPECT_EQ(reports->front().pass, "dce");
  EXPECT_FALSE(reports->front().changed) << "nothing in this graph is dead";
}

TEST(PassManagerTest, PassesAreIdempotent) {
  // Running a pass twice should be a no-op the second time. A pass that keeps reporting changes
  // would spin forever under fixpoint iteration.
  for (const std::string& name : PassRegistry::instance().names()) {
    auto graph = ir::parse(R"(graph @g(%a: f32[4], %b: f32[4]) -> f32[4] {
  %s0 = add(%a, %b) : f32[4]
  %s1 = add(%a, %b) : f32[4]
  %dead = mul(%a, %a) : f32[4]
  %out = mul(%s0, %s1) : f32[4]
  return %out
}
)");
    ASSERT_TRUE(graph.has_value());

    PassManager first;
    ASSERT_TRUE(first.add_from_pipeline(name).has_value());
    ASSERT_TRUE(first.run(**graph).has_value()) << name;

    PassManager second;
    ASSERT_TRUE(second.add_from_pipeline(name).has_value());
    auto reports = second.run(**graph);
    ASSERT_TRUE(reports.has_value()) << name;
    EXPECT_FALSE(reports->front().changed) << name << " kept changing on a second run";
  }
}

} // namespace
} // namespace tessiq::passes
