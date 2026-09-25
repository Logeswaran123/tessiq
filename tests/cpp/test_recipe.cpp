// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Lowering a graph into a recipe. A recipe outlives the graph it came from, so it must hold no
// pointers into it.

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "tessiq/ir/recipe.h"
#include "tessiq/ir/text.h"
#include "tessiq/passes/pass.h"

namespace tessiq::ir {
namespace {

constexpr std::size_t kAlignment = 512;

std::unique_ptr<Graph> parsed(const std::string& text) {
  auto graph = parse(text);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  return graph.has_value() ? std::move(*graph) : nullptr;
}

/// Parses and runs a pipeline, which is how a graph gets its cluster attributes.
std::unique_ptr<Graph> partitioned(const std::string& text) {
  auto graph = parsed(text);
  if (graph == nullptr) {
    return nullptr;
  }
  passes::PassManager manager;
  EXPECT_TRUE(manager.add_from_pipeline("mark-supported,cluster").has_value());
  EXPECT_TRUE(manager.run(*graph).has_value());
  return graph;
}

const Step& step_for(const Recipe& recipe, std::string_view result) {
  const auto it = std::find_if(recipe.steps.begin(), recipe.steps.end(), [result](const Step& s) {
    return !s.results.empty() && s.results.front() == result;
  });
  EXPECT_NE(it, recipe.steps.end()) << "no step producing %" << result;
  return *it;
}

TEST(RecipeTest, StepsFollowGraphOrder) {
  auto graph = parsed(R"(graph @g(%x: f32[4]) -> f32[4] {
  %a = relu(%x) : f32[4]
  %b = neg(%a) : f32[4]
  %c = abs(%b) : f32[4]
  return %c
}
)");
  ASSERT_NE(graph, nullptr);

  const auto recipe = build_recipe(*graph, kAlignment);
  ASSERT_TRUE(recipe.has_value());
  ASSERT_EQ(recipe->steps.size(), 3U);
  EXPECT_EQ(recipe->steps[0].results.front(), "a");
  EXPECT_EQ(recipe->steps[1].results.front(), "b");
  EXPECT_EQ(recipe->steps[2].results.front(), "c");
  EXPECT_EQ(recipe->steps[0].op, "relu");
}

TEST(RecipeTest, ClusterAssignmentIsCarriedThrough) {
  auto graph = partitioned(R"(graph @g(%a: f32[4], %b: f32[4]) -> f32[4] {
  %x = aten.add.Tensor(%a, %b) : f32[4]
  %y = aten.hypot.default(%x, %b) : f32[4]
  %z = aten.tanh.default(%y) : f32[4]
  return %z
}
)");
  ASSERT_NE(graph, nullptr);

  const auto recipe = build_recipe(*graph, kAlignment);
  ASSERT_TRUE(recipe.has_value());
  EXPECT_NE(step_for(*recipe, "x").cluster, kHost);
  // Nothing runs hypot here, so it stays on the host and splits the two clusters.
  EXPECT_EQ(step_for(*recipe, "y").cluster, kHost);
  EXPECT_NE(step_for(*recipe, "z").cluster, kHost);
  EXPECT_NE(step_for(*recipe, "x").cluster, step_for(*recipe, "z").cluster);
}

TEST(RecipeTest, AnUnpartitionedGraphIsAllHost) {
  // A backend that claims nothing still has a correct recipe.
  auto graph = parsed(R"(graph @g(%a: f32[4], %b: f32[4]) -> f32[4] {
  %x = aten.add.Tensor(%a, %b) : f32[4]
  return %x
}
)");
  ASSERT_NE(graph, nullptr);

  const auto recipe = build_recipe(*graph, kAlignment);
  ASSERT_TRUE(recipe.has_value());
  EXPECT_EQ(recipe->steps.front().cluster, kHost);
}

TEST(RecipeTest, BuffersAreNamedNotPointed) {
  // Recipes are cached, so one has to stay readable after its graph is destroyed.
  auto graph = parsed(R"(graph @g(%x: f32[1024]) -> f32[1024] {
  %a = relu(%x) : f32[1024]
  %b = neg(%a) : f32[1024]
  %c = abs(%b) : f32[1024]
  return %c
}
)");
  ASSERT_NE(graph, nullptr);

  auto recipe = build_recipe(*graph, kAlignment);
  ASSERT_TRUE(recipe.has_value());
  const std::string before = recipe->to_string();

  graph.reset();

  EXPECT_EQ(recipe->to_string(), before);
  EXPECT_FALSE(recipe->buffers.empty());
  EXPECT_EQ(recipe->buffers.front().name, "a");
}

TEST(RecipeTest, ParametersAndResultsAreNotBuffers) {
  auto graph = parsed(R"(graph @g(%x: f32[4]) -> f32[4] {
  %a = relu(%x) : f32[4]
  return %a
}
)");
  ASSERT_NE(graph, nullptr);

  const auto recipe = build_recipe(*graph, kAlignment);
  ASSERT_TRUE(recipe.has_value());
  EXPECT_EQ(recipe->steps.size(), 1U);
  EXPECT_TRUE(recipe->buffers.empty());
  EXPECT_EQ(recipe->arena_bytes, 0U);
}

TEST(RecipeTest, ADynamicShapeStillGetsARecipe) {
  auto graph = parsed(R"(graph @g(%x: f32[?]) -> f32[?] {
  %a = relu(%x) : f32[?]
  %b = neg(%a) : f32[?]
  return %b
}
)");
  ASSERT_NE(graph, nullptr);

  // Placement does not depend on sizes, so unknown extents change nothing but the arena, which is
  // empty.
  const auto recipe = build_recipe(*graph, kAlignment);
  ASSERT_TRUE(recipe.has_value()) << recipe.error().to_string();
  EXPECT_EQ(recipe->steps.size(), 2U);
  EXPECT_EQ(recipe->arena_bytes, 0U);
}

TEST(RecipeTest, ArenaMatchesThePlan) {
  auto graph = parsed(R"(graph @g(%x: f32[1024]) -> f32[1024] {
  %a = relu(%x) : f32[1024]
  %b = neg(%a) : f32[1024]
  %c = abs(%b) : f32[1024]
  %d = tanh(%c) : f32[1024]
  return %d
}
)");
  ASSERT_NE(graph, nullptr);

  const auto recipe = build_recipe(*graph, kAlignment);
  ASSERT_TRUE(recipe.has_value());
  EXPECT_EQ(recipe->naive_bytes, 3U * 4096U);
  EXPECT_EQ(recipe->arena_bytes, 2U * 4096U);
}

} // namespace
} // namespace tessiq::ir
