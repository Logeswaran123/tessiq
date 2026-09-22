// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The arena planner. Its output is only useful if it is safe, so the central test is a property
// check rather than a fixed expectation: no two values that are live at the same time may share a
// byte.

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tessiq/ir/liveness.h"
#include "tessiq/ir/memory_plan.h"
#include "tessiq/ir/text.h"

namespace tessiq::ir {
namespace {

constexpr std::size_t kAlignment = 512;

std::unique_ptr<Graph> parsed(const std::string& text) {
  auto graph = parse(text);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  return graph.has_value() ? std::move(*graph) : nullptr;
}

/// No two values that are live at the same time may share a byte. This is the property the plan
/// exists to guarantee, and it is worth checking directly rather than inferring from offsets.
void expect_no_overlap(const Graph& graph, const MemoryPlan& plan) {
  const auto ranges = compute_liveness(graph);
  ASSERT_TRUE(ranges.has_value());

  std::unordered_map<const Value*, const LiveRange*> by_value;
  for (const LiveRange& range : *ranges) {
    by_value.emplace(range.value, &range);
  }

  for (std::size_t i = 0; i < plan.allocations.size(); ++i) {
    for (std::size_t j = i + 1; j < plan.allocations.size(); ++j) {
      const Allocation& a = plan.allocations[i];
      const Allocation& b = plan.allocations[j];
      const LiveRange& ra = *by_value.at(a.value);
      const LiveRange& rb = *by_value.at(b.value);

      const bool live_together = ra.first <= rb.last && rb.first <= ra.last;
      const bool share_bytes = a.offset < b.offset + b.size && b.offset < a.offset + a.size;
      EXPECT_FALSE(live_together && share_bytes)
          << "%" << a.value->name() << " [" << ra.first << "," << ra.last << "] at " << a.offset
          << "+" << a.size << " overlaps %" << b.value->name() << " [" << rb.first << "," << rb.last
          << "] at " << b.offset << "+" << b.size;
    }
  }
}

TEST(MemoryPlanTest, AChainReusesOneBuffer) {
  auto graph = parsed(R"(graph @g(%x: f32[1024]) -> f32[1024] {
  %a = relu(%x) : f32[1024]
  %b = neg(%a) : f32[1024]
  %c = abs(%b) : f32[1024]
  %d = tanh(%c) : f32[1024]
  return %d
}
)");
  ASSERT_NE(graph, nullptr);

  const auto plan = plan_memory(*graph, kAlignment);
  ASSERT_TRUE(plan.has_value());

  // Three intermediates of 4096 bytes each, and %c can take the space %a has finished with.
  EXPECT_EQ(plan->naive_bytes, 3U * 4096U);
  EXPECT_EQ(plan->arena_bytes, 2U * 4096U);
  expect_no_overlap(*graph, *plan);
}

TEST(MemoryPlanTest, AValueHeldAcrossTheGraphBlocksItsSpace) {
  auto graph = parsed(R"(graph @g(%x: f32[1024]) -> f32[1024] {
  %a = relu(%x) : f32[1024]
  %b = neg(%a) : f32[1024]
  %c = abs(%b) : f32[1024]
  %d = add(%a, %c) : f32[1024]
  return %d
}
)");
  ASSERT_NE(graph, nullptr);

  const auto plan = plan_memory(*graph, kAlignment);
  ASSERT_TRUE(plan.has_value());

  // %a lives to the end, so it can never be reused, and the arena is larger than the chain above.
  EXPECT_GT(plan->arena_bytes, 2U * 4096U);
  expect_no_overlap(*graph, *plan);
}

TEST(MemoryPlanTest, ParametersAndResultsAreNotPlaced) {
  auto graph = parsed(R"(graph @g(%x: f32[4]) -> f32[4] {
  %a = relu(%x) : f32[4]
  return %a
}
)");
  ASSERT_NE(graph, nullptr);

  const auto plan = plan_memory(*graph, kAlignment);
  ASSERT_TRUE(plan.has_value());
  // The caller owns %x and keeps %a, so the graph owns nothing at all.
  EXPECT_TRUE(plan->allocations.empty());
  EXPECT_EQ(plan->arena_bytes, 0U);
}

TEST(MemoryPlanTest, OffsetsAreAligned) {
  auto graph = parsed(R"(graph @g(%x: f32[3]) -> f32[3] {
  %a = relu(%x) : f32[3]
  %b = neg(%a) : f32[3]
  %c = abs(%b) : f32[3]
  return %c
}
)");
  ASSERT_NE(graph, nullptr);

  const auto plan = plan_memory(*graph, kAlignment);
  ASSERT_TRUE(plan.has_value());
  for (const Allocation& allocation : plan->allocations) {
    EXPECT_EQ(allocation.offset % kAlignment, 0U) << "%" << allocation.value->name();
    // 12 bytes of floats still reserves a whole aligned block, so the next offset stays aligned.
    EXPECT_EQ(allocation.size, kAlignment);
  }
}

TEST(MemoryPlanTest, ADynamicShapeIsCountedRatherThanPlaced) {
  auto graph = parsed(R"(graph @g(%x: f32[?]) -> f32[?] {
  %a = relu(%x) : f32[?]
  %b = neg(%a) : f32[?]
  return %b
}
)");
  ASSERT_NE(graph, nullptr);

  // %a is the only intermediate, and it has no byte count until the graph runs. Refusing the
  // whole plan over it would refuse every graph traced for more than one shape.
  const auto plan = plan_memory(*graph, kAlignment);
  ASSERT_TRUE(plan.has_value()) << plan.error().to_string();
  EXPECT_EQ(plan->unsized, 1U);
  EXPECT_TRUE(plan->allocations.empty());
  EXPECT_EQ(plan->arena_bytes, 0U);
  EXPECT_NE(plan->to_string().find("1 value(s) not sized"), std::string::npos) << plan->to_string();
}

TEST(MemoryPlanTest, AMixedGraphPlacesWhatItCanSize) {
  auto graph = parsed(R"(graph @g(%x: f32[?], %y: f32[256]) -> f32[?] {
  %a = relu(%y) : f32[256]
  %b = neg(%a) : f32[256]
  %c = abs(%x) : f32[?]
  %d = tanh(%c) : f32[?]
  return %d
}
)");
  ASSERT_NE(graph, nullptr);

  const auto plan = plan_memory(*graph, kAlignment);
  ASSERT_TRUE(plan.has_value()) << plan.error().to_string();
  EXPECT_EQ(plan->unsized, 1U);
  ASSERT_EQ(plan->allocations.size(), 2U);
  EXPECT_EQ(plan->allocations[0].value->name(), "a");
  EXPECT_EQ(plan->allocations[1].value->name(), "b");
}

TEST(MemoryPlanTest, PlanningIsReproducible) {
  const std::string text = R"(graph @g(%x: f32[256], %y: f32[256]) -> f32[256] {
  %a = add(%x, %y) : f32[256]
  %b = mul(%a, %x) : f32[256]
  %c = sub(%b, %y) : f32[256]
  %d = div(%c, %a) : f32[256]
  return %d
}
)";
  auto first = parsed(text);
  auto second = parsed(text);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);

  const auto a = plan_memory(*first, kAlignment);
  const auto b = plan_memory(*second, kAlignment);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ(a->to_string(), b->to_string());
  expect_no_overlap(*first, *a);
}

} // namespace
} // namespace tessiq::ir
