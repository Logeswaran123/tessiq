// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Liveness: when each value has to exist, as a range of node positions.

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "tessiq/ir/liveness.h"
#include "tessiq/ir/text.h"

namespace tessiq::ir {
namespace {

std::unique_ptr<Graph> parsed(const std::string& text) {
  auto graph = parse(text);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  return graph.has_value() ? std::move(*graph) : nullptr;
}

const LiveRange& range_of(const std::vector<LiveRange>& ranges, std::string_view name) {
  const auto it = std::find_if(ranges.begin(), ranges.end(), [name](const LiveRange& range) {
    return range.value->name() == name;
  });
  EXPECT_NE(it, ranges.end()) << "no live range for %" << name;
  return *it;
}

TEST(LivenessTest, AChainDiesAsItGoes) {
  auto graph = parsed(R"(graph @g(%x: f32[4]) -> f32[4] {
  %a = relu(%x) : f32[4]
  %b = neg(%a) : f32[4]
  %c = abs(%b) : f32[4]
  return %c
}
)");
  ASSERT_NE(graph, nullptr);

  const auto ranges = compute_liveness(*graph);
  ASSERT_TRUE(ranges.has_value());

  // %a is written by node 0 and read by node 1, so it is live across both.
  EXPECT_EQ(range_of(*ranges, "a").first, 0U);
  EXPECT_EQ(range_of(*ranges, "a").last, 1U);
  EXPECT_EQ(range_of(*ranges, "b").first, 1U);
  EXPECT_EQ(range_of(*ranges, "b").last, 2U);
}

TEST(LivenessTest, AValueLivesUntilItsLastReader) {
  auto graph = parsed(R"(graph @g(%x: f32[4]) -> f32[4] {
  %a = relu(%x) : f32[4]
  %b = neg(%a) : f32[4]
  %c = abs(%b) : f32[4]
  %d = add(%a, %c) : f32[4]
  return %d
}
)");
  ASSERT_NE(graph, nullptr);

  const auto ranges = compute_liveness(*graph);
  ASSERT_TRUE(ranges.has_value());

  // %a is read again by node 3, so it cannot be reused in between however long the gap is.
  EXPECT_EQ(range_of(*ranges, "a").first, 0U);
  EXPECT_EQ(range_of(*ranges, "a").last, 3U);
}

TEST(LivenessTest, ParametersAndResultsAreMarked) {
  auto graph = parsed(R"(graph @g(%x: f32[4]) -> f32[4] {
  %a = relu(%x) : f32[4]
  return %a
}
)");
  ASSERT_NE(graph, nullptr);

  const auto ranges = compute_liveness(*graph);
  ASSERT_TRUE(ranges.has_value());

  EXPECT_TRUE(range_of(*ranges, "x").is_parameter);
  EXPECT_FALSE(range_of(*ranges, "a").is_parameter);
  EXPECT_TRUE(range_of(*ranges, "a").escapes);
  // The graph has one node, at position 0, so a last of 1 is past every node: a returned value
  // outlives all of them.
  EXPECT_EQ(range_of(*ranges, "a").last, 1U);
}

TEST(LivenessTest, AValueNothingReadsIsLiveOnlyWhereItIsWritten) {
  auto graph = parsed(R"(graph @g(%x: f32[4]) -> f32[4] {
  %dead = neg(%x) : f32[4]
  %a = relu(%x) : f32[4]
  return %a
}
)");
  ASSERT_NE(graph, nullptr);

  const auto ranges = compute_liveness(*graph);
  ASSERT_TRUE(ranges.has_value());
  EXPECT_EQ(range_of(*ranges, "dead").first, 0U);
  EXPECT_EQ(range_of(*ranges, "dead").last, 0U);
}

} // namespace
} // namespace tessiq::ir
