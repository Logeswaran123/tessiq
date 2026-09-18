// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "tessiq/passes/op_support.h"

namespace tessiq::passes {
namespace {

TEST(OpSupportTest, AnEmptySetSupportsNothing) {
  const OpSupport support;
  EXPECT_FALSE(support.supports("aten.add.Tensor"));
  EXPECT_EQ(support.size(), 0U);
}

TEST(OpSupportTest, HoldsWhatItWasGiven) {
  const OpSupport support({"aten.add.Tensor", "aten.mul.Tensor"});
  EXPECT_TRUE(support.supports("aten.add.Tensor"));
  EXPECT_TRUE(support.supports("aten.mul.Tensor"));
  EXPECT_FALSE(support.supports("aten.relu.default"));
}

TEST(OpSupportTest, AddAndRemove) {
  OpSupport support;
  support.add("aten.neg.default");
  EXPECT_TRUE(support.supports("aten.neg.default"));

  support.remove("aten.neg.default");
  EXPECT_FALSE(support.supports("aten.neg.default"));
}

TEST(OpSupportTest, RemovingSomethingAbsentIsHarmless) {
  OpSupport support({"aten.add.Tensor"});
  support.remove("aten.nothing.here");
  EXPECT_EQ(support.size(), 1U);
}

TEST(OpSupportTest, AddingTwiceKeepsOneEntry) {
  OpSupport support;
  support.add("aten.add.Tensor");
  support.add("aten.add.Tensor");
  EXPECT_EQ(support.size(), 1U);
}

TEST(OpSupportTest, NamesAreSorted) {
  const OpSupport support({"aten.mul.Tensor", "aten.add.Tensor", "aten.neg.default"});
  const std::vector<std::string> names = support.names();
  EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
  EXPECT_EQ(names.size(), 3U);
}

TEST(OpSupportTest, LookupDoesNotAllocateAStringToCompare) {
  // The set is transparent, so a string_view can be looked up without constructing a std::string.
  // Worth pinning: partitioning asks this once per node.
  const OpSupport support({"aten.add.Tensor"});
  const std::string_view view = "aten.add.Tensor";
  EXPECT_TRUE(support.supports(view));
}

TEST(DefaultOpSupportTest, ListsTheElementwiseOpsTheImporterProduces) {
  const OpSupport& support = default_op_support();
  EXPECT_TRUE(support.supports("aten.add.Tensor"));
  EXPECT_TRUE(support.supports("aten.mul.Tensor"));
  EXPECT_TRUE(support.supports("aten.relu.default"));
}

TEST(DefaultOpSupportTest, DoesNotClaimOpsWithNoKernel) {
  const OpSupport& support = default_op_support();
  EXPECT_FALSE(support.supports("aten.mm.default")) << "matmul is not elementwise";
  EXPECT_FALSE(support.supports("aten.sum.dim_IntList")) << "reductions are not in yet";
  EXPECT_FALSE(support.supports("aten.convolution.default"));
}

} // namespace
} // namespace tessiq::passes
