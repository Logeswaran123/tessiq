// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Which operations are elementwise, under both spellings.

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "tessiq/passes/op_support.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/passes/op_traits.h"

namespace tessiq::passes {
namespace {

TEST(OpTraitsTest, KnowsTheElementwiseOperations) {
  EXPECT_TRUE(is_elementwise("aten.add.Tensor"));
  EXPECT_TRUE(is_elementwise("aten.tanh.default"));
  EXPECT_TRUE(is_elementwise("relu"));
}

TEST(OpTraitsTest, AMatrixMultiplyIsNotElementwise) {
  // It reads positions other than the one it writes, so it cannot join a loop over one index.
  EXPECT_FALSE(is_elementwise("aten.mm.default"));
  EXPECT_FALSE(is_elementwise("mm"));
  EXPECT_FALSE(is_elementwise("aten.convolution.default"));
}

TEST(OpTraitsTest, BothSpellingsOfAnOperationAgree) {
  // The table carries a short name for hand-written IR and an ATen name for imported graphs. A
  // missing spelling only shows as a chain that stops fusing.
  const std::vector<std::pair<std::string, std::string>> pairs{
      {"abs", "aten.abs.default"},
      {"add", "aten.add.Tensor"},
      {"div", "aten.div.Tensor"},
      {"mul", "aten.mul.Tensor"},
      {"neg", "aten.neg.default"},
      {"relu", "aten.relu.default"},
      {"sigmoid", "aten.sigmoid.default"},
      {"sqrt", "aten.sqrt.default"},
      {"sub", "aten.sub.Tensor"},
      {"tanh", "aten.tanh.default"},
  };
  for (const auto& [shortname, aten] : pairs) {
    EXPECT_EQ(is_elementwise(shortname), is_elementwise(aten))
        << shortname << " and " << aten << " disagree";
    EXPECT_TRUE(is_elementwise(shortname)) << shortname;
  }
}

TEST(OpTraitsTest, EverythingTheBackendSupportsHasAWayToRun) {
  // Every claimed operation belongs to a category with an execution path. A new category added
  // without one fails here rather than reaching the fallback.
  for (const std::string& op : default_op_support().names()) {
    const OpInfo* info = find_op(op);
    ASSERT_NE(info, nullptr) << op << " is supported and not in the table";
    switch (info->category) {
    case OpCategory::kElementwise: // a generated kernel, fused with its neighbours
    case OpCategory::kReduction:   // a generated kernel of its own
    case OpCategory::kMovement:    // the device's view registrations, so no code at all
    case OpCategory::kMatmul:      // cuBLAS
    case OpCategory::kCopy:        // the device's clone registration, over a generated kernel
    case OpCategory::kConvolution: // cuDNN
    case OpCategory::kCustom:      // a registered kernel, launched on its own
      break;
    }
  }
}

TEST(OpTraitsTest, TheListIsSortedAndComplete) {
  const std::vector<std::string> names = elementwise_ops();
  EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
  EXPECT_EQ(names.size(), 70U) << "an operation was added or removed without updating this count";
}

TEST(OpTraitsTest, EveryElementwiseRowCanBeEmitted) {
  // The emitter substitutes $0 and $1, so a row whose arity and expression disagree would drop an
  // input or read past its arguments. A bad ops.toml row fails here rather than on the GPU.
  for (const std::string& name : elementwise_ops()) {
    const int arity = elementwise_arity(name);
    const std::string_view expression = elementwise_expression(name);
    EXPECT_GE(arity, 1) << name;
    EXPECT_LE(arity, 2) << name;
    EXPECT_FALSE(expression.empty()) << name << " is elementwise with no expression";
    EXPECT_NE(expression.find("$0"), std::string_view::npos)
        << name << " never reads its first input";
    const bool reads_second = expression.find("$1") != std::string_view::npos;
    EXPECT_EQ(reads_second, arity == 2) << name << " reads $1 but is not binary, or the reverse";
    EXPECT_EQ(expression.find("$2"), std::string_view::npos) << name << " reads a third input";
  }
}

} // namespace
} // namespace tessiq::passes
