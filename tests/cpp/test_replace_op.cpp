// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>

#include "tessiq/ir/text.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/replace_op.h"

namespace tessiq::passes {
namespace {

constexpr const char* kTwoSoftmaxes = R"(graph @g(%x: f32[8,128], %y: f64[8,128]) -> f32[8,128] {
  %a = aten._softmax.default(%x) {arg1 = -1, arg2 = false} : f32[8,128]
  %b = aten._softmax.default(%y) {arg1 = 0, arg2 = false} : f64[8,128]
  %c = aten.relu.default(%a) : f32[8,128]
  return %c
}
)";

ReplaceOp softmax_for_f32() {
  return {"override-softmax", "aten._softmax.default", "tessiq.softmax.default", {ir::DType::kF32}};
}

TEST(ReplaceOp, ReplacesTheNodesWithTheTypesAndNothingElse) {
  auto graph = ir::parse(kTwoSoftmaxes);
  ASSERT_TRUE(graph.has_value()) << graph.error().to_string();

  ReplaceOp pass = softmax_for_f32();
  const auto changed = pass.run(**graph);
  ASSERT_TRUE(changed.has_value());
  EXPECT_TRUE(*changed);

  // The float32 node is the custom operation now, with its name, operand and attributes as they
  // were. The float64 one is left alone, and so is the node that reads the replaced one.
  const std::string printed = ir::print(**graph);
  EXPECT_NE(printed.find("%a = tessiq.softmax.default(%x) {arg1 = -1, arg2 = false} : f32[8,128]"),
            std::string::npos)
      << printed;
  EXPECT_NE(printed.find("%b = aten._softmax.default(%y)"), std::string::npos) << printed;
  EXPECT_NE(printed.find("%c = aten.relu.default(%a)"), std::string::npos) << printed;
}

TEST(ReplaceOp, ReportsNoChangeWhenThereIsNothingToReplace) {
  auto graph = ir::parse(kTwoSoftmaxes);
  ASSERT_TRUE(graph.has_value());

  ReplaceOp pass = softmax_for_f32();
  ASSERT_TRUE(pass.run(**graph).has_value());
  const auto again = pass.run(**graph);
  ASSERT_TRUE(again.has_value());
  EXPECT_FALSE(*again);
}

TEST(ReplaceOp, NeedsEveryOperandToHaveTheTypes) {
  auto graph = ir::parse(R"(graph @g(%x: f32[4], %y: f64[4]) -> f32[4] {
  %a = aten.add.Tensor(%x, %y) : f32[4]
  return %a
}
)");
  ASSERT_TRUE(graph.has_value()) << graph.error().to_string();

  ReplaceOp pass{"override-add", "aten.add.Tensor", "tessiq.add.default", {ir::DType::kF32}};
  const auto changed = pass.run(**graph);
  ASSERT_TRUE(changed.has_value());
  EXPECT_FALSE(*changed);
}

TEST(ReplaceOp, DescribesItselfForTheListing) {
  const ReplaceOp pass = softmax_for_f32();
  EXPECT_EQ(pass.name(), "override-softmax");
  EXPECT_EQ(pass.description(),
            "aten._softmax.default becomes tessiq.softmax.default where tessiq.softmax.default "
            "has the types");
}

} // namespace
} // namespace tessiq::passes
