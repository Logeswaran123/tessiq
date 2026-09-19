// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "tessiq/ir/types.h"
#include "tessiq/passes/shape_rules.h"

namespace tessiq::passes {
namespace {

ir::Shape shape(std::vector<std::int64_t> dims) { return ir::Shape(std::move(dims)); }

ir::TensorType f32(std::vector<std::int64_t> dims) {
  return {ir::DType::kF32, shape(std::move(dims))};
}

TEST(BroadcastTest, EqualShapesAreUnchanged) {
  const auto result = broadcast_shapes(shape({4, 8}), shape({4, 8}));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dims(), (std::vector<std::int64_t>{4, 8}));
}

TEST(BroadcastTest, OneStretchesToMeetTheOther) {
  const auto result = broadcast_shapes(shape({4, 8}), shape({1, 8}));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dims(), (std::vector<std::int64_t>{4, 8}));
}

TEST(BroadcastTest, ARankDifferenceReadsAsLeadingOnes) {
  const auto result = broadcast_shapes(shape({3, 4, 8}), shape({8}));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dims(), (std::vector<std::int64_t>{3, 4, 8}));
}

TEST(BroadcastTest, AScalarBroadcastsAgainstAnything) {
  const auto result = broadcast_shapes(shape({}), shape({2, 5}));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dims(), (std::vector<std::int64_t>{2, 5}));
}

TEST(BroadcastTest, AnUnknownDimensionStaysUnknown) {
  // Even against a 1. A dimension nobody knows cannot be proven to be 1, and claiming a size that
  // turns out to be wrong puts a false type on a value every later pass trusts.
  const auto result = broadcast_shapes(shape({ir::kDynamic}), shape({1}));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->dims(), (std::vector<std::int64_t>{ir::kDynamic}));
}

TEST(BroadcastTest, IncompatibleDimensionsAreRejectedWithBothSizes) {
  const auto result = broadcast_shapes(shape({4, 8}), shape({3, 8}));
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("4 against 3"), std::string::npos)
      << result.error().message();
}

TEST(ShapeRuleTest, TheTableKnowsWhatItKnows) {
  EXPECT_TRUE(has_shape_rule("aten.relu.default"));
  EXPECT_TRUE(has_shape_rule("relu"));
  // Asking about an op with no rule is the caller's job to avoid, so the question comes first.
  EXPECT_FALSE(has_shape_rule("aten.convolution.default"));
}

TEST(ShapeRuleTest, UnaryElementwiseCarriesItsOperandThrough) {
  const auto result = infer_result_types("aten.relu.default", {f32({2, 3})});
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), 1U);
  EXPECT_EQ(result->front(), f32({2, 3}));
}

TEST(ShapeRuleTest, BinaryElementwiseBroadcasts) {
  const auto result = infer_result_types("aten.add.Tensor", {f32({4, 1}), f32({1, 8})});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->front(), f32({4, 8}));
}

TEST(ShapeRuleTest, BinaryElementwiseAcceptsTheScalarAttributeForm) {
  // The importer turns a Python scalar argument into an attribute, so `x * 2.0` arrives here with
  // one operand rather than two.
  const auto result = infer_result_types("aten.mul.Tensor", {f32({4, 8})});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->front(), f32({4, 8}));
}

TEST(ShapeRuleTest, MixedDtypesAreRefusedRatherThanPromoted) {
  const auto result =
      infer_result_types("aten.add.Tensor", {f32({4}), {ir::DType::kI32, shape({4})}});
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("promotion"), std::string::npos)
      << result.error().message();
}

TEST(ShapeRuleTest, MatmulTakesTheOuterDimensions) {
  const auto result = infer_result_types("aten.mm.default", {f32({4, 16}), f32({16, 2})});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->front(), f32({4, 2}));
}

TEST(ShapeRuleTest, MatmulRejectsDisagreeingInnerDimensions) {
  const auto result = infer_result_types("aten.mm.default", {f32({4, 16}), f32({8, 2})});
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("inner dimensions"), std::string::npos)
      << result.error().message();
}

TEST(ShapeRuleTest, MatmulAcceptsAnUnknownInnerDimension) {
  // Unknown is not a contradiction, so this has to pass rather than guess.
  const auto result = infer_result_types("aten.mm.default", {f32({4, ir::kDynamic}), f32({16, 2})});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->front(), f32({4, 2}));
}

TEST(ShapeRuleTest, TheWrongOperandCountIsAnError) {
  const auto result = infer_result_types("aten.relu.default", {f32({4}), f32({4})});
  ASSERT_FALSE(result.has_value());
}

} // namespace
} // namespace tessiq::passes
