// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/passes/shape_rules.h"

namespace tessiq::passes {
namespace {

ir::Shape shape(std::vector<std::int64_t> dims) { return ir::Shape(std::move(dims)); }

ir::TensorType f32(std::vector<std::int64_t> dims) {
  return {ir::DType::kF32, shape(std::move(dims))};
}

/// Result types for an op with no attributes, which is every rule except the reductions.
Result<std::vector<ir::TensorType>> infer(std::string_view op,
                                          const std::vector<ir::TensorType>& operands) {
  return infer_result_types(op, operands, {});
}

/// Result types for a reduction over `dims`, spelled the way an imported graph spells it.
Result<std::vector<ir::TensorType>> reduce(std::string_view op,
                                           const ir::TensorType& operand,
                                           std::vector<std::int64_t> dims,
                                           bool keepdim = false) {
  ir::AttributeMap attributes;
  attributes.emplace("arg1", std::move(dims));
  attributes.emplace("arg2", keepdim);
  return infer_result_types(op, {operand}, attributes);
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
  const auto result = infer("aten.relu.default", {f32({2, 3})});
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->size(), 1U);
  EXPECT_EQ(result->front(), f32({2, 3}));
}

TEST(ShapeRuleTest, BinaryElementwiseBroadcasts) {
  const auto result = infer("aten.add.Tensor", {f32({4, 1}), f32({1, 8})});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->front(), f32({4, 8}));
}

TEST(ShapeRuleTest, BinaryElementwiseAcceptsTheScalarAttributeForm) {
  // The importer turns a Python scalar argument into an attribute, so `x * 2.0` arrives here with
  // one operand rather than two.
  const auto result = infer("aten.mul.Tensor", {f32({4, 8})});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->front(), f32({4, 8}));
}

TEST(ShapeRuleTest, MixedDtypesAreRefusedRatherThanPromoted) {
  const auto result = infer("aten.add.Tensor", {f32({4}), {ir::DType::kI32, shape({4})}});
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("promotion"), std::string::npos)
      << result.error().message();
}

TEST(ShapeRuleTest, MatmulTakesTheOuterDimensions) {
  const auto result = infer("aten.mm.default", {f32({4, 16}), f32({16, 2})});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->front(), f32({4, 2}));
}

TEST(ShapeRuleTest, MatmulRejectsDisagreeingInnerDimensions) {
  const auto result = infer("aten.mm.default", {f32({4, 16}), f32({8, 2})});
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("inner dimensions"), std::string::npos)
      << result.error().message();
}

TEST(ShapeRuleTest, MatmulAcceptsAnUnknownInnerDimension) {
  // Unknown is not a contradiction, so this has to pass rather than guess.
  const auto result = infer("aten.mm.default", {f32({4, ir::kDynamic}), f32({16, 2})});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->front(), f32({4, 2}));
}

TEST(ShapeRuleTest, TheWrongOperandCountIsAnError) {
  const auto result = infer("aten.relu.default", {f32({4}), f32({4})});
  ASSERT_FALSE(result.has_value());
}

// --- reductions ----------------------------------------------------------------------------------

TEST(ReductionShapeTest, ReducingOneDimensionDropsIt) {
  const auto result = reduce("aten.sum.dim_IntList", f32({4, 8, 3}), {1});
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({4, 3}));
}

TEST(ReductionShapeTest, KeepdimLeavesItAsOne) {
  const auto result = reduce("aten.sum.dim_IntList", f32({4, 8, 3}), {1}, true);
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({4, 1, 3}));
}

TEST(ReductionShapeTest, ANegativeDimensionCountsFromTheEnd) {
  const auto result = reduce("aten.mean.dim", f32({4, 8, 3}), {-1});
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({4, 8}));
}

TEST(ReductionShapeTest, SeveralDimensionsGoAtOnce) {
  const auto result = reduce("aten.sum.dim_IntList", f32({4, 8, 3}), {0, 2});
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({8}));
}

TEST(ReductionShapeTest, AnEmptyDimensionListReducesEverything) {
  // How torch spells sum(x): the dim_IntList form with nothing in the list.
  const auto result = reduce("aten.sum.dim_IntList", f32({4, 8, 3}), {});
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({}));
}

TEST(ReductionShapeTest, NoDimensionAttributeReducesEverything) {
  // How the .default spellings arrive: no list at all. It has to mean the same as an empty one.
  const auto result = infer_result_types("aten.mean.default", {f32({4, 8})}, {});
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({}));
}

TEST(ReductionShapeTest, ReducingEverythingWithKeepdimLeavesOnes) {
  const auto result = reduce("aten.sum.dim_IntList", f32({4, 8}), {}, true);
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({1, 1}));
}

TEST(ReductionShapeTest, ASingleIntegerIsAcceptedForHandWrittenIr) {
  ir::AttributeMap attributes;
  attributes.emplace("arg1", std::int64_t{1});
  const auto result = infer_result_types("aten.amax.default", {f32({4, 8, 3})}, attributes);
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({4, 3}));
}

TEST(ReductionShapeTest, AKeptDimensionStaysUnknownIfItWas) {
  const auto result = reduce("aten.sum.dim_IntList", f32({ir::kDynamic, 8}), {1});
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({ir::kDynamic}));
}

TEST(ReductionShapeTest, AnUnknownDimensionCanStillBeReduced) {
  // The extent is not needed to know the shape of the answer, only to run the kernel.
  const auto result = reduce("aten.sum.dim_IntList", f32({4, ir::kDynamic}), {1});
  ASSERT_TRUE(result.has_value()) << result.error().to_string();
  EXPECT_EQ(result->front(), f32({4}));
}

TEST(ReductionShapeTest, ADimensionOutOfRangeIsAnError) {
  const auto result = reduce("aten.sum.dim_IntList", f32({4, 8}), {2});
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("rank-2"), std::string::npos) << result.error().message();
}

TEST(ReductionShapeTest, TheSameDimensionTwiceIsAnError) {
  // Reducing it twice would drop two axes and leave a shape the operation cannot produce.
  const auto result = reduce("aten.sum.dim_IntList", f32({4, 8, 3}), {1, -2});
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("twice"), std::string::npos) << result.error().message();
}

TEST(ReductionShapeTest, TheWrongOperandCountIsAnError) {
  const auto result = infer_result_types("aten.sum.dim_IntList", {f32({4}), f32({4})}, {});
  ASSERT_FALSE(result.has_value());
}

TEST(ShapeRuleTest, EveryComputingOperationHasARule) {
  // The rules are derived from the table rather than listed beside it, so this holds by
  // construction. It is here because it did not used to: the two were separate lists, and an
  // operation could be fusable, have a kernel, and still have no idea what shape it produced.
  //
  // Movement operations are the exception and are checked the other way below. Each reads its
  // attributes differently and changes layout as well as shape, so there is no rule to derive.
  for (const std::string& name : op_names()) {
    const OpInfo* info = find_op(name);
    ASSERT_NE(info, nullptr) << name;
    if (info->category == OpCategory::kMovement || info->category == OpCategory::kConvolution) {
      // A convolution's spatial extents come from stride, padding and dilation, and imported
      // graphs carry the answer, so there is nothing to derive that ATen has not already worked
      // out. The same reasoning as movement, for a different reason.
      continue;
    }
    EXPECT_TRUE(has_shape_rule(name)) << name << " is in the table with no shape rule";
  }
}

TEST(ShapeRuleTest, AMovementOperationHasNoRuleAndSaysSo) {
  // Deliberate, not missing. shape-infer skips an operation with no rule, which is what these
  // need until something computes both the shape and the layout they produce.
  for (const std::string& name : op_names()) {
    const OpInfo* info = find_op(name);
    ASSERT_NE(info, nullptr) << name;
    if (info->category != OpCategory::kMovement) {
      continue;
    }
    EXPECT_FALSE(has_shape_rule(name)) << name << " has a rule this test did not expect";
  }
}

TEST(ShapeRuleTest, AUnaryOperationCarriesItsOperandTypeThrough) {
  for (const std::string& name : op_names()) {
    const OpInfo* info = find_op(name);
    ASSERT_NE(info, nullptr) << name;
    if (info->category != OpCategory::kElementwise || info->arity != 1) {
      continue;
    }
    const auto result = infer(name, {f32({4, 8})});
    ASSERT_TRUE(result.has_value()) << name << ": " << result.error().to_string();
    EXPECT_EQ(result->front(), f32({4, 8})) << name;
  }
}

} // namespace
} // namespace tessiq::passes
