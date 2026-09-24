// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Operations added while the program runs. The registry is process-wide and nothing leaves it, so
// every test registers under a name of its own.

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tessiq/codegen/call.h"
#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/ir/graph.h"
#include "tessiq/ir/text.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_registry.h"
#include "tessiq/passes/op_support.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/passes/pass.h"

namespace tessiq::passes {
namespace {

OpDefinition twice(const std::string& name) {
  OpDefinition definition;
  definition.name = name + ".default";
  definition.canonical = name;
  definition.arity = 1;
  definition.expression = "($0 * scalar_t(2))";
  definition.dtypes = {ir::DType::kF32};
  return definition;
}

/// The message a refused definition gets, or a note that it was accepted.
std::string refusal(const OpDefinition& definition) {
  const auto registered = register_op(definition);
  return registered.has_value() ? std::string("accepted") : registered.error().message();
}

OpDefinition with_expression(const std::string& name, const std::string& expression) {
  OpDefinition definition = twice(name);
  definition.expression = expression;
  return definition;
}

std::string marked(const std::string& ir) {
  auto graph = ir::parse(ir);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  PassManager manager;
  EXPECT_TRUE(manager.add_from_pipeline("mark-supported,cluster,fuse-elementwise").has_value());
  EXPECT_TRUE(manager.run(**graph).has_value());
  return ir::print(**graph);
}

TEST(OpRegistryTest, ARegisteredOperationIsFoundLikeAGeneratedOne) {
  const auto registered = register_op(twice("test.found"));
  ASSERT_TRUE(registered.has_value()) << registered.error().to_string();

  const OpInfo* info = find_op("test.found.default");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info, *registered);
  EXPECT_EQ(info->category, OpCategory::kElementwise);
  EXPECT_EQ(info->expression, "($0 * scalar_t(2))");
  EXPECT_EQ(info->canonical, "test.found");
  EXPECT_TRUE(info->accepts(ir::DType::kF32));
  EXPECT_FALSE(info->accepts(ir::DType::kF64));
  EXPECT_TRUE(default_op_support().supports("test.found.default"));
  EXPECT_EQ(find_builtin_op("test.found.default"), nullptr);
}

TEST(OpRegistryTest, RegisteringTheSameDefinitionAgainReturnsTheSameRow) {
  const auto first = register_op(twice("test.again"));
  const auto second = register_op(twice("test.again"));
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*first, *second);
}

TEST(OpRegistryTest, ADifferentDefinitionUnderATakenNameIsRefused) {
  ASSERT_TRUE(register_op(twice("test.taken")).has_value());
  EXPECT_NE(refusal(with_expression("test.taken", "($0 * scalar_t(3))")).find("already registered"),
            std::string::npos);
}

TEST(OpRegistryTest, ABuiltInOperationCannotBeReplaced) {
  OpDefinition definition = twice("unused");
  definition.name = "aten.add.Tensor";
  EXPECT_NE(refusal(definition).find("built-in"), std::string::npos);
}

TEST(OpRegistryTest, AFloatingConstantWrittenBareIsRefused) {
  // A bare 1.702 is a double, and one double turns a float32 kernel into double arithmetic.
  for (const std::string expression : {"($0 * 1.702)", "($0 * 1e-3)", "($0 + .5)", "($0*1.5f)"}) {
    EXPECT_NE(refusal(with_expression("test.bare", expression)).find("scalar_t("),
              std::string::npos)
        << expression;
  }
}

TEST(OpRegistryTest, AConstantInsideScalarTIsAccepted) {
  EXPECT_EQ(refusal(with_expression("test.wrapped", "($0 * scalar_t(1.702))")), "accepted");
  EXPECT_EQ(refusal(with_expression("test.negative", "($0 * scalar_t( -1.702 ))")), "accepted");
}

TEST(OpRegistryTest, AnIntegerConstantAndADigitInAFunctionNameAreFine) {
  // An integer meets a float operand as a float, so nothing promotes. log1p is a name, not a 1.
  EXPECT_EQ(refusal(with_expression("test.integer", "($0 * 2)")), "accepted");
  EXPECT_EQ(refusal(with_expression("test.digit", "log1p($0)")), "accepted");
}

TEST(OpRegistryTest, AnExpressionMustReadEveryOperandAndNoOthers) {
  EXPECT_NE(refusal(with_expression("test.reads", "($1 * scalar_t(2))")).find("reads $1"),
            std::string::npos);

  OpDefinition two = with_expression("test.unread", "($0 * scalar_t(2))");
  two.arity = 2;
  EXPECT_NE(refusal(two).find("never reads $1"), std::string::npos);
}

TEST(OpRegistryTest, ANameTheTextualFormCannotReadIsRefused) {
  OpDefinition definition = twice("unused");
  definition.name = "demo::colon";
  EXPECT_NE(refusal(definition).find("textual IR"), std::string::npos);
}

TEST(OpRegistryTest, OnlyFloatingElementTypesAreAccepted) {
  OpDefinition definition = twice("test.integer_type");
  definition.dtypes = {ir::DType::kI32};
  EXPECT_NE(refusal(definition).find("f32 and f64"), std::string::npos);

  definition.dtypes.clear();
  EXPECT_NE(refusal(definition).find("no element type"), std::string::npos);
}

TEST(OpRegistryTest, ANodeOfATypeItWasNotWrittenForStaysOnTheHost) {
  ASSERT_TRUE(register_op(twice("test.typed")).has_value());

  const std::string f32 = marked(R"(graph @g(%a: f32[8]) -> f32[8] {
  %r = test.typed.default(%a) : f32[8]
  return %r
}
)");
  EXPECT_NE(f32.find("tessiq.supported = true"), std::string::npos) << f32;

  const std::string f64 = marked(R"(graph @g(%a: f64[8]) -> f64[8] {
  %r = test.typed.default(%a) : f64[8]
  return %r
}
)");
  EXPECT_NE(f64.find("tessiq.supported = false"), std::string::npos) << f64;
}

TEST(OpRegistryTest, ItFusesWithItsNeighboursAndTheEmitterWritesItsExpression) {
  ASSERT_TRUE(register_op(twice("test.fused")).has_value());

  auto graph = ir::parse(marked(R"(graph @g(%a: f32[8], %b: f32[8]) -> f32[8] {
  %m = aten.mul.Tensor(%a, %b) : f32[8]
  %t = test.fused.default(%m) : f32[8]
  %r = aten.relu.default(%t) : f32[8]
  return %r
}
)"));
  ASSERT_TRUE(graph.has_value());
  ASSERT_EQ(codegen::fusion_groups(**graph).size(), 1U);

  auto kernel = codegen::extract_kernel(**graph, 0);
  ASSERT_TRUE(kernel.has_value()) << kernel.error().to_string();
  EXPECT_EQ(kernel->body.size(), 3U);
  auto source = codegen::cuda_emitter().emit(*kernel);
  ASSERT_TRUE(source.has_value());
  EXPECT_NE(source->find("(v_m * scalar_t(2))"), std::string::npos) << *source;
}

OpDefinition custom(const std::string& name, int arity, std::vector<std::string> parameters) {
  OpDefinition definition;
  definition.name = name + ".default";
  definition.canonical = name;
  definition.category = OpCategory::kCustom;
  definition.arity = arity;
  definition.parameters = std::move(parameters);
  definition.dtypes = {ir::DType::kF32};
  return definition;
}

/// The arguments of the one call in `ir`, written out: tensors by name, scalars by value.
std::string call_in(const std::string& ir) {
  auto graph = ir::parse(marked(ir));
  EXPECT_TRUE(graph.has_value());
  const std::vector<codegen::CustomCall> calls = codegen::custom_calls(**graph);
  if (calls.size() != 1) {
    return std::to_string(calls.size()) + " calls";
  }
  std::string text;
  for (const codegen::CallArgument& argument : calls.front().arguments) {
    text += text.empty() ? "" : " ";
    if (const auto* operand = std::get_if<codegen::CallOperand>(&argument)) {
      text += "%" + operand->name;
    } else if (const auto* real = std::get_if<double>(&argument)) {
      text += std::to_string(*real);
    } else if (const auto* integer = std::get_if<std::int64_t>(&argument)) {
      text += std::to_string(*integer);
    }
  }
  return text;
}

TEST(OpRegistryTest, ACustomKernelIsClaimedWhenItsShapeIsNotKnownUntilTheCall) {
  ASSERT_TRUE(register_op(custom("test.custom_claimed", 1, {"x"})).has_value());
  const std::string ir = marked(R"(graph @g(%a: f32[?]) -> f32[?] {
  %r = test.custom_claimed.default(%a) : f32[?]
  return %r
}
)");
  EXPECT_NE(ir.find("tessiq.supported = true"), std::string::npos) << ir;
}

TEST(OpRegistryTest, ACustomKernelIsNeverFusedSoItSplitsAChain) {
  ASSERT_TRUE(register_op(custom("test.custom_alone", 1, {"x"})).has_value());
  auto graph = ir::parse(marked(R"(graph @g(%a: f32[8]) -> f32[8] {
  %m = aten.neg.default(%a) : f32[8]
  %c = test.custom_alone.default(%m) : f32[8]
  %r = aten.relu.default(%c) : f32[8]
  return %r
}
)"));
  ASSERT_TRUE(graph.has_value());
  EXPECT_EQ(codegen::fusion_groups(**graph).size(), 2U);
  for (const ir::Node* node : (*graph)->nodes()) {
    if (node->op() == "test.custom_alone.default") {
      EXPECT_EQ(node->attribute("tessiq.fusion"), nullptr);
    }
  }
}

TEST(OpRegistryTest, ACallPutsItsArgumentsBackInTheOrderTheSchemaGivesThem) {
  ASSERT_TRUE(register_op(custom("test.rms", 2, {"x", "weight", "eps"})).has_value());
  ASSERT_TRUE(register_op(custom("test.middle", 2, {"a", "scale", "b"})).has_value());

  // A positional scalar, the same scalar passed by keyword, and one between two tensors.
  EXPECT_EQ(call_in(R"(graph @g(%x: f32[4,8], %w: f32[8]) -> f32[4,8] {
  %r = test.rms.default(%x, %w) {arg2 = 0.5} : f32[4,8]
  return %r
}
)"),
            "%x %w 0.500000");
  EXPECT_EQ(call_in(R"(graph @g(%x: f32[4,8], %w: f32[8]) -> f32[4,8] {
  %r = test.rms.default(%x, %w) {eps = 0.5} : f32[4,8]
  return %r
}
)"),
            "%x %w 0.500000");
  EXPECT_EQ(call_in(R"(graph @g(%a: f32[8], %b: f32[8]) -> f32[8] {
  %r = test.middle.default(%a, %b) {arg1 = 3} : f32[8]
  return %r
}
)"),
            "%a 3 %b");
}

TEST(OpRegistryTest, ACallWithAnArgumentALaunchCannotPassIsLeftOut) {
  ASSERT_TRUE(register_op(custom("test.listed", 1, {"x", "sizes"})).has_value());
  EXPECT_EQ(call_in(R"(graph @g(%x: f32[8]) -> f32[8] {
  %r = test.listed.default(%x) {arg1 = [1, 2]} : f32[8]
  return %r
}
)"),
            "0 calls");
  EXPECT_EQ(call_in(R"(graph @g(%x: f32[8]) -> f32[8] {
  %r = test.listed.default(%x) {undeclared = 2} : f32[8]
  return %r
}
)"),
            "0 calls");
}

TEST(OpRegistryTest, ACustomKernelIsCheckedForWhatItsCallNeeds) {
  OpDefinition with_expression = custom("test.custom_expression", 1, {"x"});
  with_expression.expression = "$0";
  EXPECT_NE(refusal(with_expression).find("own code"), std::string::npos);

  EXPECT_NE(refusal(custom("test.custom_repeated", 2, {"x", "x"})).find("distinct"),
            std::string::npos);
  EXPECT_NE(refusal(custom("test.custom_short", 2, {"x"})).find("fewer parameters"),
            std::string::npos);
  EXPECT_NE(refusal(custom("test.custom_nothing", 0, {})).find("no tensor"), std::string::npos);
}

} // namespace
} // namespace tessiq::passes
