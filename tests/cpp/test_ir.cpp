// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/text.h"
#include "tessiq/ir/types.h"

namespace tessiq::ir {
namespace {

TensorType f32(std::vector<std::int64_t> dims) { return {DType::kF32, Shape(std::move(dims))}; }

// --- types ---------------------------------------------------------------------------------------

TEST(TypeTest, NamesAndSizesAgree) {
  EXPECT_EQ(dtype_name(DType::kF32), "f32");
  EXPECT_EQ(dtype_size(DType::kF32), 4U);
  EXPECT_TRUE(is_float(DType::kBF16));
  EXPECT_FALSE(is_float(DType::kI64));
}

TEST(TypeTest, ParsesEveryDtypeItPrints) {
  for (const DType dtype : {DType::kF16,
                            DType::kBF16,
                            DType::kF32,
                            DType::kF64,
                            DType::kI1,
                            DType::kI32,
                            DType::kI64}) {
    auto parsed = parse_dtype(dtype_name(dtype));
    ASSERT_TRUE(parsed.has_value()) << dtype_name(dtype);
    EXPECT_EQ(*parsed, dtype);
  }
}

TEST(TypeTest, UnknownDtypeListsWhatIsAvailable) {
  auto parsed = parse_dtype("float32");
  ASSERT_FALSE(parsed.has_value());
  EXPECT_NE(parsed.error().message().find("f32"), std::string::npos) << parsed.error().message();
}

TEST(TypeTest, ScalarIsNotAOneElementVector) {
  EXPECT_TRUE(Shape(std::vector<std::int64_t>{}).is_scalar());
  EXPECT_FALSE(Shape({1}).is_scalar());
  EXPECT_NE(f32({}), f32({1}));
}

TEST(TypeTest, DynamicDimensionsAreTrackedOnTheType) {
  const Shape shape({4, kDynamic});
  EXPECT_FALSE(shape.is_static());
  EXPECT_TRUE(Shape({4, 8}).is_static());
  EXPECT_EQ(TensorType(DType::kF32, shape).to_string(), "f32[4,?]");
}

TEST(TypeTest, ShapePrintsWithoutItsDtype) {
  // TensorType::to_string is this plus a dtype prefix, so the two cannot drift apart.
  EXPECT_EQ(Shape({4, 8}).to_string(), "[4,8]");
  EXPECT_EQ(Shape(std::vector<std::int64_t>{}).to_string(), "[]");
  EXPECT_EQ(Shape({4, kDynamic}).to_string(), "[4,?]");
}

TEST(TypeTest, ElementCountMultipliesTheDimensions) {
  EXPECT_EQ(Shape({4, 8}).element_count(), 32);
  EXPECT_EQ(Shape(std::vector<std::int64_t>{}).element_count(), 1) << "a scalar holds one element";
}

TEST(TypeTest, TypesRoundTripThroughText) {
  for (const std::string text : {"f32[4,8]", "i64[]", "f32[?,16]", "bf16[1,1,1]"}) {
    auto parsed = parse_tensor_type(text);
    ASSERT_TRUE(parsed.has_value()) << text << ": " << parsed.error().to_string();
    EXPECT_EQ(parsed->to_string(), text);
  }
}

TEST(TypeTest, MalformedTypeIsRejected) {
  EXPECT_FALSE(parse_tensor_type("f32").has_value());
  EXPECT_FALSE(parse_tensor_type("f32[4").has_value());
  EXPECT_FALSE(parse_tensor_type("f32[-1]").has_value());
  EXPECT_FALSE(parse_tensor_type("f32[x]").has_value());
}

// --- graph ---------------------------------------------------------------------------------------

TEST(GraphTest, TracksUsersAsOperandsAreAdded) {
  Graph graph("g");
  Value* a = graph.add_param("a", f32({4}));

  EXPECT_FALSE(a->has_users());
  Node* node = graph.add_node("neg");
  node->append_operand(a);
  EXPECT_EQ(a->users().size(), 1U);
  EXPECT_EQ(a->users().front(), node);
}

TEST(GraphTest, AValueUsedTwiceByOneNodeIsListedTwice) {
  // One entry per operand slot: dropping one operand must not unregister the other.
  Graph graph("g");
  Value* a = graph.add_param("a", f32({4}));
  Node* node = graph.add_node("mul");
  node->append_operand(a);
  node->append_operand(a);
  EXPECT_EQ(a->users().size(), 2U);

  node->drop_operands();
  EXPECT_FALSE(a->has_users());
}

TEST(GraphTest, SetOperandMovesTheUse) {
  Graph graph("g");
  Value* a = graph.add_param("a", f32({4}));
  Value* b = graph.add_param("b", f32({4}));
  Node* node = graph.add_node("neg");
  node->append_operand(a);

  node->set_operand(0, b);
  EXPECT_FALSE(a->has_users());
  EXPECT_EQ(b->users().size(), 1U);
}

TEST(GraphTest, ResultsPointBackAtTheirProducer) {
  Graph graph("g");
  Node* node = graph.add_node("split");
  Value* first = node->append_result("x", f32({4}));
  Value* second = node->append_result("y", f32({4}));

  EXPECT_EQ(first->producer(), node);
  EXPECT_EQ(second->producer(), node);
  EXPECT_EQ(first->result_index(), 0U);
  EXPECT_EQ(second->result_index(), 1U);
  EXPECT_EQ(graph.add_param("p", f32({4}))->producer(), nullptr) << "a parameter has no producer";
}

TEST(GraphTest, ReplaceAllUsesRewritesEveryOperandAndTheResults) {
  Graph graph("g");
  Value* a = graph.add_param("a", f32({4}));
  Node* first = graph.add_node("neg");
  first->append_operand(a);
  Value* x = first->append_result("x", f32({4}));

  Node* second = graph.add_node("neg");
  second->append_operand(a);
  Value* y = second->append_result("y", f32({4}));

  Node* user = graph.add_node("add");
  user->append_operand(x);
  user->append_operand(x);
  Value* out = user->append_result("out", f32({4}));
  graph.set_results({x, out});

  ASSERT_TRUE(graph.replace_all_uses(x, y).has_value());
  EXPECT_EQ(user->operand(0), y);
  EXPECT_EQ(user->operand(1), y);
  EXPECT_EQ(graph.results().front(), y) << "graph results are uses too";
  EXPECT_FALSE(x->has_users());
}

TEST(GraphTest, ReplaceRejectsATypeChange) {
  Graph graph("g");
  Node* first = graph.add_node("a");
  Value* x = first->append_result("x", f32({4}));
  Node* second = graph.add_node("b");
  Value* y = second->append_result("y", f32({8}));

  auto result = graph.replace_all_uses(x, y);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("f32[4]"), std::string::npos);
}

TEST(GraphTest, EraseRefusesANodeWithLiveUses) {
  Graph graph("g");
  Node* producer = graph.add_node("neg");
  Value* x = producer->append_result("x", f32({4}));
  Node* consumer = graph.add_node("neg");
  consumer->append_operand(x);

  auto result = graph.erase(producer);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("still used"), std::string::npos)
      << result.error().message();
}

TEST(GraphTest, EraseRefusesANodeProducingAGraphResult) {
  Graph graph("g");
  Node* node = graph.add_node("neg");
  graph.set_results({node->append_result("x", f32({4}))});

  auto result = graph.erase(node);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("graph result"), std::string::npos);
}

TEST(GraphTest, EraseDropsTheNodeAndItsUses) {
  Graph graph("g");
  Value* a = graph.add_param("a", f32({4}));
  Node* node = graph.add_node("neg");
  node->append_operand(a);
  node->append_result("x", f32({4}));

  ASSERT_TRUE(graph.erase(node).has_value());
  EXPECT_EQ(graph.node_count(), 0U);
  EXPECT_FALSE(a->has_users()) << "erasing a node must release its operands";
}

TEST(GraphTest, InsertNodePlacesItBeforeTheGivenNode) {
  Graph graph("g");
  Node* second = graph.add_node("second");
  Node* first = graph.insert_node("first", second);

  const std::vector<Node*> nodes = graph.nodes();
  ASSERT_EQ(nodes.size(), 2U);
  EXPECT_EQ(nodes[0], first);
  EXPECT_EQ(nodes[1], second);
}

TEST(GraphTest, UniqueNameAvoidsCollisions) {
  Graph graph("g");
  graph.add_param("x", f32({4}));
  EXPECT_NE(graph.unique_name("x"), "x");
  EXPECT_EQ(graph.unique_name("y"), "y");
}

// --- text round trip
// ------------------------------------------------------------------------------

/// print(parse(text)) == text for anything the printer emits, which is what makes golden files a
/// fair test: a diff means the pass changed something, not that the printer drifted.
void expect_round_trips(const std::string& text) {
  auto graph = parse(text);
  ASSERT_TRUE(graph.has_value()) << text << "\n-> " << graph.error().to_string();
  const std::string printed = print(**graph);
  EXPECT_EQ(printed, text);

  auto again = parse(printed);
  ASSERT_TRUE(again.has_value()) << again.error().to_string();
  EXPECT_EQ(print(**again), printed) << "printing is not a fixpoint";
}

TEST(TextTest, RoundTripsASimpleGraph) {
  expect_round_trips(R"(graph @g(%a: f32[4,8], %b: f32[4,8]) -> f32[4,8] {
  %c = add(%a, %b) : f32[4,8]
  return %c
}
)");
}

TEST(TextTest, RoundTripsAttributesOfEveryKind) {
  expect_round_trips(R"(graph @g(%a: f32[4]) -> f32[4] {
  %b = op(%a) {flag = true, name = "x", rate = 2.5, shape = [1,2,3], steps = 7} : f32[4]
  return %b
}
)");
}

TEST(TextTest, RoundTripsDynamicShapesAndScalars) {
  expect_round_trips(R"(graph @g(%a: f32[?,16], %s: i64[]) -> (f32[?,16], i64[]) {
  %b = neg(%a) : f32[?,16]
  return %b, %s
}
)");
}

TEST(TextTest, RoundTripsAGraphWithNoNodes) {
  expect_round_trips(R"(graph @g(%a: f32[4]) -> f32[4] {
  return %a
}
)");
}

TEST(TextTest, RoundTripsMultipleResultsFromOneNode) {
  expect_round_trips(R"(graph @g(%a: f32[8]) -> (f32[4], f32[4]) {
  %lo, %hi = split(%a) {axis = 0} : f32[4], f32[4]
  return %lo, %hi
}
)");
}

TEST(TextTest, WholeNumberFloatsStayFloats) {
  // 1.0 must not print as "1", or parsing it back gives an integer attribute and the type of the
  // graph quietly changes.
  expect_round_trips(R"(graph @g() -> f32[] {
  %c = const() {value = 1.0} : f32[]
  return %c
}
)");
}

TEST(TextTest, CommentsAndBlankLinesAreIgnored) {
  auto graph = parse(R"(// leading comment

graph @g(%a: f32[4]) -> f32[4] {
  // what this does
  %b = neg(%a) : f32[4]

  return %b
}
)");
  ASSERT_TRUE(graph.has_value()) << graph.error().to_string();
  EXPECT_EQ((*graph)->node_count(), 1U);
}

TEST(TextTest, ReportsTheLineOfAParseError) {
  auto graph = parse(R"(graph @g(%a: f32[4]) -> f32[4] {
  %b = neg(%missing) : f32[4]
  return %b
}
)");
  ASSERT_FALSE(graph.has_value());
  EXPECT_NE(graph.error().message().find("line 2"), std::string::npos) << graph.error().message();
  EXPECT_NE(graph.error().message().find("undefined"), std::string::npos);
}

TEST(TextTest, RejectsAResultCountMismatch) {
  auto graph = parse(R"(graph @g(%a: f32[4]) -> f32[4] {
  %b, %c = neg(%a) : f32[4]
  return %b
}
)");
  ASSERT_FALSE(graph.has_value());
  EXPECT_NE(graph.error().message().find("result"), std::string::npos) << graph.error().message();
}

TEST(TextTest, RejectsARedefinition) {
  auto graph = parse(R"(graph @g(%a: f32[4]) -> f32[4] {
  %b = neg(%a) : f32[4]
  %b = neg(%a) : f32[4]
  return %b
}
)");
  ASSERT_FALSE(graph.has_value());
  EXPECT_NE(graph.error().message().find("defined twice"), std::string::npos);
}

TEST(TextTest, RejectsEmptyInput) { EXPECT_FALSE(parse("").has_value()); }

// --- verifier ------------------------------------------------------------------------------------

TEST(VerifierTest, AcceptsAWellFormedGraph) {
  Graph graph("g");
  Value* a = graph.add_param("a", f32({4}));
  Node* node = graph.add_node("neg");
  node->append_operand(a);
  graph.set_results({node->append_result("x", f32({4}))});

  EXPECT_TRUE(verify(graph).has_value());
}

TEST(VerifierTest, CatchesAUseBeforeItsDefinition) {
  Graph graph("g");
  Node* producer = graph.add_node("later");
  Value* x = producer->append_result("x", f32({4}));

  Node* consumer = graph.insert_node("earlier", producer);
  consumer->append_operand(x);

  auto result = verify(graph);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("before it is defined"), std::string::npos);
}

TEST(VerifierTest, CatchesDuplicateNames) {
  Graph graph("g");
  graph.add_param("x", f32({4}));
  graph.add_node("neg")->append_result("x", f32({4}));

  auto result = verify(graph);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("duplicate"), std::string::npos);
}

} // namespace
} // namespace tessiq::ir
