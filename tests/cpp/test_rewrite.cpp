// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Finding a computation written out op by op and replacing it with one call. The graphs below are
// what AOTAutograd emits for RMSNorm and the sigmoid approximation of GELU written in plain
// PyTorch, so the patterns are tested against what they will actually see.

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/text.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/rewrite.h"

namespace tessiq::passes {
namespace {

constexpr std::string_view kRmsNormPattern =
    R"(graph @rms_norm(%x: f32[?], %weight: f32[?]) -> f32[?] {
  %pow = aten.pow.Tensor_Scalar(%x) {arg1 = 2} : f32[?]
  %mean = aten.mean.dim(%pow) {arg1 = [-1], arg2 = true} : f32[?]
  %add = aten.add.Tensor(%mean) {arg1 = "$eps"} : f32[?]
  %rsqrt = aten.rsqrt.default(%add) : f32[?]
  %normed = aten.mul.Tensor(%x, %rsqrt) : f32[?]
  %out = aten.mul.Tensor(%normed, %weight) : f32[?]
  return %out
}
)";

constexpr std::string_view kGeluPattern = R"(graph @fast_gelu(%x: f32[?]) -> f32[?] {
  %scaled = aten.mul.Tensor(%x) {arg1 = 1.702} : f32[?]
  %gate = aten.sigmoid.default(%scaled) : f32[?]
  %out = aten.mul.Tensor(%x, %gate) : f32[?]
  return %out
}
)";

/// RMSNorm as AOTAutograd hands it over, with the power, `keepdim` and the type substituted in.
std::string rms_norm_graph(const std::string& power = "2",
                           const std::string& keepdim = "true",
                           const std::string& dtype = "f32") {
  std::string text = R"(graph @forward(%arg0_1: T[2,8,64], %arg1_1: T[64]) -> T[2,8,64] {
  %pow_1 = aten.pow.Tensor_Scalar(%arg0_1) {arg1 = POWER} : T[2,8,64]
  %mean = aten.mean.dim(%pow_1) {arg1 = [-1], arg2 = KEEPDIM} : T[2,8,1]
  %add = aten.add.Tensor(%mean) {arg1 = 9.9999999999999995e-07} : T[2,8,1]
  %rsqrt = aten.rsqrt.default(%add) : T[2,8,1]
  %mul = aten.mul.Tensor(%arg0_1, %rsqrt) : T[2,8,64]
  %mul_1 = aten.mul.Tensor(%mul, %arg1_1) : T[2,8,64]
  return %mul_1
}
)";
  const auto substitute = [&text](const std::string& from, const std::string& to) {
    for (std::size_t at = text.find(from); at != std::string::npos; at = text.find(from, at)) {
      text.replace(at, from.size(), to);
      at += to.size();
    }
  };
  substitute("POWER", power);
  substitute("KEEPDIM", keepdim);
  substitute(" T[", " " + dtype + "[");
  return text;
}

std::shared_ptr<const ir::Graph> parsed(std::string_view text) {
  auto graph = ir::parse(text);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  if (!graph.has_value()) {
    return nullptr;
  }
  return std::move(*graph);
}

Rewrite rms_norm_rewrite() {
  return {"rewrite-rms-norm",
          parsed(kRmsNormPattern),
          "tessiq.rms_norm.default",
          {{"x", true}, {"weight", true}, {"eps", false}},
          {ir::DType::kF32}};
}

Rewrite gelu_rewrite() {
  return {"rewrite-fast-gelu",
          parsed(kGeluPattern),
          "tessiq.fast_gelu.default",
          {{"x", true}},
          {ir::DType::kF32, ir::DType::kF64}};
}

/// `text` after `pass`, printed, and whether the pass said it changed anything.
std::pair<std::string, bool> after(Rewrite pass, std::string_view text) {
  auto graph = ir::parse(text);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  if (!graph.has_value()) {
    return {"", false};
  }
  const auto changed = pass.run(**graph);
  EXPECT_TRUE(changed.has_value()) << (changed.has_value() ? "" : changed.error().to_string());
  return {ir::print(**graph), changed.has_value() && *changed};
}

TEST(RewriteTest, RmsNormWrittenOutBecomesOneCall) {
  const auto [text, changed] = after(rms_norm_rewrite(), rms_norm_graph());
  EXPECT_TRUE(changed);
  // The call keeps the name and type of the value the pattern returned, takes the tensors in the
  // order the operation does, puts the captured epsilon where the operation's third argument goes,
  // and records the five values it absorbed.
  EXPECT_EQ(text,
            "graph @forward(%arg0_1: f32[2,8,64], %arg1_1: f32[64]) -> f32[2,8,64] {\n"
            "  %mul_1 = tessiq.rms_norm.default(%arg0_1, %arg1_1) {arg2 = "
            "9.9999999999999995e-07, tessiq.absorbed = \"pow_1,mean,add,rsqrt,mul\"} : "
            "f32[2,8,64]\n"
            "  return %mul_1\n}\n");
}

TEST(RewriteTest, AnIntegerAndADoubleOfTheSameNumberMatch) {
  EXPECT_TRUE(after(rms_norm_rewrite(), rms_norm_graph("2.0")).second);
}

TEST(RewriteTest, ADifferentConstantIsLeftAlone) {
  EXPECT_FALSE(after(rms_norm_rewrite(), rms_norm_graph("3")).second);
  EXPECT_FALSE(after(rms_norm_rewrite(), rms_norm_graph("2", "false")).second);
}

TEST(RewriteTest, ATypeTheOperationDoesNotDeclareIsLeftAlone) {
  EXPECT_FALSE(after(rms_norm_rewrite(), rms_norm_graph("2", "true", "f64")).second);
}

TEST(RewriteTest, EitherOrderOfAMultiplyMatches) {
  const auto [text, changed] = after(gelu_rewrite(), R"(graph @g(%x: f32[8]) -> f32[8] {
  %mul = aten.mul.Tensor(%x) {arg1 = 1.702} : f32[8]
  %sigmoid = aten.sigmoid.default(%mul) : f32[8]
  %mul_1 = aten.mul.Tensor(%sigmoid, %x) : f32[8]
  return %mul_1
}
)");
  EXPECT_TRUE(changed);
  EXPECT_NE(text.find("%mul_1 = tessiq.fast_gelu.default(%x)"), std::string::npos) << text;
}

TEST(RewriteTest, AnInteriorValueReadElsewhereIsLeftAlone) {
  // %sigmoid is read by %other as well, so replacing the match would take it away from %other.
  const auto [text, changed] = after(gelu_rewrite(), R"(graph @g(%x: f32[8]) -> (f32[8], f32[8]) {
  %mul = aten.mul.Tensor(%x) {arg1 = 1.702} : f32[8]
  %sigmoid = aten.sigmoid.default(%mul) : f32[8]
  %mul_1 = aten.mul.Tensor(%x, %sigmoid) : f32[8]
  %other = aten.neg.default(%sigmoid) : f32[8]
  return %mul_1, %other
}
)");
  EXPECT_FALSE(changed) << text;
}

TEST(RewriteTest, AnInteriorValueTheGraphReturnsIsLeftAlone) {
  // What a training graph looks like: %sigmoid is returned because the backward graph needs it.
  const auto [text, changed] = after(gelu_rewrite(), R"(graph @g(%x: f32[8]) -> (f32[8], f32[8]) {
  %mul = aten.mul.Tensor(%x) {arg1 = 1.702} : f32[8]
  %sigmoid = aten.sigmoid.default(%mul) : f32[8]
  %mul_1 = aten.mul.Tensor(%x, %sigmoid) : f32[8]
  return %mul_1, %sigmoid
}
)");
  EXPECT_FALSE(changed) << text;
}

TEST(RewriteTest, AnInputHasToBeTheSameValueEveryWhereItAppears) {
  // The pattern reads %x twice. Here the two places hold different values, so it is not RMSNorm.
  const auto [text, changed] =
      after(rms_norm_rewrite(), R"(graph @g(%a: f32[4,8], %b: f32[4,8], %w: f32[8]) -> f32[4,8] {
  %pow = aten.pow.Tensor_Scalar(%a) {arg1 = 2} : f32[4,8]
  %mean = aten.mean.dim(%pow) {arg1 = [-1], arg2 = true} : f32[4,1]
  %add = aten.add.Tensor(%mean) {arg1 = 1e-05} : f32[4,1]
  %rsqrt = aten.rsqrt.default(%add) : f32[4,1]
  %mul = aten.mul.Tensor(%b, %rsqrt) : f32[4,8]
  %mul_1 = aten.mul.Tensor(%mul, %w) : f32[4,8]
  return %mul_1
}
)");
  EXPECT_FALSE(changed) << text;
}

TEST(RewriteTest, EveryMatchIsReplaced) {
  const auto [text, changed] = after(gelu_rewrite(), R"(graph @g(%x: f32[8]) -> f32[8] {
  %mul = aten.mul.Tensor(%x) {arg1 = 1.702} : f32[8]
  %sigmoid = aten.sigmoid.default(%mul) : f32[8]
  %mul_1 = aten.mul.Tensor(%x, %sigmoid) : f32[8]
  %mul_2 = aten.mul.Tensor(%mul_1) {arg1 = 1.702} : f32[8]
  %sigmoid_1 = aten.sigmoid.default(%mul_2) : f32[8]
  %mul_3 = aten.mul.Tensor(%mul_1, %sigmoid_1) : f32[8]
  return %mul_3
}
)");
  EXPECT_TRUE(changed);
  EXPECT_EQ(text,
            "graph @g(%x: f32[8]) -> f32[8] {\n"
            "  %mul_1 = tessiq.fast_gelu.default(%x) {tessiq.absorbed = \"mul,sigmoid\"} : f32[8]\n"
            "  %mul_3 = tessiq.fast_gelu.default(%mul_1) {tessiq.absorbed = \"mul_2,sigmoid_1\"} : "
            "f32[8]\n"
            "  return %mul_3\n}\n");
}

TEST(RewriteTest, APatternSaysWhatItBinds) {
  const auto shape = check_pattern(*parsed(kRmsNormPattern));
  ASSERT_TRUE(shape.has_value()) << shape.error().to_string();
  EXPECT_EQ(shape->inputs, (std::vector<std::string>{"x", "weight"}));
  EXPECT_EQ(shape->captures, (std::vector<std::string>{"eps"}));
}

TEST(RewriteTest, APatternThatCannotBeMatchedIsRefused) {
  const auto refused = [](std::string_view text, std::string_view why) {
    const auto shape = check_pattern(*parsed(text));
    ASSERT_FALSE(shape.has_value());
    EXPECT_NE(shape.error().to_string().find(why), std::string::npos) << shape.error().to_string();
  };
  refused(R"(graph @p(%x: f32[?]) -> f32[?] {
  return %x
}
)",
          "one of its inputs");
  refused(R"(graph @p(%x: f32[?], %y: f32[?]) -> f32[?] {
  %r = aten.relu.default(%x) : f32[?]
  return %r
}
)",
          "input %y of a pattern is never read");
  refused(R"(graph @p(%x: f32[?]) -> f32[?] {
  %n = aten.neg.default(%x) : f32[?]
  %r = aten.relu.default(%x) : f32[?]
  return %r
}
)",
          "%n in a pattern does not lead to the value it returns");
}

} // namespace
} // namespace tessiq::passes
