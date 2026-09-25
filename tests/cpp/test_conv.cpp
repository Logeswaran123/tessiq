// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Recognising a convolution and describing it exactly. cuDNN reads every tensor the way these
// numbers say, so a misread stride or padding is a wrong answer rather than an error. Needs no GPU.

#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "tessiq/codegen/conv.h"
#include "tessiq/ir/graph.h"
#include "tessiq/ir/text.h"

namespace tessiq::codegen {
namespace {

/// The one convolution in `text`, described.
///
/// Fails the test and returns a default-constructed kernel when there is none, so callers can read
/// fields directly.
ConvKernel describe(std::string_view text) {
  auto graph = ir::parse(text);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  if (!graph.has_value()) {
    return {};
  }
  const auto nodes = conv_nodes(**graph);
  EXPECT_EQ(nodes.size(), 1U);
  if (nodes.size() != 1) {
    return {};
  }
  auto described = extract_conv(*nodes.front());
  EXPECT_TRUE(described.has_value())
      << (described.has_value() ? "" : described.error().to_string());
  if (!described.has_value() || !described->has_value()) {
    ADD_FAILURE() << "no description for this convolution";
    return {};
  }
  return **std::move(described);
}

/// A convolution over `input` giving `output`, with attributes as ATen would carry them.
std::string conv(std::string_view input,
                 std::string_view weight,
                 std::string_view output,
                 std::string_view attributes) {
  return "graph @g(%a: f32" + std::string(input) + ", %w: f32" + std::string(weight) + ") -> f32" +
         std::string(output) + " {\n  %c = aten.convolution.default(%a, %w) {" +
         std::string(attributes) + "} : f32" + std::string(output) + "\n  return %c\n}\n";
}

constexpr std::string_view kPlain = "arg2 = none, arg3 = [1,1], arg4 = [1,1], arg5 = [1,1], "
                                    "arg6 = false, arg7 = [0,0], arg8 = 1";

TEST(ConvTest, AResNetConvolutionIsDescribed) {
  const auto kernel = describe(conv("[1,16,8,8]", "[16,16,3,3]", "[1,16,8,8]", kPlain));
  EXPECT_EQ(kernel.input, "a");
  EXPECT_EQ(kernel.weight, "w");
  EXPECT_EQ(kernel.output, "c");
  EXPECT_FALSE(kernel.bias.has_value()) << "a convolution before a batch norm has no bias";
  EXPECT_EQ(kernel.stride[0], 1);
  EXPECT_EQ(kernel.padding[1], 1);
  EXPECT_EQ(kernel.groups, 1);
  EXPECT_EQ(kernel.output_shape[1], 16);
}

TEST(ConvTest, StridePaddingAndDilationAreReadPerAxis) {
  // Different on each axis, so a description that swapped height and width would be caught.
  const auto kernel = describe(conv("[2,4,32,16]",
                                    "[8,4,3,3]",
                                    "[2,8,15,6]",
                                    "arg2 = none, arg3 = [2,3], arg4 = [0,1], arg5 = [1,2], "
                                    "arg6 = false, arg7 = [0,0], arg8 = 1"));
  EXPECT_EQ(kernel.stride[0], 2);
  EXPECT_EQ(kernel.stride[1], 3);
  EXPECT_EQ(kernel.padding[0], 0);
  EXPECT_EQ(kernel.padding[1], 1);
  EXPECT_EQ(kernel.dilation[0], 1);
  EXPECT_EQ(kernel.dilation[1], 2);
}

TEST(ConvTest, ASingleValueMeansBothAxes) {
  // ATen accepts one number for both axes, and a model that wrote `stride=2` carries one.
  const auto kernel = describe(conv("[1,3,8,8]",
                                    "[4,3,3,3]",
                                    "[1,4,4,4]",
                                    "arg2 = none, arg3 = [2], arg4 = [1], arg5 = [1], "
                                    "arg6 = false, arg7 = [0], arg8 = 1"));
  EXPECT_EQ(kernel.stride[0], 2);
  EXPECT_EQ(kernel.stride[1], 2);
}

TEST(ConvTest, GroupsAreCarried) {
  // A depthwise convolution: one group per channel. Dropping the count would convolve every
  // channel with every filter.
  const auto kernel = describe(conv("[1,8,8,8]",
                                    "[8,1,3,3]",
                                    "[1,8,8,8]",
                                    "arg2 = none, arg3 = [1,1], arg4 = [1,1], arg5 = [1,1], "
                                    "arg6 = false, arg7 = [0,0], arg8 = 8"));
  EXPECT_EQ(kernel.groups, 8);
}

TEST(ConvTest, ABiasBecomesAThirdInput) {
  const auto kernel = describe(
      "graph @g(%a: f32[1,3,8,8], %w: f32[4,3,3,3], %b: f32[4]) -> f32[1,4,6,6] {\n"
      "  %c = aten.convolution.default(%a, %w, %b) {arg3 = [1,1], arg4 = [0,0], arg5 = [1,1], "
      "arg6 = false, arg7 = [0,0], arg8 = 1} : f32[1,4,6,6]\n  return %c\n}\n");
  EXPECT_EQ(kernel.bias.value_or(""), "b");
  EXPECT_EQ(kernel.inputs().size(), 3U);
}

/// Whether `text`'s convolution has no description, as each of the kinds below should not. They
/// run on the host.
bool has_no_description(std::string_view text) {
  auto graph = ir::parse(text);
  EXPECT_TRUE(graph.has_value());
  if (!graph.has_value()) {
    return false;
  }
  const auto nodes = conv_nodes(**graph);
  if (nodes.size() != 1) {
    return false;
  }
  const auto described = extract_conv(*nodes.front());
  EXPECT_TRUE(described.has_value()) << "declining is not a failure";
  return described.has_value() && !described->has_value();
}

TEST(ConvTest, ATransposedConvolutionIsADifferentOperation) {
  EXPECT_TRUE(has_no_description(conv("[1,4,8,8]",
                                      "[4,4,3,3]",
                                      "[1,4,10,10]",
                                      "arg2 = none, arg3 = [1,1], arg4 = [0,0], arg5 = [1,1], "
                                      "arg6 = true, arg7 = [0,0], arg8 = 1")));
}

TEST(ConvTest, AOneDimensionalConvolutionIsNotDescribed) {
  EXPECT_TRUE(has_no_description(conv("[1,3,8]",
                                      "[4,3,3]",
                                      "[1,4,6]",
                                      "arg2 = none, arg3 = [1], arg4 = [0], arg5 = [1], "
                                      "arg6 = false, arg7 = [0], arg8 = 1")));
}

TEST(ConvTest, AShapeNotKnownUntilTheCallIsNotDescribed) {
  // cuDNN descriptors are built at compile time, so they need every extent.
  EXPECT_TRUE(has_no_description(conv("[?,16,8,8]", "[16,16,3,3]", "[?,16,8,8]", kPlain)));
}

TEST(ConvTest, AnElementwiseOperationIsNotOne) {
  auto graph = ir::parse(
      "graph @g(%a: f32[4]) -> f32[4] {\n  %r = aten.relu.default(%a) : f32[4]\n  return %r\n}\n");
  ASSERT_TRUE(graph.has_value());
  EXPECT_TRUE(conv_nodes(**graph).empty());
}

} // namespace
} // namespace tessiq::codegen
