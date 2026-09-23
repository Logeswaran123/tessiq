// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/conv.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

namespace {

/// `aten.convolution` takes (input, weight, bias?, stride, padding, dilation, transposed,
/// output_padding, groups). Tensors became operands and the rest became attributes keyed by the
/// position they held, so the positions below are the ATen schema's own.
constexpr const char* kStride = "arg3";
constexpr const char* kPadding = "arg4";
constexpr const char* kDilation = "arg5";
constexpr const char* kTransposed = "arg6";
constexpr const char* kOutputPadding = "arg7";
constexpr const char* kGroups = "arg8";

/// A pair of spatial values, from an attribute holding one or two of them.
///
/// ATen accepts a single value for both axes as well as one per axis, and the graph carries
/// whichever the model wrote.
std::optional<std::array<std::int64_t, 2>> pair_of(const ir::Node& node, const char* name) {
  const ir::Attribute* attribute = node.attribute(name);
  if (attribute == nullptr) {
    return std::nullopt;
  }
  if (const auto* list = std::get_if<std::vector<std::int64_t>>(attribute)) {
    if (list->size() == 1) {
      return std::array<std::int64_t, 2>{(*list)[0], (*list)[0]};
    }
    if (list->size() == 2) {
      return std::array<std::int64_t, 2>{(*list)[0], (*list)[1]};
    }
    return std::nullopt;
  }
  if (const auto* single = std::get_if<std::int64_t>(attribute)) {
    return std::array<std::int64_t, 2>{*single, *single};
  }
  return std::nullopt;
}

/// The four extents of a packed NCHW tensor, or nothing when it is anything else.
std::optional<std::array<std::int64_t, 4>> nchw(const ir::TensorType& type) {
  const std::vector<std::int64_t>& dims = type.shape().dims();
  if (dims.size() != 4 || !type.shape().is_static() || !type.is_contiguous()) {
    return std::nullopt;
  }
  return std::array<std::int64_t, 4>{dims[0], dims[1], dims[2], dims[3]};
}

bool is_false(const ir::Node& node, const char* name) {
  const ir::Attribute* attribute = node.attribute(name);
  const auto* value = attribute == nullptr ? nullptr : std::get_if<bool>(attribute);
  return value != nullptr && !*value;
}

bool all_zero(const std::optional<std::array<std::int64_t, 2>>& values) {
  return values.has_value() && (*values)[0] == 0 && (*values)[1] == 0;
}

} // namespace

std::vector<std::string> ConvKernel::inputs() const {
  std::vector<std::string> names{input, weight};
  if (bias.has_value()) {
    names.push_back(*bias);
  }
  return names;
}

Result<std::optional<ConvKernel>> extract_conv(const ir::Node& node) {
  const passes::OpInfo* info = passes::find_op(node.op());
  if (info == nullptr || info->category != passes::OpCategory::kConvolution) {
    return std::optional<ConvKernel>{};
  }
  if (node.result_count() != 1 || node.operand_count() < 2 || node.operand_count() > 3) {
    return invalid_argument(node.op() + " takes an input, a weight and an optional bias");
  }

  // A transposed convolution is a different operation that happens to share a name. So is one
  // with output padding, which only means anything when transposed.
  if (!is_false(node, kTransposed) || !all_zero(pair_of(node, kOutputPadding))) {
    return std::optional<ConvKernel>{};
  }

  const ir::Value* input = node.operand(0);
  const ir::Value* weight = node.operand(1);
  const ir::Value* output = node.result(0);
  if (input == nullptr || weight == nullptr || output == nullptr) {
    return internal_error(node.op() + " has an operand or result that is not set");
  }

  const auto input_shape = nchw(input->type());
  const auto weight_shape = nchw(weight->type());
  const auto output_shape = nchw(output->type());
  if (!input_shape || !weight_shape || !output_shape) {
    // Two dimensional, packed and with every extent known. Anything else is a convolution this
    // does not describe, which is not a failure: it runs on the host.
    return std::optional<ConvKernel>{};
  }
  if (input->type().dtype() != weight->type().dtype() ||
      input->type().dtype() != output->type().dtype()) {
    return unsupported(node.op() + " on mixed dtypes; promotion happens before import");
  }

  const auto stride = pair_of(node, kStride);
  const auto padding = pair_of(node, kPadding);
  const auto dilation = pair_of(node, kDilation);
  const ir::Attribute* groups_attribute = node.attribute(kGroups);
  const auto* groups =
      groups_attribute == nullptr ? nullptr : std::get_if<std::int64_t>(groups_attribute);
  if (!stride || !padding || !dilation || groups == nullptr || *groups < 1) {
    return std::optional<ConvKernel>{};
  }

  ConvKernel kernel;
  kernel.input = input->name();
  kernel.weight = weight->name();
  kernel.output = output->name();
  kernel.dtype = output->type().dtype();
  kernel.input_shape = *input_shape;
  kernel.weight_shape = *weight_shape;
  kernel.output_shape = *output_shape;
  kernel.stride = *stride;
  kernel.padding = *padding;
  kernel.dilation = *dilation;
  kernel.groups = *groups;

  if (node.operand_count() == 3) {
    const ir::Value* bias = node.operand(2);
    if (bias == nullptr) {
      return internal_error(node.op() + " has a bias operand that is not set");
    }
    // One value per output channel, which is the only bias a convolution takes.
    const std::vector<std::int64_t>& dims = bias->type().shape().dims();
    if (dims.size() != 1 || dims[0] != kernel.output_shape[1] || !bias->type().is_contiguous()) {
      return std::optional<ConvKernel>{};
    }
    kernel.bias = bias->name();
  }

  return std::optional<ConvKernel>{std::move(kernel)};
}

std::vector<const ir::Node*> conv_nodes(const ir::Graph& graph) {
  std::vector<const ir::Node*> nodes;
  for (const ir::Node* node : graph.nodes()) {
    const passes::OpInfo* info = passes::find_op(node->op());
    if (info != nullptr && info->category == passes::OpCategory::kConvolution) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

} // namespace tessiq::codegen
