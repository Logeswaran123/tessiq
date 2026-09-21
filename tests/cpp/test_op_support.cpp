// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/text.h"
#include "tessiq/passes/op_support.h"
#include "tessiq/passes/pass.h"

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

/// Takes an operation out of the global support set for the duration of a test and puts it back.
///
/// The set is a global because partitioning reads it without being handed one, so a test that
/// wants a different answer has to change it and change it back, including when the test fails.
class WithoutSupportFor {
 public:
  explicit WithoutSupportFor(std::string op) : op_(std::move(op)) {
    default_op_support().remove(op_);
  }

  WithoutSupportFor(const WithoutSupportFor&) = delete;
  WithoutSupportFor& operator=(const WithoutSupportFor&) = delete;
  WithoutSupportFor(WithoutSupportFor&&) = delete;
  WithoutSupportFor& operator=(WithoutSupportFor&&) = delete;

  ~WithoutSupportFor() { default_op_support().add(op_); }

 private:
  std::string op_;
};

TEST(DefaultOpSupportTest, SupportDecidesWhereAChainBreaks) {
  // Fusability and support are separate questions. Every elementwise operation in the table has a
  // kernel today, so the two agree, and the only way to see them come apart is to take one away.
  // The relu is as fusable as its neighbours either way; what changes is whether the backend can
  // run it, and the chain breaks where support ends rather than where fusability does.
  const std::string_view text = "graph @g(%a: f32[64]) -> f32[64] {\n"
                                "  %w = aten.abs.default(%a) : f32[64]\n"
                                "  %x = aten.neg.default(%w) : f32[64]\n"
                                "  %y = aten.relu.default(%x) : f32[64]\n"
                                "  %z = aten.tanh.default(%y) : f32[64]\n"
                                "  %o = aten.sigmoid.default(%z) : f32[64]\n"
                                "  return %o\n"
                                "}\n";

  const auto fusion_groups = [&text]() {
    auto graph = ir::parse(text);
    EXPECT_TRUE(graph.has_value());
    PassManager manager;
    EXPECT_TRUE(manager.add_from_pipeline("mark-supported,cluster,fuse-elementwise").has_value());
    EXPECT_TRUE(manager.run(**graph).has_value());

    std::set<std::int64_t> groups;
    for (const ir::Node* node : (*graph)->nodes()) {
      const ir::Attribute* attribute = node->attribute("tessiq.fusion");
      if (attribute != nullptr) {
        groups.insert(std::get<std::int64_t>(*attribute));
      }
    }
    return groups.size();
  };

  EXPECT_EQ(fusion_groups(), 1U) << "the whole chain should fuse while relu has a kernel";
  {
    const WithoutSupportFor no_relu("aten.relu.default");
    EXPECT_EQ(fusion_groups(), 2U) << "abs/neg and tanh/sigmoid should fuse either side of relu";
  }
  EXPECT_EQ(fusion_groups(), 1U) << "the support set was not restored";
}

TEST(DefaultOpSupportTest, DoesNotClaimOpsWithNoKernel) {
  const OpSupport& support = default_op_support();
  EXPECT_FALSE(support.supports("aten.mm.default")) << "matmul is not elementwise";
  EXPECT_FALSE(support.supports("aten.sum.dim_IntList")) << "reductions are not in yet";
  EXPECT_FALSE(support.supports("aten.convolution.default"));
}

} // namespace
} // namespace tessiq::passes
