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

/// Claims or unclaims an operation for the duration of a test and restores what it found.
///
/// The set is a global because partitioning reads it without being handed one, so a test that
/// wants a different answer has to change it and change it back, including when the test fails.
class SupportOverride {
 public:
  SupportOverride(std::string op, bool claimed)
      : op_(std::move(op)), restore_(default_op_support().supports(op_)) {
    apply(claimed);
  }

  SupportOverride(const SupportOverride&) = delete;
  SupportOverride& operator=(const SupportOverride&) = delete;
  SupportOverride(SupportOverride&&) = delete;
  SupportOverride& operator=(SupportOverride&&) = delete;

  ~SupportOverride() { apply(restore_); }

 private:
  void apply(bool claimed) {
    if (claimed) {
      default_op_support().add(op_);
    } else {
      default_op_support().remove(op_);
    }
  }

  std::string op_;
  bool restore_;
};

/// How many fusion groups the pipeline puts `text` into, and which operations went into one.
struct Grouping {
  std::size_t groups = 0;
  std::set<std::string> fused;
};

Grouping group(std::string_view text) {
  auto graph = ir::parse(text);
  EXPECT_TRUE(graph.has_value());
  PassManager manager;
  EXPECT_TRUE(manager.add_from_pipeline("mark-supported,cluster,fuse-elementwise").has_value());
  EXPECT_TRUE(manager.run(**graph).has_value());

  Grouping found;
  std::set<std::int64_t> seen;
  for (const ir::Node* node : (*graph)->nodes()) {
    const ir::Attribute* attribute = node->attribute("tessiq.fusion");
    if (attribute != nullptr) {
      seen.insert(std::get<std::int64_t>(*attribute));
      found.fused.insert(node->op());
    }
  }
  found.groups = seen.size();
  return found;
}

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

  EXPECT_EQ(group(text).groups, 1U) << "the whole chain should fuse while relu has a kernel";
  {
    const SupportOverride no_relu("aten.relu.default", false);
    EXPECT_EQ(group(text).groups, 2U) << "abs/neg and tanh/sigmoid should fuse either side of relu";
  }
  EXPECT_EQ(group(text).groups, 1U) << "the support set was not restored";
}

TEST(DefaultOpSupportTest, AReductionStaysOutOfAnElementwiseGroupEvenWhenClaimed) {
  // sum has no kernel yet, so today it is unclaimed and could not fuse for that reason alone.
  // Claiming it here removes that reason and leaves only the category, which is what has to refuse
  // the fusion once the reduction kernel lands. A reduction reads a whole axis to write one
  // element, so it cannot share a loop that gives each thread one index.
  const std::string_view text = "graph @g(%a: f32[4,8]) -> f32[4] {\n"
                                "  %x = aten.abs.default(%a) : f32[4,8]\n"
                                "  %y = aten.neg.default(%x) : f32[4,8]\n"
                                "  %s = aten.sum.dim_IntList(%y) {arg1 = [1]} : f32[4]\n"
                                "  return %s\n"
                                "}\n";

  const SupportOverride claim_sum("aten.sum.dim_IntList", true);
  const Grouping found = group(text);
  EXPECT_EQ(found.groups, 1U) << "abs and neg should still fuse with each other";
  EXPECT_EQ(found.fused.count("aten.sum.dim_IntList"), 0U) << "the reduction joined the loop";
  EXPECT_EQ(found.fused.count("aten.abs.default"), 1U);
}

TEST(DefaultOpSupportTest, DoesNotClaimOpsWithNoKernel) {
  const OpSupport& support = default_op_support();
  EXPECT_FALSE(support.supports("aten.mm.default")) << "matmul has no kernel shape";
  EXPECT_FALSE(support.supports("aten.convolution.default"));
}

TEST(DefaultOpSupportTest, ClaimsReductions) {
  const OpSupport& support = default_op_support();
  EXPECT_TRUE(support.supports("aten.sum.dim_IntList"));
  EXPECT_TRUE(support.supports("aten.mean.dim"));
  EXPECT_TRUE(support.supports("aten.amax.default"));
  EXPECT_TRUE(support.supports("aten.amin.default"));
}

} // namespace
} // namespace tessiq::passes
