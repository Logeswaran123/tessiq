// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The kernel that reads a strided tensor and writes a packed one. Emitting the text needs no GPU,
// and what it says is the whole contract: the extents and the steps arrive as arguments, so one
// kernel per rank serves every layout of every shape.

#include <cstddef>
#include <gtest/gtest.h>
#include <string>

#include "tessiq/codegen/pack.h"

namespace tessiq::codegen {
namespace {

/// The source for this width and rank, having already failed the test if there is none.
std::string source_for(std::size_t element_bytes, std::size_t rank) {
  const auto emitted = pack_source(element_bytes, rank);
  EXPECT_TRUE(emitted.has_value()) << (emitted.has_value() ? "" : emitted.error().to_string());
  return emitted.has_value() ? *emitted : std::string{};
}

TEST(PackTest, TheExtentsAndStepsAreArguments) {
  const std::string source = source_for(4, 3);

  // The point of the whole thing. No extent and no step is written into the body, so the same
  // compiled kernel packs a [2,3,4] and a [500,1,7] and a shape nobody knew at compile time.
  EXPECT_NE(source.find("long long e0,\n    long long e1,\n    long long e2"), std::string::npos)
      << source;
  EXPECT_NE(source.find("long long s0,\n    long long s1,\n    long long s2"), std::string::npos)
      << source;
  EXPECT_NE(source.find("p_out[i] = p_in[offset];"), std::string::npos) << source;
}

TEST(PackTest, TheIndexIsTakenApartFromTheLastAxis) {
  const std::string source = source_for(4, 2);

  // Last axis first, because the destination is packed and its fastest axis is the last one.
  const std::size_t inner = source.find("rest % e1");
  const std::size_t outer = source.find("rest % e0");
  ASSERT_NE(inner, std::string::npos) << source;
  ASSERT_NE(outer, std::string::npos) << source;
  EXPECT_LT(inner, outer) << "the last axis has to be unpicked first\n" << source;
}

TEST(PackTest, ACopyIsWrittenForAWidthRatherThanAType) {
  // A pack does no arithmetic, so what the bits mean does not come into it. That is what lets
  // half precision and booleans be packed without the emitter having a spelling for either.
  EXPECT_NE(source_for(1, 2).find("typedef unsigned char word_t;"), std::string::npos);
  EXPECT_NE(source_for(2, 2).find("typedef unsigned short word_t;"), std::string::npos);
  EXPECT_NE(source_for(4, 2).find("typedef unsigned int word_t;"), std::string::npos);
  EXPECT_NE(source_for(8, 2).find("typedef unsigned long long word_t;"), std::string::npos);
}

TEST(PackTest, TheNameSaysWidthAndRank) {
  // The cubin cache is keyed on this, and two kernels that differ only in rank have to not
  // collide, which is the bug that cost us a day in phase 7.
  EXPECT_EQ(pack_name(4, 3), "tessiq_pack_b4_r3");
  EXPECT_NE(pack_name(4, 3), pack_name(4, 2));
  EXPECT_NE(pack_name(4, 3), pack_name(8, 3));
  EXPECT_NE(source_for(4, 3), source_for(4, 2));
}

TEST(PackTest, AScalarHasNoAxesToUnpick) {
  const std::string source = source_for(4, 0);
  EXPECT_NE(source.find("p_out[0] = p_in[0];"), std::string::npos) << source;
  EXPECT_EQ(source.find("long long rest"), std::string::npos) << source;
}

TEST(PackTest, TheBoundsCheckIsThere) {
  EXPECT_NE(source_for(4, 4).find("if (i >= n) return;"), std::string::npos);
}

TEST(PackTest, AWidthNoIntegerHasIsRefused) {
  // Three bytes, or ten. Not a shape anything produces, and refusing says so rather than
  // generating a kernel that moves the wrong number of bytes.
  EXPECT_FALSE(pack_source(3, 2).has_value());
  EXPECT_FALSE(pack_source(0, 2).has_value());
}

TEST(PackTest, ARankDeeperThanTheKernelsGoIsRefused) {
  EXPECT_TRUE(pack_source(4, kMaxPackRank).has_value());
  EXPECT_FALSE(pack_source(4, kMaxPackRank + 1).has_value())
      << "a tensor deeper than this falls back, which is correct and slow";
}

} // namespace
} // namespace tessiq::codegen
