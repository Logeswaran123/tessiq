// Copyright 2026 The Tessera Authors.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tessera/support/error.h"
#include "tessera/support/result.h"

namespace tessera {
namespace {

Result<int> parse_positive(int value) {
  if (value <= 0) {
    return invalid_argument("expected a positive value, got " + std::to_string(value));
  }
  return value;
}

Result<void> require_even(int value) {
  if (value % 2 != 0) {
    return invalid_argument("expected an even value, got " + std::to_string(value));
  }
  return ok();
}

TEST(ResultTest, HoldsAValueOnSuccess) {
  const Result<int> result = parse_positive(7);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result.value(), 7);
  EXPECT_EQ(*result, 7);
}

TEST(ResultTest, HoldsAnErrorOnFailure) {
  const Result<int> result = parse_positive(-1);
  ASSERT_FALSE(result.has_value());
  EXPECT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(result.error().code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(result.error().message(), "expected a positive value, got -1");
}

TEST(ResultTest, SupportsMoveOnlyValues) {
  // Kernels and compiled artifacts are move-only; Result must not force them to be copyable.
  Result<std::unique_ptr<int>> result = std::make_unique<int>(42);
  ASSERT_TRUE(result.has_value());

  const std::unique_ptr<int> moved = std::move(result).value();
  ASSERT_NE(moved, nullptr);
  EXPECT_EQ(*moved, 42);
}

TEST(ResultTest, ArrowOperatorReachesMembers) {
  const Result<std::string> result = std::string("fusion");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->size(), 6U);
}

TEST(ResultTest, ValueOrReturnsFallbackOnError) {
  EXPECT_EQ(parse_positive(5).value_or(-1), 5);
  EXPECT_EQ(parse_positive(0).value_or(-1), -1);
}

TEST(ResultVoidTest, DistinguishesSuccessFromFailure) {
  EXPECT_TRUE(require_even(4).has_value());

  const Result<void> failure = require_even(3);
  ASSERT_FALSE(failure.has_value());
  EXPECT_EQ(failure.error().code(), ErrorCode::kInvalidArgument);
}

// --- Propagation macros -------------------------------------------------------------------------

Result<int> doubled_if_positive(int value) {
  TESSERA_ASSIGN_OR_RETURN(const int parsed, parse_positive(value));
  return parsed * 2;
}

Result<void> validate_all(const std::vector<int>& values) {
  for (const int value : values) {
    TESSERA_RETURN_IF_ERROR(require_even(value));
  }
  return ok();
}

TEST(ResultMacroTest, AssignOrReturnBindsOnSuccess) {
  const Result<int> result = doubled_if_positive(21);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 42);
}

TEST(ResultMacroTest, AssignOrReturnPropagatesTheError) {
  const Result<int> result = doubled_if_positive(-3);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().message(), "expected a positive value, got -3");
}

TEST(ResultMacroTest, TwoAssignmentsInOneScopeDoNotCollide) {
  // Regression guard: the temporary is named after __LINE__, which only works if the macro
  // expands __LINE__ before pasting. A naive `name##__LINE__` produces one identifier for both
  // uses and fails to compile.
  const auto sum = [](int a, int b) -> Result<int> {
    TESSERA_ASSIGN_OR_RETURN(const int first, parse_positive(a));
    TESSERA_ASSIGN_OR_RETURN(const int second, parse_positive(b));
    return first + second;
  };

  const Result<int> result = sum(20, 22);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 42);
}

TEST(ResultMacroTest, ReturnIfErrorStopsAtTheFirstFailure) {
  EXPECT_TRUE(validate_all({2, 4, 6}).has_value());

  const Result<void> result = validate_all({2, 3, 4});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().message(), "expected an even value, got 3");
}

} // namespace
} // namespace tessera
