// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>

#include "tessiq/support/error.h"

namespace tessiq {
namespace {

TEST(ErrorTest, CarriesCodeAndMessage) {
  const Error error = unsupported("dtype f64 not available");
  EXPECT_EQ(error.code(), ErrorCode::kUnsupported);
  EXPECT_EQ(error.message(), "dtype f64 not available");
}

TEST(ErrorTest, ToStringPrefixesTheCodeName) {
  EXPECT_EQ(invalid_argument("rank must be positive").to_string(),
            "invalid_argument: rank must be positive");
}

TEST(ErrorTest, WithContextPrependsAndPreservesCode) {
  const Error inner = not_found("kernel 'add'");
  const Error outer = inner.with_context("lowering node 'add_3'");

  EXPECT_EQ(outer.code(), ErrorCode::kNotFound) << "context must not change the classification";
  EXPECT_EQ(outer.message(), "lowering node 'add_3': kernel 'add'");
}

TEST(ErrorTest, WithContextNestsOutermostFirst) {
  // Provenance reads outside-in, so the most general context appears first.
  const Error error = not_found("kernel 'add'")
                          .with_context("lowering node 'add_3'")
                          .with_context("compiling graph 'forward'");
  EXPECT_EQ(error.message(), "compiling graph 'forward': lowering node 'add_3': kernel 'add'");
}

TEST(ErrorTest, EveryCodeHasAStableName) {
  // Names appear in user-facing messages and test expectations, so they are part of the contract.
  EXPECT_EQ(error_code_name(ErrorCode::kInvalidArgument), "invalid_argument");
  EXPECT_EQ(error_code_name(ErrorCode::kUnsupported), "unsupported");
  EXPECT_EQ(error_code_name(ErrorCode::kNotFound), "not_found");
  EXPECT_EQ(error_code_name(ErrorCode::kResourceExhausted), "resource_exhausted");
  EXPECT_EQ(error_code_name(ErrorCode::kExternal), "external");
  EXPECT_EQ(error_code_name(ErrorCode::kInternal), "internal");
}

} // namespace
} // namespace tessiq
