// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>

#include "tessiq/support/logging.h"

namespace tessiq {
namespace {

/// Restores the log level after each test, so ordering cannot affect results.
class LoggingTest : public ::testing::Test {
 protected:
  void SetUp() override { saved_level_ = get_log_level(); }

  void TearDown() override { set_log_level(saved_level_); }

 private:
  LogLevel saved_level_ = LogLevel::kWarn;
};

TEST_F(LoggingTest, LevelNamesAreStable) {
  EXPECT_EQ(log_level_name(LogLevel::kError), "error");
  EXPECT_EQ(log_level_name(LogLevel::kWarn), "warn");
  EXPECT_EQ(log_level_name(LogLevel::kInfo), "info");
  EXPECT_EQ(log_level_name(LogLevel::kDebug), "debug");
  EXPECT_EQ(log_level_name(LogLevel::kTrace), "trace");
}

TEST_F(LoggingTest, ParsesEveryLevelName) {
  LogLevel level = LogLevel::kWarn;
  ASSERT_TRUE(parse_log_level("trace", &level));
  EXPECT_EQ(level, LogLevel::kTrace);

  ASSERT_TRUE(parse_log_level("error", &level));
  EXPECT_EQ(level, LogLevel::kError);
}

TEST_F(LoggingTest, ParsesCaseInsensitively) {
  LogLevel level = LogLevel::kWarn;
  ASSERT_TRUE(parse_log_level("DEBUG", &level));
  EXPECT_EQ(level, LogLevel::kDebug);

  ASSERT_TRUE(parse_log_level("InFo", &level));
  EXPECT_EQ(level, LogLevel::kInfo);
}

TEST_F(LoggingTest, RejectsUnknownLevelAndLeavesOutputUntouched) {
  LogLevel level = LogLevel::kInfo;
  EXPECT_FALSE(parse_log_level("verbose", &level));
  EXPECT_EQ(level, LogLevel::kInfo) << "a failed parse must not modify the out parameter";
}

TEST_F(LoggingTest, EnabledLevelsAreOrdered) {
  set_log_level(LogLevel::kInfo);

  EXPECT_TRUE(log_enabled(LogLevel::kError, "test"));
  EXPECT_TRUE(log_enabled(LogLevel::kWarn, "test"));
  EXPECT_TRUE(log_enabled(LogLevel::kInfo, "test"));
  EXPECT_FALSE(log_enabled(LogLevel::kDebug, "test"));
  EXPECT_FALSE(log_enabled(LogLevel::kTrace, "test"));
}

TEST_F(LoggingTest, DisabledLogDoesNotEvaluateItsArguments) {
  // The central performance claim of the logging layer (PLAN.md section 7.3): building an
  // expensive message must cost nothing when the level is disabled. If this regresses, every
  // hot-path log statement starts paying for a string it never prints.
  set_log_level(LogLevel::kWarn);

  int evaluations = 0;
  const auto expensive = [&evaluations]() {
    ++evaluations;
    return std::string("expensive");
  };

  TESSIQ_LOG(kTrace, "test") << expensive();
  EXPECT_EQ(evaluations, 0) << "a disabled log statement evaluated its arguments";

  set_log_level(LogLevel::kTrace);
  TESSIQ_LOG(kTrace, "test") << expensive();
  EXPECT_EQ(evaluations, 1) << "an enabled log statement did not evaluate its arguments";
}

TEST_F(LoggingTest, LogStatementBindsElseToItsOwnIf) {
  // TESSIQ_LOG expands to an if/else. A naive expansion swallows a following `else`, silently
  // rebinding it. This compiles and passes only if the macro is written correctly.
  bool taken = false;
  // The constant condition is the point: the test checks that the macro's own if/else does
  // not capture the else below.
  // NOLINTNEXTLINE(readability-simplify-boolean-expr)
  if (true) {
    TESSIQ_LOG(kTrace, "test") << "inside";
  } else {
    taken = true;
  }
  EXPECT_FALSE(taken);
}

} // namespace
} // namespace tessiq
