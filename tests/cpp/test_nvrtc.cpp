// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Compiling generated source at run time, and keeping the result between runs. Everything here
// needs the CUDA toolkit and skips without it, because a machine with no toolkit is an ordinary
// machine rather than a failure.

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <system_error>

#include "tessiq/codegen/cubin_cache.h"
#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/nvrtc.h"
#include "tessiq/ir/text.h"
#include "tessiq/passes/pass.h"

namespace tessiq::codegen {
namespace {

/// Every device this project targets is at least this, and the cubin only has to load somewhere
/// for these tests, not on the card in the machine.
constexpr int kMajor = 8;
constexpr int kMinor = 0;

#define TESSIQ_SKIP_WITHOUT_CUDA()                                                                 \
  do {                                                                                             \
    if (!nvrtc_available()) {                                                                      \
      GTEST_SKIP() << "built without the CUDA toolkit";                                            \
    }                                                                                              \
  } while (false)

std::string chain_source() {
  auto graph = ir::parse(R"(graph @g(%a: f32[1024], %b: f32[1024], %c: f32[1024]) -> f32[1024] {
  %m = aten.mul.Tensor(%a, %b) : f32[1024]
  %s = aten.add.Tensor(%m, %c) : f32[1024]
  %r = aten.relu.default(%s) : f32[1024]
  %g = aten.sigmoid.default(%r) : f32[1024]
  return %g
}
)");
  EXPECT_TRUE(graph.has_value());
  passes::PassManager manager;
  EXPECT_TRUE(manager.add_from_pipeline("mark-supported,cluster,fuse-elementwise").has_value());
  EXPECT_TRUE(manager.run(**graph).has_value());

  auto kernel = extract_kernel(**graph, 0);
  EXPECT_TRUE(kernel.has_value());
  auto source = cuda_emitter().emit(*kernel);
  EXPECT_TRUE(source.has_value());
  return source.has_value() ? *source : std::string{};
}

/// A cache directory of its own, so these tests neither read nor disturb the real one.
class ScopedCacheDir {
 public:
  ScopedCacheDir()
      : path_(std::filesystem::temp_directory_path() /
              ("tessiq-test-" + std::to_string(::getpid()))) {
    // Read once into a local. Calling getenv twice puts the null check on a different call than
    // the use, which the analyser objects to and is also just true.
    const char* current = std::getenv("TESSIQ_CACHE_DIR");
    previous_ = current == nullptr ? std::string{} : std::string(current);
    ::setenv("TESSIQ_CACHE_DIR", path_.c_str(), 1);
    reset_cubin_cache_stats();
  }

  ~ScopedCacheDir() {
    if (previous_.empty()) {
      ::unsetenv("TESSIQ_CACHE_DIR");
    } else {
      ::setenv("TESSIQ_CACHE_DIR", previous_.c_str(), 1);
    }
    std::error_code code;
    std::filesystem::remove_all(path_, code);
  }

  ScopedCacheDir(const ScopedCacheDir&) = delete;
  ScopedCacheDir& operator=(const ScopedCacheDir&) = delete;
  ScopedCacheDir(ScopedCacheDir&&) = delete;
  ScopedCacheDir& operator=(ScopedCacheDir&&) = delete;

 private:
  std::filesystem::path path_;
  std::string previous_;
};

TEST(NvrtcTest, CompilesAGeneratedKernel) {
  TESSIQ_SKIP_WITHOUT_CUDA();

  const auto cubin = compile_cuda(chain_source(), "tessiq_fused_0", kMajor, kMinor);
  ASSERT_TRUE(cubin.has_value()) << (cubin.has_value() ? "" : cubin.error().to_string());
  EXPECT_FALSE(cubin->data.empty());
  EXPECT_EQ(cubin->architecture, "sm_80");
}

TEST(NvrtcTest, ACompileErrorCarriesTheCompilersOwnLog) {
  TESSIQ_SKIP_WITHOUT_CUDA();

  // The source is generated, so nobody has a file to look at. The compiler's line and column are
  // the only thing that makes a failure locatable, and summarising them away would be the one
  // mistake this wrapper must not make.
  const auto cubin =
      compile_cuda("__global__ void k() { this_does_not_exist(); }", "broken", kMajor, kMinor);
  ASSERT_FALSE(cubin.has_value());
  EXPECT_NE(cubin.error().message().find("this_does_not_exist"), std::string::npos)
      << cubin.error().message();
}

TEST(NvrtcTest, ReportsItsVersion) {
  TESSIQ_SKIP_WITHOUT_CUDA();
  EXPECT_FALSE(nvrtc_version().empty());
}

TEST(CubinCacheTest, TheSecondCompileComesFromDisk) {
  TESSIQ_SKIP_WITHOUT_CUDA();
  const ScopedCacheDir cache;

  const std::string source = chain_source();
  const auto first = compile_cached(source, "tessiq_fused_0", kMajor, kMinor);
  ASSERT_TRUE(first.has_value()) << (first.has_value() ? "" : first.error().to_string());
  EXPECT_EQ(cubin_cache_stats().misses, 1U);
  EXPECT_EQ(cubin_cache_stats().hits, 0U);

  const auto second = compile_cached(source, "tessiq_fused_0", kMajor, kMinor);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(cubin_cache_stats().hits, 1U);
  EXPECT_EQ(second->data, first->data) << "the cached bytes differ from the compiled ones";
}

TEST(CubinCacheTest, DifferentSourceIsADifferentEntry) {
  TESSIQ_SKIP_WITHOUT_CUDA();
  const ScopedCacheDir cache;

  ASSERT_TRUE(compile_cached(chain_source(), "a", kMajor, kMinor).has_value());
  ASSERT_TRUE(compile_cached(
                  "extern \"C\" __global__ void k(float* p) { p[0] = 1.0f; }", "b", kMajor, kMinor)
                  .has_value());
  EXPECT_EQ(cubin_cache_stats().misses, 2U);
  EXPECT_EQ(cubin_cache_stats().hits, 0U);
}

TEST(CubinCacheTest, DifferentArchitectureIsADifferentEntry) {
  TESSIQ_SKIP_WITHOUT_CUDA();
  const ScopedCacheDir cache;

  // A cubin only loads on the architecture it was built for, so the architecture has to be part
  // of the key rather than a detail of how it was produced.
  const std::string source = chain_source();
  ASSERT_TRUE(compile_cached(source, "k", 8, 0).has_value());
  ASSERT_TRUE(compile_cached(source, "k", 8, 9).has_value());
  EXPECT_EQ(cubin_cache_stats().misses, 2U);
}

TEST(CubinCacheTest, AnUnwritableDirectoryStillCompiles) {
  TESSIQ_SKIP_WITHOUT_CUDA();

  const char* current = std::getenv("TESSIQ_CACHE_DIR");
  const std::string previous = current == nullptr ? std::string{} : std::string(current);
  ::setenv("TESSIQ_CACHE_DIR", "/proc/tessiq-cannot-write-here", 1);
  reset_cubin_cache_stats();

  const auto cubin = compile_cached(chain_source(), "k", kMajor, kMinor);

  if (previous.empty()) {
    ::unsetenv("TESSIQ_CACHE_DIR");
  } else {
    ::setenv("TESSIQ_CACHE_DIR", previous.c_str(), 1);
  }

  // A read-only home directory should make compilation slower, not stop it.
  ASSERT_TRUE(cubin.has_value()) << (cubin.has_value() ? "" : cubin.error().to_string());
  EXPECT_FALSE(cubin->data.empty());
  EXPECT_EQ(cubin_cache_stats().write_failures, 1U);
}

} // namespace
} // namespace tessiq::codegen
