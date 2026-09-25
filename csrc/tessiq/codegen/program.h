// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tessiq/codegen/nvrtc.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// Threads per block, for every kernel this backend generates. Not tuned per kernel.
///
/// Shared with the emitter, because a reduction sizes its shared memory for one block. A power of
/// two, because the reduction halves the block on each combine step.
constexpr unsigned int kBlockSize = 256;

/// One scalar argument to a kernel, as the C type of the parameter it fills.
///
/// Typed because the launch cannot see the kernel's signature: a float passed for a `double eps`
/// is read as the wrong bytes, with no error.
using KernelArgument = std::variant<std::int64_t, std::int32_t, float, double>;

/// How a launch is laid over the device: a grid of blocks, the threads in each block, and the
/// dynamic shared memory each block gets, in bytes.
struct LaunchShape {
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
  std::size_t shared_bytes = 0;
};

/// A kernel loaded onto a device and ready to launch.
///
/// Owned by the loader and shared; see `load_program`.
///
/// Thread-compatible: launching from several threads at once is fine, since a launch writes
/// nothing here, but the driver context has to be current on the calling thread.
class Program {
 public:
  virtual ~Program() = default;

  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  Program(Program&&) = delete;
  Program& operator=(Program&&) = delete;

  /// Launches the kernel over `shape` on `stream`.
  ///
  /// `buffers` are device pointers in the order the kernel takes them, and `arguments` follow them
  /// in the order the kernel's remaining parameters are declared. Each argument's type has to be
  /// the parameter's type exactly.
  ///
  /// Refused when a grid or block dimension is zero or the block has more than 1024 threads.
  /// Dynamic shared memory above 48 KB is opted into with the driver first.
  ///
  /// Asynchronous; the caller orders the result through `stream`.
  [[nodiscard]] virtual Result<void> launch(const std::vector<void*>& buffers,
                                            const std::vector<KernelArgument>& arguments,
                                            const LaunchShape& shape,
                                            Stream stream) const = 0;

  /// Launches the kernel with `blocks` blocks of `kBlockSize` threads on `stream`.
  ///
  /// The form every generated kernel takes: `buffers` are the inputs then the output, and
  /// `scalars` follow as 64-bit integers carrying the extents.
  [[nodiscard]] Result<void> launch(const std::vector<void*>& buffers,
                                    const std::vector<std::int64_t>& scalars,
                                    std::int64_t blocks,
                                    Stream stream) const;

  /// Launches one thread per element, with the element count as the only scalar.
  [[nodiscard]] Result<void>
  launch(const std::vector<void*>& buffers, std::int64_t elements, Stream stream) const {
    const auto blocks = (elements + static_cast<std::int64_t>(kBlockSize) - 1) /
                        static_cast<std::int64_t>(kBlockSize);
    return launch(buffers, {elements}, blocks, stream);
  }

 protected:
  Program() = default;
};

/// Loads `cubin` and finds `function` in it, reusing a module already loaded for the same bytes.
///
/// The returned pointer is owned by the loader and valid for the life of the process. Each module
/// is loaded once per device and kept.
Result<const Program*> load_program(const Cubin& cubin, std::string_view function, int ordinal);

/// Compiles `source` for CUDA device `ordinal` and loads `entry` from it.
///
/// Goes through the cubin cache, so the same source is compiled once per machine and loaded once
/// per process. Used for hand-written kernels, such as a custom operation's.
Result<const Program*>
compile_program(std::string_view source, std::string_view entry, int ordinal);

/// Whether this build can load and launch device code.
[[nodiscard]] bool launcher_available();

/// The compute capability of CUDA device `ordinal`, as major and minor.
///
/// Kernels are compiled for the device they run on, since a cubin does not load on a newer major
/// architecture: one built for sm_80 does not run on an sm_120 card.
Result<std::pair<int, int>> device_capability(int ordinal);

/// Loaded modules, for tests that want to know whether one was reused.
[[nodiscard]] std::size_t loaded_program_count();

} // namespace tessiq::codegen
