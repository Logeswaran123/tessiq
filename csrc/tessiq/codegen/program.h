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

/// Threads per block, for every kernel this backend generates.
///
/// Shared with the emitter rather than chosen twice: a reduction declares shared memory for one
/// block's worth of values, and a block launched with a different size would read past it. It is a
/// power of two because the reduction halves the block on each step of its combine.
constexpr unsigned int kBlockSize = 256;

/// One scalar argument to a kernel, as the C type of the parameter it fills.
///
/// Typed rather than a number the launcher converts, because the kernel's own signature decides
/// how many bytes a parameter is and what they mean, and nothing at the launch can read that
/// signature. A float handed to a `double eps` is read as the wrong bytes without any error.
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
/// Loading is not free, so a program outlives the call that needed it and is shared. Instances are
/// owned by the loader and handed out by pointer.
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
  /// Refused when a dimension of the grid or the block is zero, or the block has more than 1024
  /// threads, which no device runs. Shared memory beyond the 48 KB every kernel gets is asked for
  /// from the driver first, which is how a kernel opts into the rest of what the card has.
  ///
  /// The launch is asynchronous. Nothing here synchronises, so the caller orders the result
  /// against whatever reads it, which is what choosing the stream is for.
  [[nodiscard]] virtual Result<void> launch(const std::vector<void*>& buffers,
                                            const std::vector<KernelArgument>& arguments,
                                            const LaunchShape& shape,
                                            Stream stream) const = 0;

  /// Launches the kernel with `blocks` blocks of `kBlockSize` threads on `stream`.
  ///
  /// The shape every kernel the backend generates takes. `buffers` are every input and then the
  /// output, and `scalars` follow them as 64-bit integers, which is how the kernel learns its
  /// extents. An elementwise kernel takes one, its element count; a reduction takes three.
  [[nodiscard]] Result<void> launch(const std::vector<void*>& buffers,
                                    const std::vector<std::int64_t>& scalars,
                                    std::int64_t blocks,
                                    Stream stream) const;

  /// Launches one thread per element, which is what every elementwise kernel wants.
  ///
  /// The grid is derived from the count rather than passed, because an elementwise kernel's grid
  /// is never anything else and making the caller compute it is one more thing to get wrong.
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
/// The returned pointer is owned by the loader and stays valid for the life of the process. A
/// module is loaded once per device and kept, because loading walks the whole image and a model
/// launches the same kernel thousands of times.
Result<const Program*> load_program(const Cubin& cubin, std::string_view function, int ordinal);

/// Compiles `source` for CUDA device `ordinal` and loads `entry` from it.
///
/// The compile goes through the cubin cache, so the same source costs a compile once per machine
/// and a load once per process. For a kernel written by hand rather than generated, which is what
/// a custom operation brings.
Result<const Program*>
compile_program(std::string_view source, std::string_view entry, int ordinal);

/// Whether this build can load and launch device code.
[[nodiscard]] bool launcher_available();

/// The compute capability of CUDA device `ordinal`, as major and minor.
///
/// Whoever compiles a kernel has to ask, because a cubin only loads on the architecture it names
/// and there is no forward compatibility to fall back on: code built for sm_80 does not run on an
/// sm_120 card. Guessing a low architecture and hoping is the obvious mistake and it fails at
/// load time with an invalid image.
Result<std::pair<int, int>> device_capability(int ordinal);

/// Loaded modules, for tests that want to know whether one was reused.
[[nodiscard]] std::size_t loaded_program_count();

} // namespace tessiq::codegen
