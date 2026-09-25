// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// What AddressSanitizer and LeakSanitizer do by default when they run these tests. Compiled into
// every build of the tests, and empty unless AddressSanitizer is on.
//
// On a machine with a GPU, every test that initialises the CUDA driver used to fail at exit, though
// the test itself passed. LeakSanitizer reported 50960 bytes in 5 allocations inside libcuda, the
// same in every such test whatever CUDA work it did, so it is state the driver allocates once and
// never frees. tessiq did not allocate it and cannot free it. CI never saw it, because its runners
// have no GPU and the driver is never initialised there.
//
// Suppressing libcuda is half of the fix. The default unwinder follows frame pointers, and libcuda
// is built without them, so three of the five allocations came back with an unknown frame where
// libcuda should be. A suppression matching libcuda then caught 280 of the 50960 bytes. The slow
// unwinder reads the unwind tables instead, which gives every frame its module. It roughly doubles
// how long the suite takes under the sanitizers.
//
// These are defaults: anything set in ASAN_OPTIONS or LSAN_OPTIONS when the tests run is applied on
// top of them, which is how CI's halt_on_error still takes effect.

#ifdef __SANITIZE_ADDRESS__
#define TESSIQ_ADDRESS_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define TESSIQ_ADDRESS_SANITIZER 1
#endif
#endif

#ifdef TESSIQ_ADDRESS_SANITIZER

// NOLINTBEGIN(bugprone-reserved-identifier,readability-identifier-naming): these are the names the
// sanitizer runtime looks for, and a declaration under any other name is never called.
extern "C" const char* __asan_default_options() { return "fast_unwind_on_malloc=0"; }

extern "C" const char* __lsan_default_suppressions() { return "leak:libcuda\n"; }

// NOLINTEND(bugprone-reserved-identifier,readability-identifier-naming)

#endif
