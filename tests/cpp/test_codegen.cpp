// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Pulling a fusion group out of a graph and turning it into source. Needs no GPU or CUDA toolkit.

#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/norm.h"
#include "tessiq/ir/text.h"
#include "tessiq/passes/pass.h"

namespace tessiq::codegen {
namespace {

/// Parses, then runs the pipeline that produces fusion groups.
std::unique_ptr<ir::Graph> fused(const std::string& text) {
  auto graph = ir::parse(text);
  EXPECT_TRUE(graph.has_value()) << (graph.has_value() ? "" : graph.error().to_string());
  if (!graph.has_value()) {
    return nullptr;
  }
  passes::PassManager manager;
  EXPECT_TRUE(manager.add_from_pipeline("mark-supported,cluster,fuse-elementwise").has_value());
  EXPECT_TRUE(manager.run(**graph).has_value());
  return std::move(*graph);
}

constexpr std::string_view kChain =
    R"(graph @g(%a: f32[1024], %b: f32[1024], %c: f32[1024]) -> f32[1024] {
  %m = aten.mul.Tensor(%a, %b) : f32[1024]
  %s = aten.add.Tensor(%m, %c) : f32[1024]
  %r = aten.relu.default(%s) : f32[1024]
  %g = aten.sigmoid.default(%r) : f32[1024]
  return %g
}
)";

TEST(ExtractTest, InputsAreTheValuesTheGroupDoesNotCompute) {
  auto graph = fused(std::string(kChain));
  ASSERT_NE(graph, nullptr);

  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());

  // %m, %s and %r are computed inside and never leave, so they are not parameters.
  EXPECT_EQ(kernel->input_names(), (std::vector<std::string>{"a", "b", "c"}));
  EXPECT_EQ(kernel->output, "g");
  EXPECT_EQ(kernel->body.size(), 4U);
  EXPECT_EQ(kernel->elements, 1024);
  EXPECT_EQ(kernel->dtype, ir::DType::kF32);
}

TEST(ExtractTest, AnInputLayoutAssignCopiedIsAskedForUnderItsSourcesName) {
  // The executor makes layout-assign's tessiq.contiguous copy itself when it packs the kernel's
  // inputs, so the kernel names the original value and reads it packed.
  auto graph = ir::parse(R"(graph @g(%a: f32[4,8:1,4]) -> f32[4,8] {
  %r = aten.relu.default(%a) : f32[4,8]
  return %r
}
)");
  ASSERT_TRUE(graph.has_value()) << graph.error().to_string();
  passes::PassManager manager;
  ASSERT_TRUE(manager.add_from_pipeline("mark-supported,layout-assign,cluster,fuse-elementwise")
                  .has_value());
  ASSERT_TRUE(manager.run(**graph).has_value());
  ASSERT_NE(ir::print(**graph).find("%a.contiguous = tessiq.contiguous(%a)"), std::string::npos);

  const auto kernel = extract_kernel(**graph, 0);
  ASSERT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());
  EXPECT_EQ(kernel->input_names(), (std::vector<std::string>{"a"}));
  EXPECT_TRUE(kernel->inputs.front().type.is_contiguous());
  EXPECT_EQ(kernel->inputs.front().index, "i");

  // The copy's name has a dot in it, which a C identifier cannot.
  const auto source = cuda_emitter().emit(*kernel);
  ASSERT_TRUE(source.has_value()) << source.error().to_string();
  EXPECT_EQ(source->find("contiguous"), std::string::npos) << *source;
  EXPECT_NE(source->find("p_a"), std::string::npos) << *source;
}

TEST(ExtractTest, AValueReadOutsideTheGroupIsTheOutput) {
  auto graph = fused(R"(graph @g(%a: f32[8]) -> (f32[8], f32[8]) {
  %p = aten.relu.default(%a) : f32[8]
  %q = aten.tanh.default(%p) : f32[8]
  %x = aten.abs.default(%q) : f32[8]
  %y = aten.neg.default(%q) : f32[8]
  return %x, %y
}
)");
  ASSERT_NE(graph, nullptr);

  // %p fuses into %q because it has one reader. %q has two, so it cannot fuse onwards and is
  // where the group ends.
  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());
  EXPECT_EQ(kernel->output, "q");
  EXPECT_EQ(kernel->input_names(), (std::vector<std::string>{"a"}));
}

TEST(ExtractTest, AConstantArgumentBecomesALiteral) {
  auto graph = fused(R"(graph @g(%a: f32[8]) -> f32[8] {
  %d = aten.mul.Tensor(%a) {arg1 = 2.0} : f32[8]
  %r = aten.relu.default(%d) : f32[8]
  return %r
}
)");
  ASSERT_NE(graph, nullptr);

  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());
  ASSERT_EQ(kernel->body.front().args.size(), 2U);
  EXPECT_TRUE(kernel->body.front().args[0].is_value);
  EXPECT_FALSE(kernel->body.front().args[1].is_value);
  EXPECT_EQ(kernel->body.front().args[1].text, "2.0");
}

TEST(ExtractTest, AConstantBeforeATensorKeepsItsPosition) {
  // `1 - x` imports as sub with the constant at position 0, which backward graphs produce often.
  // Appending constants after operands would emit `x - 1`: same shape and type, wrong number.
  auto graph = fused(R"(graph @g(%a: f32[8]) -> f32[8] {
  %s = aten.sub.Tensor(%a) {arg0 = 1.0} : f32[8]
  %r = aten.relu.default(%s) : f32[8]
  return %r
}
)");
  ASSERT_NE(graph, nullptr);

  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());
  ASSERT_EQ(kernel->body.front().args.size(), 2U);
  EXPECT_FALSE(kernel->body.front().args[0].is_value) << "the constant moved off position 0";
  EXPECT_EQ(kernel->body.front().args[0].text, "1.0");
  EXPECT_TRUE(kernel->body.front().args[1].is_value);

  const auto source = cuda_emitter().emit(*kernel);
  ASSERT_TRUE(source.has_value());
  EXPECT_NE(source->find("scalar_t v_s = (scalar_t(1.0) - v_a);"), std::string::npos) << *source;
}

TEST(ExtractTest, AGroupThatIsNotThereIsReported) {
  auto graph = fused(std::string(kChain));
  ASSERT_NE(graph, nullptr);
  EXPECT_FALSE(extract_kernel(*graph, 7).has_value());
}

TEST(ExtractTest, GroupsAreListedInOrder) {
  auto graph = fused(R"(graph @g(%a: f32[4,4], %b: f32[4,4]) -> f32[4,4] {
  %x = aten.add.Tensor(%a, %b) : f32[4,4]
  %r = aten.relu.default(%x) : f32[4,4]
  %m = aten.mm.default(%r, %b) : f32[4,4]
  %t = aten.tanh.default(%m) : f32[4,4]
  %s = aten.sigmoid.default(%t) : f32[4,4]
  return %s
}
)");
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(fusion_groups(*graph), (std::vector<std::int64_t>{0, 1}));
}

TEST(CudaEmitterTest, EmitsTheWholeChainAsOneKernel) {
  auto graph = fused(std::string(kChain));
  ASSERT_NE(graph, nullptr);
  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value());

  const Emitter& emitter = cuda_emitter();
  ASSERT_TRUE(emitter.supports(*kernel));
  const auto source = emitter.emit(*kernel);
  ASSERT_TRUE(source.has_value());

  EXPECT_EQ(*source, R"(// Generated by tessiq. Do not edit.
typedef float scalar_t;

extern "C" __global__ void tessiq_fused_0(
    const scalar_t* __restrict__ p_a,
    const scalar_t* __restrict__ p_b,
    const scalar_t* __restrict__ p_c,
    scalar_t* __restrict__ p_g,
    long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;

  scalar_t v_a = p_a[i];
  scalar_t v_b = p_b[i];
  scalar_t v_c = p_c[i];

  scalar_t v_m = (v_a * v_b);
  scalar_t v_s = (v_m + v_c);
  scalar_t v_r = (v_s < scalar_t(0) ? scalar_t(0) : v_s);
  scalar_t v_g = (scalar_t(1) / (scalar_t(1) + exp(-v_r)));

  p_g[i] = v_g;
}
)");
}

TEST(CudaEmitterTest, TheElementTypeIsATypedefSoOneSpellingServesBoth) {
  auto graph = fused(R"(graph @g(%a: f64[8], %b: f64[8]) -> f64[8] {
  %m = aten.mul.Tensor(%a, %b) : f64[8]
  %r = aten.relu.default(%m) : f64[8]
  return %r
}
)");
  ASSERT_NE(graph, nullptr);
  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value());

  const auto source = cuda_emitter().emit(*kernel);
  ASSERT_TRUE(source.has_value());
  EXPECT_NE(source->find("typedef double scalar_t;"), std::string::npos) << *source;
  // The body is identical to the float version; only the scalar_t typedef differs.
  EXPECT_NE(source->find("scalar_t v_r = (v_m < scalar_t(0) ? scalar_t(0) : v_m);"),
            std::string::npos)
      << *source;
}

TEST(CudaEmitterTest, RefusesATypeItHasNoSpellingFor) {
  auto graph = fused(R"(graph @g(%a: f16[8], %b: f16[8]) -> f16[8] {
  %m = aten.mul.Tensor(%a, %b) : f16[8]
  %r = aten.relu.default(%m) : f16[8]
  return %r
}
)");
  ASSERT_NE(graph, nullptr);
  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value());

  // Half precision needs a header and its own arithmetic, so the emitter declines and the
  // operations run one at a time.
  EXPECT_FALSE(cuda_emitter().supports(*kernel));
}

TEST(CudaEmitterTest, TheBoundsCheckIsThere) {
  auto graph = fused(std::string(kChain));
  ASSERT_NE(graph, nullptr);
  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value());
  const auto source = cuda_emitter().emit(*kernel);
  ASSERT_TRUE(source.has_value());

  // The bounds check lets the grid overshoot the element count.
  EXPECT_NE(source->find("if (i >= n) return;"), std::string::npos) << *source;
}

// --- reading an operand that broadcasts ------------------------------------------------------

namespace {

/// A contiguous f32 type of these extents, which is what almost every operand is.
ir::TensorType dims(std::vector<std::int64_t> extents) {
  return {ir::DType::kF32, ir::Shape(std::move(extents))};
}

/// An f32 type of these extents laid out with these steps, which is what a permute produces.
ir::TensorType strided(std::vector<std::int64_t> extents, ir::Strides steps) {
  return {ir::DType::kF32, ir::Shape(std::move(extents)), std::move(steps)};
}

} // namespace

TEST(IndexForTest, AnOperandTheSameShapeIsReadAtTheLoopIndex) {
  EXPECT_EQ(index_for(dims({4, 8}), dims({4, 8})), "i");
}

TEST(IndexForTest, APerRowValueIsOneDivision) {
  // What a softmax and a layer norm subtract. Every element of a row reads the same position.
  EXPECT_EQ(index_for(dims({4, 8}), dims({4, 1})), "i / 8");
  EXPECT_EQ(index_for(dims({2, 6, 8}), dims({2, 6, 1})), "i / 8");
}

TEST(IndexForTest, ARowAddedToEveryRowIsOneRemainder) {
  // What a bias looks like. The position walks the row and starts again.
  EXPECT_EQ(index_for(dims({4, 8}), dims({8})), "i % 8");
  EXPECT_EQ(index_for(dims({2, 6, 8}), dims({6, 8})), "i % 48");
}

TEST(IndexForTest, AMiddleAxisNeedsTheGeneralForm) {
  // Neither a run nor a cycle, so the position is split into coordinates and mapped onto the
  // operand's extents.
  EXPECT_EQ(index_for(dims({2, 6, 8}), dims({6, 1})), "((i / 8) % 6)");
}

TEST(IndexForTest, AScalarReadsOnePosition) {
  EXPECT_EQ(index_for(dims({4, 8}), dims({1, 1})), "i / 32");
}

TEST(IndexForTest, APermutedOperandIsSteppedThroughItsOwnLayout) {
  // A transposed operand: the same bytes in a different order, so the loop steps through them.
  EXPECT_EQ(index_for(dims({4, 8}), strided({4, 8}, {1, 4})), "((i / 8) % 4) + (i % 8) * 4");
}

TEST(IndexForTest, AContiguousOperandOfTheSameShapeIsStillJustTheIndex) {
  // Packed strides spelled out are still read at `i`.
  EXPECT_EQ(index_for(dims({2, 6, 8}), strided({2, 6, 8}, {48, 8, 1})), "i");
}

TEST(IndexForTest, AnOperandThatDoesNotBroadcastHasNoExpression) {
  // Fusion asks this, so a group the emitter would decline never forms.
  EXPECT_TRUE(index_for(dims({4, 8}), dims({4, 4})).empty());
  EXPECT_TRUE(index_for(dims({4, 8}), dims({2, 4, 8})).empty()) << "more axes than the result";
}

// --- extents that are not known until the call ------------------------------------------------

TEST(IndexForTest, AnUnknownExtentIsStillTheLoopIndexWhenTheShapesMatch) {
  // Identical dim lists read contiguously need no extent: one index walks every buffer.
  EXPECT_EQ(index_for(dims({ir::kDynamic, 32}), dims({ir::kDynamic, 32})), "i");
  EXPECT_EQ(index_for(dims({ir::kDynamic, ir::kDynamic}), dims({ir::kDynamic, ir::kDynamic})), "i");
}

TEST(IndexForTest, AnUnknownExtentStillFoldsADivisorThatIsKnown) {
  // A layer norm: the row length is fixed by the model, and the index needs only that.
  EXPECT_EQ(index_for(dims({ir::kDynamic, 32}), dims({ir::kDynamic, 1})), "i / 32");
  EXPECT_EQ(index_for(dims({ir::kDynamic, 32}), dims({32})), "i % 32") << "a bias row";
}

TEST(IndexForTest, ADivisorThatIsNotKnownHasNoExpression) {
  // A softmax over a sequence: the row length is unknown, so `index_for` has no answer. The read
  // is still a run, as `operand_read` says, so the divisor can be passed to the kernel.
  const ir::TensorType out = dims({ir::kDynamic, 4, ir::kDynamic, ir::kDynamic});
  const ir::TensorType row = dims({ir::kDynamic, 4, ir::kDynamic, 1});
  EXPECT_TRUE(index_for(out, row).empty());

  const OperandRead found = operand_read(out, row).value_or(OperandRead{});
  EXPECT_EQ(found.kind, OperandRead::Kind::kRepeated);
  EXPECT_FALSE(found.is_known()) << "a softmax row is as long as the sequence";
}

TEST(IndexForTest, AnOperandThatCannotBroadcastIsStillRefused) {
  EXPECT_FALSE(operand_read(dims({4, 8}), dims({ir::kDynamic, 1})).has_value());
  EXPECT_FALSE(operand_read(dims({4, 8}), dims({4, 4})).has_value());
}

TEST(IndexForTest, APermutedOperandNeedsEveryExtent) {
  // The strided form needs every extent, so without them there is no kernel.
  EXPECT_TRUE(operand_read(dims({4, 8}), strided({4, 8}, {1, 4})).has_value());
  EXPECT_FALSE(
      operand_read(dims({ir::kDynamic, 8}), strided({ir::kDynamic, 8}, {1, 4})).has_value());
}

TEST(ExtractTest, AGroupWithUnknownExtentsCarriesNoElementCount) {
  auto graph = fused(R"(graph @g(%a: f32[?,32], %b: f32[?,32]) -> f32[?,32] {
  %m = aten.mul.Tensor(%a, %b) : f32[?,32]
  %r = aten.relu.default(%m) : f32[?,32]
  return %r
}
)");
  ASSERT_NE(graph, nullptr);

  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());
  EXPECT_TRUE(kernel->has_dynamic_extent());
  EXPECT_EQ(kernel->elements, ir::kDynamic);
  EXPECT_EQ(kernel->body.size(), 2U);
}

TEST(CudaEmitterTest, AKernelForUnknownExtentsHoldsNoExtent) {
  auto graph = fused(R"(graph @g(%a: f32[?,32], %b: f32[?,32]) -> f32[?,32] {
  %m = aten.mul.Tensor(%a, %b) : f32[?,32]
  %r = aten.relu.default(%m) : f32[?,32]
  return %r
}
)");
  ASSERT_NE(graph, nullptr);
  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value());
  ASSERT_TRUE(cuda_emitter().supports(*kernel));

  const auto source = cuda_emitter().emit(*kernel);
  ASSERT_TRUE(source.has_value()) << (source.has_value() ? "" : source.error().to_string());

  // The count reaches the kernel as `n`, so one compile serves every shape.
  EXPECT_NE(source->find("long long n) {"), std::string::npos) << *source;
  EXPECT_NE(source->find("[i]"), std::string::npos) << *source;
  EXPECT_EQ(source->find("32"), std::string::npos) << "an extent was written into the body";
}

TEST(CudaEmitterTest, ADivisorThatIsNotKnownBecomesAParameter) {
  // A softmax row of unknown length. The kernel takes the divisor beside the element count.
  auto graph = fused(R"(graph @g(%s: f32[?,4,?,?], %m: f32[?,4,?,1]) -> f32[?,4,?,?] {
  %d = aten.sub.Tensor(%s, %m) : f32[?,4,?,?]
  %e = aten.exp.default(%d) : f32[?,4,?,?]
  return %e
}
)");
  ASSERT_NE(graph, nullptr);
  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value()) << (kernel.has_value() ? "" : kernel.error().to_string());
  ASSERT_TRUE(cuda_emitter().supports(*kernel));

  const auto source = cuda_emitter().emit(*kernel);
  ASSERT_TRUE(source.has_value()) << (source.has_value() ? "" : source.error().to_string());
  EXPECT_NE(source->find("long long n,\n    long long d_m) {"), std::string::npos) << *source;
  EXPECT_NE(source->find("p_m[i / d_m]"), std::string::npos) << *source;
  EXPECT_NE(source->find("p_s[i]"), std::string::npos) << *source;
}

TEST(CudaEmitterTest, ADivisorThatIsKnownIsStillFoldedIn) {
  // Division by a constant is cheaper than by a register, so a known divisor is written in and
  // only an unknown one becomes a parameter.
  auto graph = fused(R"(graph @g(%x: f32[?,32], %m: f32[?,1]) -> f32[?,32] {
  %d = aten.sub.Tensor(%x, %m) : f32[?,32]
  %e = aten.exp.default(%d) : f32[?,32]
  return %e
}
)");
  ASSERT_NE(graph, nullptr);
  const auto kernel = extract_kernel(*graph, 0);
  ASSERT_TRUE(kernel.has_value());
  const auto source = cuda_emitter().emit(*kernel);
  ASSERT_TRUE(source.has_value());
  EXPECT_NE(source->find("p_m[i / 32]"), std::string::npos) << *source;
  EXPECT_EQ(source->find("d_m"), std::string::npos) << *source;
}

TEST(CudaEmitterTest, AScalarIsAParameterAfterTheDivisors) {
  FusedKernel kernel;
  kernel.name = "scaled";
  kernel.elements = ir::kDynamic;
  KernelInput bias;
  bias.name = "b";
  bias.index.clear();
  bias.read.kind = OperandRead::Kind::kCycled;
  bias.read.divisor = ir::kDynamic;
  kernel.inputs = {bias};
  kernel.scalars = {"s"};
  kernel.output = "y";
  kernel.body = {Step{"mul", {{"b", true}, {"s", false}}, "y"}};

  ASSERT_TRUE(cuda_emitter().supports(kernel));
  const auto source = cuda_emitter().emit(kernel);
  ASSERT_TRUE(source.has_value()) << (source.has_value() ? "" : source.error().to_string());
  EXPECT_NE(source->find("long long n,\n    long long d_b,\n    scalar_t s) {"), std::string::npos)
      << *source;
  EXPECT_NE(source->find("scalar_t v_y = (v_b * scalar_t(s));"), std::string::npos) << *source;
}

TEST(NormSourceTest, LayerNormWritesEachRowsStatistics) {
  const std::string source = layer_norm_source(ir::DType::kF64, "ln");
  EXPECT_NE(source.find("typedef double scalar_t;"), std::string::npos) << source;
  EXPECT_NE(source.find("extern \"C\" __global__ void ln("), std::string::npos) << source;
  EXPECT_NE(source.find("p_mean[row] = mean;"), std::string::npos) << source;
  EXPECT_NE(source.find("p_rstd[row] = rstd;"), std::string::npos) << source;
  // The variance is taken over centred values, as torch takes it.
  EXPECT_NE(source.find("centred * centred"), std::string::npos) << source;
}

TEST(NormSourceTest, BatchNormHasAnEntryPerLayout) {
  const std::string source = batch_norm_source(ir::DType::kF32, "bn");
  EXPECT_NE(source.find("typedef float scalar_t;"), std::string::npos) << source;
  EXPECT_NE(source.find("void bn_by_plane("), std::string::npos) << source;
  EXPECT_NE(source.find("void bn_by_element("), std::string::npos) << source;
  EXPECT_NE(source.find("p_in[i] * alpha + beta"), std::string::npos) << source;
}

} // namespace
} // namespace tessiq::codegen
