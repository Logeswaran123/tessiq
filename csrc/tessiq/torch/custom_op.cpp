// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/torch/custom_op.h"

#include <c10/core/ScalarType.h>
#include <deque>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tessiq::custom_ops {

namespace {

/// Owns the definitions. A deque, because a definition is referred to by the static reference
/// TESSIQ_CUSTOM_OP binds, which must stay valid as later ones are added.
std::deque<CustomOp>& owned_definitions() {
  static std::deque<CustomOp> owned;
  return owned;
}

std::vector<CustomOp*>& definition_list() {
  static std::vector<CustomOp*> list;
  return list;
}

std::map<std::string, std::string_view, std::less<>>& kernel_files() {
  static std::map<std::string, std::string_view, std::less<>> files;
  return files;
}

} // namespace

CustomOp::CustomOp(std::string name, std::string signature, std::string_view file)
    : name_(std::move(name)), signature_(std::move(signature)),
      // The directory's own name, which is also how the build keys the kernel files it carries.
      directory_(std::filesystem::path(file).parent_path().filename().string()),
      kernel_file_(name_ + ".cu"), entry_(name_) {}

CustomOp& CustomOp::expression(std::string expression) {
  expression_ = std::move(expression);
  return *this;
}

CustomOp& CustomOp::dtypes(std::initializer_list<at::ScalarType> dtypes) {
  dtypes_.assign(dtypes);
  return *this;
}

CustomOp& CustomOp::kernel(std::string file, std::string entry) {
  kernel_file_ = std::move(file);
  entry_ = std::move(entry);
  return *this;
}

CustomOp& define(std::string name, std::string signature, std::string_view file) {
  CustomOp& op = owned_definitions().emplace_back(std::move(name), std::move(signature), file);
  definition_list().push_back(&op);
  return op;
}

const std::vector<CustomOp*>& definitions() { return definition_list(); }

KernelFile::KernelFile(std::string_view key, std::string_view text) {
  kernel_files().emplace(std::string(key), text);
}

std::optional<std::string_view> find_kernel_file(std::string_view key) {
  const auto& files = kernel_files();
  const auto found = files.find(key);
  if (found == files.end()) {
    return std::nullopt;
  }
  return found->second;
}

} // namespace tessiq::custom_ops
