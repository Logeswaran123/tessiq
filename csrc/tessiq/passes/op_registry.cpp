// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/passes/op_registry.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tessiq/ir/types.h"
#include "tessiq/passes/op_support.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/support/result.h"

namespace tessiq::passes {

namespace {

/// The most operands an expression can name, since `$` is followed by a single digit.
constexpr int kMostOperands = 10;

/// A registered operation, owning the strings its row points into.
///
/// Held by pointer so that the row's address never changes. `find_op` hands the row out and a pass
/// may keep it for as long as it runs.
struct Entry {
  OpDefinition definition;
  OpInfo info{};
};

std::mutex& registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, std::unique_ptr<Entry>, std::less<>>& registry() {
  static std::map<std::string, std::unique_ptr<Entry>, std::less<>> entries;
  return entries;
}

bool is_name_character(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '.';
}

bool is_identifier_character(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

Result<void> check_name(const OpDefinition& definition) {
  const std::string& name = definition.name;
  if (name.empty() || !std::all_of(name.begin(), name.end(), is_name_character)) {
    return invalid_argument("'" + name +
                            "' is not a name the textual IR can read; use letters, digits, "
                            "underscores and dots, as in demo.fast_gelu.default");
  }
  if (definition.canonical.empty()) {
    return invalid_argument(name + " needs its canonical name, such as demo::fast_gelu");
  }
  return ok();
}

/// Where the numeric literal starting at `start` ends, and whether it is a floating-point one.
std::pair<std::size_t, bool> scan_number(std::string_view text, std::size_t start) {
  std::size_t i = start;
  bool floating = false;
  const auto digits = [&] {
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])) != 0) {
      ++i;
    }
  };
  digits();
  if (i < text.size() && text[i] == '.') {
    floating = true;
    ++i;
    digits();
  }
  if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
    floating = true;
    ++i;
    if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
      ++i;
    }
    digits();
  }
  while (i < text.size() &&
         (text[i] == 'f' || text[i] == 'F' || text[i] == 'l' || text[i] == 'L')) {
    ++i;
  }
  return {i, floating};
}

/// Whether the literal between `start` and `end` is the whole argument of a `scalar_t(...)`.
bool is_wrapped(std::string_view text, std::size_t start, std::size_t end) {
  std::size_t before = start;
  while (before > 0 && text[before - 1] == ' ') {
    --before;
  }
  if (before > 0 && (text[before - 1] == '-' || text[before - 1] == '+')) {
    --before;
    while (before > 0 && text[before - 1] == ' ') {
      --before;
    }
  }
  constexpr std::string_view kOpen = "scalar_t(";
  if (before < kOpen.size() || text.substr(before - kOpen.size(), kOpen.size()) != kOpen) {
    return false;
  }
  std::size_t after = end;
  while (after < text.size() && text[after] == ' ') {
    ++after;
  }
  return after < text.size() && text[after] == ')';
}

/// Every `$n` names an operand the operation has, and every operand is named at least once.
Result<void> check_operands(const OpDefinition& definition) {
  const std::string_view text = definition.expression;
  std::vector<bool> read(static_cast<std::size_t>(definition.arity), false);
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '$') {
      continue;
    }
    const bool digit =
        i + 1 < text.size() && std::isdigit(static_cast<unsigned char>(text[i + 1])) != 0;
    const int operand = digit ? text[i + 1] - '0' : -1;
    if (operand < 0 || operand >= definition.arity) {
      return invalid_argument(definition.name + " takes " + std::to_string(definition.arity) +
                              " operands and its expression reads " +
                              std::string(text.substr(i, digit ? 2 : 1)));
    }
    read[static_cast<std::size_t>(operand)] = true;
    ++i;
  }
  for (std::size_t operand = 0; operand < read.size(); ++operand) {
    if (!read[operand]) {
      return invalid_argument(definition.name + " takes " + std::to_string(definition.arity) +
                              " operands and its expression never reads $" +
                              std::to_string(operand));
    }
  }
  return ok();
}

/// No floating-point constant is written bare, which C++ would read as a double.
Result<void> check_constants(const OpDefinition& definition) {
  const std::string_view text = definition.expression;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    const bool starts_number = std::isdigit(static_cast<unsigned char>(c)) != 0 ||
                               (c == '.' && i + 1 < text.size() &&
                                std::isdigit(static_cast<unsigned char>(text[i + 1])) != 0);
    // `$1` is an operand and the `1` in `log1p` is part of a name, neither of them a constant.
    const bool part_of_something =
        i > 0 && (is_identifier_character(text[i - 1]) || text[i - 1] == '$');
    if (!starts_number || part_of_something) {
      continue;
    }
    const auto [end, floating] = scan_number(text, i);
    if (floating && !is_wrapped(text, i, end)) {
      const std::string literal(text.substr(i, end - i));
      std::string message = definition.name;
      message += " writes the constant ";
      message += literal;
      message += " bare, which C++ reads as a double and which turns a float32 kernel into double "
                 "arithmetic; write scalar_t(";
      message += literal;
      message += ") instead";
      return invalid_argument(message);
    }
    i = end - 1;
  }
  return ok();
}

Result<void> check_expression(const OpDefinition& definition) {
  if (definition.expression.empty()) {
    return invalid_argument(definition.name + " is elementwise and needs an expression");
  }
  TESSIQ_RETURN_IF_ERROR(check_operands(definition));
  return check_constants(definition);
}

Result<std::uint32_t> check_dtypes(const OpDefinition& definition) {
  if (definition.dtypes.empty()) {
    return invalid_argument(definition.name + " names no element type to run on");
  }
  std::uint32_t mask = 0;
  for (const ir::DType dtype : definition.dtypes) {
    if (dtype != ir::DType::kF32 && dtype != ir::DType::kF64) {
      return invalid_argument(definition.name + " asks for " + std::string(ir::dtype_name(dtype)) +
                              ", and a registered expression runs on f32 and f64 only");
    }
    mask |= dtype_bit(dtype);
  }
  return mask;
}

/// A custom kernel brings its own code, so what is checked is that its call can be put together:
/// at least one tensor, and parameter names that say where a keyword argument goes.
Result<std::uint32_t> check_custom(const OpDefinition& definition) {
  const std::string& name = definition.name;
  if (!definition.expression.empty()) {
    return invalid_argument(name +
                            " is a custom kernel and brings its own code, not an expression");
  }
  if (definition.arity < 1) {
    return invalid_argument(name + " takes no tensor, so it has nothing to launch over");
  }
  const std::vector<std::string>& parameters = definition.parameters;
  if (!parameters.empty() && parameters.size() < static_cast<std::size_t>(definition.arity)) {
    return invalid_argument(name + " names fewer parameters than it has tensor operands");
  }
  for (auto it = parameters.begin(); it != parameters.end(); ++it) {
    if (it->empty() || std::find(parameters.begin(), it, *it) != it) {
      return invalid_argument(name + " needs a distinct name for every parameter");
    }
  }
  if (definition.dtypes.empty()) {
    return invalid_argument(name + " names no element type to run on");
  }
  std::uint32_t mask = 0;
  for (const ir::DType dtype : definition.dtypes) {
    mask |= dtype_bit(dtype);
  }
  return mask;
}

Result<std::uint32_t> check(const OpDefinition& definition) {
  TESSIQ_RETURN_IF_ERROR(check_name(definition));
  if (definition.category == OpCategory::kCustom) {
    return check_custom(definition);
  }
  if (definition.category != OpCategory::kElementwise) {
    return unsupported(definition.name +
                       ": an operation is registered as elementwise or as a custom kernel");
  }
  if (definition.arity < 1 || definition.arity > kMostOperands) {
    return invalid_argument(definition.name + " takes " + std::to_string(definition.arity) +
                            " operands; an expression can name between 1 and 10");
  }
  TESSIQ_RETURN_IF_ERROR(check_expression(definition));
  return check_dtypes(definition);
}

} // namespace

bool same_definition(const OpDefinition& a, const OpDefinition& b) {
  return a.name == b.name && a.canonical == b.canonical && a.category == b.category &&
         a.arity == b.arity && a.parameters == b.parameters && a.expression == b.expression &&
         a.dtypes == b.dtypes;
}

Result<const OpInfo*> register_op(const OpDefinition& definition) {
  TESSIQ_ASSIGN_OR_RETURN(const std::uint32_t dtypes, check(definition));
  if (find_builtin_op(definition.name) != nullptr) {
    return invalid_argument(definition.name + " is a built-in operation and cannot be replaced");
  }

  const std::scoped_lock guard(registry_mutex());
  auto& entries = registry();
  if (const auto found = entries.find(definition.name); found != entries.end()) {
    if (same_definition(found->second->definition, definition)) {
      return &found->second->info;
    }
    return invalid_argument(definition.name + " is already registered with a different definition");
  }
  auto entry = std::make_unique<Entry>();
  entry->definition = definition;
  const OpDefinition& owned = entry->definition;
  entry->info.name = owned.name;
  entry->info.canonical = owned.canonical;
  entry->info.category = owned.category;
  entry->info.arity = owned.arity;
  entry->info.expression = owned.expression;
  entry->info.has_kernel = true;
  entry->info.dtypes = dtypes;
  const OpInfo* info = &entry->info;
  entries.emplace(owned.name, std::move(entry));

  // Claimed from here on, exactly like a generated operation with a kernel. Whether a particular
  // node is claimed still depends on its types, which mark-supported checks against `dtypes`.
  default_op_support().add(definition.name);
  return info;
}

const OpInfo* find_registered_op(std::string_view name) {
  const std::scoped_lock guard(registry_mutex());
  const auto& entries = registry();
  const auto found = entries.find(name);
  return found == entries.end() ? nullptr : &found->second->info;
}

const OpDefinition* find_registered_definition(std::string_view name) {
  const std::scoped_lock guard(registry_mutex());
  const auto& entries = registry();
  const auto found = entries.find(name);
  return found == entries.end() ? nullptr : &found->second->definition;
}

std::vector<std::string> registered_op_names() {
  const std::scoped_lock guard(registry_mutex());
  std::vector<std::string> names;
  for (const auto& [name, entry] : registry()) {
    names.push_back(name);
  }
  return names;
}

} // namespace tessiq::passes
