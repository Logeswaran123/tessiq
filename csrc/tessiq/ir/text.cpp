// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/ir/text.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "tessiq/support/macros.h"

namespace tessiq::ir {

namespace {

std::string_view trim(std::string_view text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
    text.remove_prefix(1);
  }
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    text.remove_suffix(1);
  }
  return text;
}

/// Splits on `separator`, ignoring separators inside brackets so that `f32[4,8]` stays whole, and
/// inside a quoted string, so that an attribute `"a,b"` does too.
std::vector<std::string_view> split_top_level(std::string_view text, char separator) {
  std::vector<std::string_view> parts;
  int depth = 0;
  bool quoted = false;
  std::size_t start = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '"') {
      quoted = !quoted;
    } else if (quoted) {
      continue;
    } else if (c == '[' || c == '(' || c == '{') {
      depth++;
    } else if (c == ']' || c == ')' || c == '}') {
      depth--;
    } else if (c == separator && depth == 0) {
      parts.push_back(trim(text.substr(start, i - start)));
      start = i + 1;
    }
  }
  const std::string_view tail = trim(text.substr(start));
  if (!tail.empty() || !parts.empty()) {
    parts.push_back(tail);
  }
  return parts;
}

Error at_line(std::size_t line, const std::string& message) {
  return invalid_argument("line " + std::to_string(line) + ": " + message);
}

/// Parses one attribute value. The spelling decides the type, which is why the printer takes care
/// to keep a whole-number double looking like a float.
Result<Attribute> parse_attribute_value(std::string_view text) {
  if (text == "true") {
    return Attribute{true};
  }
  if (text == "false") {
    return Attribute{false};
  }
  if (text == "none") {
    return Attribute{std::monostate{}};
  }
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
    return Attribute{std::string(text.substr(1, text.size() - 2))};
  }
  if (text.size() >= 2 && text.front() == '[' && text.back() == ']') {
    std::vector<std::int64_t> values;
    for (const std::string_view token : split_top_level(text.substr(1, text.size() - 2), ',')) {
      if (token.empty()) {
        continue;
      }
      auto value = parse_int(token);
      if (!value) {
        return invalid_argument("bad integer '" + std::string(token) + "' in attribute list");
      }
      values.push_back(*value);
    }
    return Attribute{std::move(values)};
  }

  // The spelling decides the type: 1 is an integer attribute, 1.0 a float one. That is why the
  // printer keeps a whole-number double looking like a float.
  const bool looks_float = text.find_first_of(".eE") != std::string_view::npos;
  if (!looks_float) {
    if (auto value = parse_int(text)) {
      return Attribute{*value};
    }
  }
  if (auto value = parse_double(text)) {
    return Attribute{*value};
  }
  return invalid_argument("cannot parse attribute value '" + std::string(text) + "'");
}

/// `%name: f32[4,8]` -> the pair.
Result<std::pair<std::string, TensorType>> parse_typed_name(std::string_view text) {
  const std::size_t colon = text.find(':');
  if (colon == std::string_view::npos) {
    return invalid_argument("expected '%name: type', got '" + std::string(text) + "'");
  }
  std::string_view name = trim(text.substr(0, colon));
  if (name.empty() || name.front() != '%') {
    return invalid_argument("expected a value name starting with %, got '" + std::string(name) +
                            "'");
  }
  name.remove_prefix(1);
  TESSIQ_ASSIGN_OR_RETURN(const TensorType type, parse_tensor_type(trim(text.substr(colon + 1))));
  return std::pair{std::string(name), type};
}

} // namespace

namespace {

/// Renders one node: results, op, operands, attributes, result types.
void print_node(std::ostringstream& out, const Node* node) {
  out << "  ";
  const std::vector<Value*> results = node->results();
  for (std::size_t i = 0; i < results.size(); ++i) {
    out << (i != 0 ? ", " : "") << "%" << results[i]->name();
  }
  if (!results.empty()) {
    out << " = ";
  }

  out << node->op() << "(";
  for (std::size_t i = 0; i < node->operand_count(); ++i) {
    out << (i != 0 ? ", " : "") << "%" << node->operand(i)->name();
  }
  out << ")";

  if (!node->attributes().empty()) {
    out << " {";
    bool first = true;
    for (const auto& [name, value] : node->attributes()) {
      out << (first ? "" : ", ") << name << " = " << attribute_to_string(value);
      first = false;
    }
    out << "}";
  }

  if (!results.empty()) {
    out << " : ";
    for (std::size_t i = 0; i < results.size(); ++i) {
      out << (i != 0 ? ", " : "") << results[i]->type().to_string();
    }
  }
  out << "\n";
}

/// Renders `graph @name(params) -> results {`.
void print_signature(std::ostringstream& out, const Graph& graph) {
  out << "graph @" << graph.name() << "(";
  const std::vector<Value*> params = graph.params();
  for (std::size_t i = 0; i < params.size(); ++i) {
    out << (i != 0 ? ", " : "") << "%" << params[i]->name() << ": "
        << params[i]->type().to_string();
  }
  out << ") -> ";

  const std::vector<Value*>& results = graph.results();
  // One result needs no parentheses; zero or several do, so the form round-trips either way.
  const bool parenthesise = results.size() != 1;
  out << (parenthesise ? "(" : "");
  for (std::size_t i = 0; i < results.size(); ++i) {
    out << (i != 0 ? ", " : "") << results[i]->type().to_string();
  }
  out << (parenthesise ? ")" : "") << " {\n";
}

} // namespace

std::string print(const Graph& graph) {
  std::ostringstream out;
  print_signature(out, graph);
  for (const Node* node : graph.nodes()) {
    print_node(out, node);
  }

  out << "  return";
  const std::vector<Value*>& results = graph.results();
  for (std::size_t i = 0; i < results.size(); ++i) {
    out << (i != 0 ? "," : "") << " %" << results[i]->name();
  }
  out << "\n}\n";
  return out.str();
}

namespace {

/// A source line with its number, comments and surrounding space removed. Blank lines are dropped.
struct SourceLine {
  std::size_t number;
  std::string_view text;
};

std::vector<SourceLine> tokenise(std::string_view text) {
  std::vector<SourceLine> lines;
  std::size_t number = 0;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t newline = text.find('\n', start);
    const std::size_t end = newline == std::string_view::npos ? text.size() : newline;
    number++;
    std::string_view line = trim(text.substr(start, end - start));
    const std::size_t comment = line.find("//");
    if (comment != std::string_view::npos) {
      line = trim(line.substr(0, comment));
    }
    if (!line.empty()) {
      lines.push_back({number, line});
    }
    if (newline == std::string_view::npos) {
      break;
    }
    start = newline + 1;
  }
  return lines;
}

/// The index of the paren closing the one at `open`, or npos.
std::size_t matching_paren(std::string_view text, std::size_t open) {
  int depth = 0;
  for (std::size_t i = open; i < text.size(); ++i) {
    if (text[i] == '(') {
      depth++;
    } else if (text[i] == ')' && --depth == 0) {
      return i;
    }
  }
  return std::string_view::npos;
}

/// `graph @name(%a: t, ...) -> ...` — creates the graph and its parameters.
Result<std::unique_ptr<Graph>> parse_header(const SourceLine& line,
                                            std::unordered_map<std::string, Value*>& scope) {
  if (line.text.substr(0, 6) != "graph ") {
    return at_line(line.number, "expected the file to start with 'graph @name(...)'");
  }

  const std::size_t open = line.text.find('(');
  if (open == std::string_view::npos) {
    return at_line(line.number, "malformed graph header: no parameter list");
  }

  // The paren matching `open`, not the last on the line: a multi-result signature ends with
  // `-> (f32[4], i64[])`, whose closing paren is not the parameter list's.
  const std::size_t close = matching_paren(line.text, open);
  if (close == std::string_view::npos) {
    return at_line(line.number, "malformed graph header: unbalanced parentheses");
  }

  std::string_view name = trim(line.text.substr(6, open - 6));
  if (name.empty() || name.front() != '@') {
    return at_line(line.number, "expected a graph name starting with @");
  }
  name.remove_prefix(1);

  auto graph = std::make_unique<Graph>(std::string(name));
  const std::string_view params = trim(line.text.substr(open + 1, close - open - 1));
  if (!params.empty()) {
    for (const std::string_view part : split_top_level(params, ',')) {
      auto param = parse_typed_name(part);
      if (!param) {
        return at_line(line.number, param.error().message());
      }
      if (scope.count(param->first) != 0) {
        return at_line(line.number, "duplicate parameter %" + param->first);
      }
      scope[param->first] = graph->add_param(param->first, param->second);
    }
  }
  return graph;
}

/// `return %a, %b`
Result<void> parse_return(const SourceLine& line,
                          const std::unordered_map<std::string, Value*>& scope,
                          Graph& graph) {
  std::vector<Value*> results;
  const std::string_view rest = trim(line.text.substr(6));
  if (!rest.empty()) {
    for (std::string_view token : split_top_level(rest, ',')) {
      if (token.empty() || token.front() != '%') {
        return at_line(line.number,
                       "expected a value name after return, got '" + std::string(token) + "'");
      }
      token.remove_prefix(1);
      const auto it = scope.find(std::string(token));
      if (it == scope.end()) {
        return at_line(line.number, "return references undefined value %" + std::string(token));
      }
      results.push_back(it->second);
    }
  }
  graph.set_results(std::move(results));
  return ok();
}

/// The result names on the left of `=`, if there is one. Consumes them from `remaining`.
Result<std::vector<std::string>> parse_result_names(const SourceLine& line,
                                                    std::string_view& remaining) {
  std::vector<std::string> names;
  const std::size_t equals = remaining.find('=');
  const std::size_t paren = remaining.find('(');
  if (equals == std::string_view::npos || (paren != std::string_view::npos && paren < equals)) {
    return names;
  }

  for (std::string_view token : split_top_level(remaining.substr(0, equals), ',')) {
    if (token.empty() || token.front() != '%') {
      return at_line(line.number,
                     "expected a result name starting with %, got '" + std::string(token) + "'");
    }
    token.remove_prefix(1);
    names.emplace_back(token);
  }
  remaining = trim(remaining.substr(equals + 1));
  return names;
}

/// `{a = 1, b = "x"}` if present. Consumes it from `tail`.
Result<void> parse_attributes(const SourceLine& line, std::string_view& tail, Node& node) {
  if (tail.empty() || tail.front() != '{') {
    return ok();
  }
  const std::size_t brace = tail.find('}');
  if (brace == std::string_view::npos) {
    return at_line(line.number, "unterminated attribute list");
  }
  for (const std::string_view entry : split_top_level(tail.substr(1, brace - 1), ',')) {
    if (entry.empty()) {
      continue;
    }
    const std::size_t eq = entry.find('=');
    if (eq == std::string_view::npos) {
      return at_line(line.number,
                     "expected 'name = value' in attributes, got '" + std::string(entry) + "'");
    }
    auto value = parse_attribute_value(trim(entry.substr(eq + 1)));
    if (!value) {
      return at_line(line.number, value.error().message());
    }
    node.set_attribute(std::string(trim(entry.substr(0, eq))), std::move(value).value());
  }
  tail = trim(tail.substr(brace + 1));
  return ok();
}

/// `: f32[4], i64[]` if present.
Result<std::vector<TensorType>> parse_result_types(const SourceLine& line, std::string_view tail) {
  std::vector<TensorType> types;
  if (tail.empty()) {
    return types;
  }
  if (tail.front() != ':') {
    return at_line(line.number,
                   "expected ':' before the result type, got '" + std::string(tail) + "'");
  }
  for (const std::string_view token : split_top_level(trim(tail.substr(1)), ',')) {
    auto type = parse_tensor_type(token);
    if (!type) {
      return at_line(line.number, type.error().message());
    }
    types.push_back(std::move(type).value());
  }
  return types;
}

/// `%a, %b` — resolves each against the scope and attaches it to the node.
Result<void> parse_operands(const SourceLine& line,
                            std::string_view operands,
                            const std::unordered_map<std::string, Value*>& scope,
                            Node& node) {
  if (operands.empty()) {
    return ok();
  }
  for (std::string_view token : split_top_level(operands, ',')) {
    if (token.empty() || token.front() != '%') {
      return at_line(line.number,
                     "expected an operand starting with %, got '" + std::string(token) + "'");
    }
    token.remove_prefix(1);
    const auto it = scope.find(std::string(token));
    if (it == scope.end()) {
      return at_line(line.number, "use of undefined value %" + std::string(token));
    }
    node.append_operand(it->second);
  }
  return ok();
}

/// `%r = op(%a, %b) {attr = v} : type`
Result<void>
parse_node(const SourceLine& line, std::unordered_map<std::string, Value*>& scope, Graph& graph) {
  std::string_view remaining = line.text;
  TESSIQ_ASSIGN_OR_RETURN(const std::vector<std::string> result_names,
                          parse_result_names(line, remaining));

  const std::size_t op_open = remaining.find('(');
  if (op_open == std::string_view::npos) {
    return at_line(line.number, "expected an operation call, got '" + std::string(remaining) + "'");
  }
  const std::string_view op = trim(remaining.substr(0, op_open));
  if (op.empty()) {
    return at_line(line.number, "missing operation name");
  }

  const std::size_t op_close = matching_paren(remaining, op_open);
  if (op_close == std::string_view::npos) {
    return at_line(line.number, "unbalanced parentheses in operand list");
  }

  Node* node = graph.add_node(std::string(op));
  TESSIQ_RETURN_IF_ERROR(parse_operands(
      line, trim(remaining.substr(op_open + 1, op_close - op_open - 1)), scope, *node));

  std::string_view tail = trim(remaining.substr(op_close + 1));
  TESSIQ_RETURN_IF_ERROR(parse_attributes(line, tail, *node));
  TESSIQ_ASSIGN_OR_RETURN(const std::vector<TensorType> types, parse_result_types(line, tail));

  if (types.size() != result_names.size()) {
    return at_line(line.number,
                   "'" + std::string(op) + "' names " + std::to_string(result_names.size()) +
                       " result(s) but gives " + std::to_string(types.size()) + " type(s)");
  }

  for (std::size_t i = 0; i < result_names.size(); ++i) {
    if (scope.count(result_names[i]) != 0) {
      return at_line(line.number, "%" + result_names[i] + " is defined twice");
    }
    scope[result_names[i]] = node->append_result(result_names[i], types[i]);
  }
  return ok();
}

} // namespace

Result<std::unique_ptr<Graph>> parse(std::string_view text) {
  const std::vector<SourceLine> lines = tokenise(text);
  if (lines.empty()) {
    return invalid_argument("empty input: expected a graph");
  }

  std::unordered_map<std::string, Value*> scope;
  TESSIQ_ASSIGN_OR_RETURN(auto graph, parse_header(lines.front(), scope));

  for (std::size_t i = 1; i < lines.size(); ++i) {
    const SourceLine& line = lines[i];
    if (line.text == "}") {
      continue;
    }
    if (line.text.substr(0, 6) == "return") {
      TESSIQ_RETURN_IF_ERROR(parse_return(line, scope, *graph));
      continue;
    }
    TESSIQ_RETURN_IF_ERROR(parse_node(line, scope, *graph));
  }

  TESSIQ_RETURN_IF_ERROR(verify(*graph));
  return graph;
}

} // namespace tessiq::ir
