// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Reads textual IR, runs a named pass pipeline, prints the result.
//
// This is what makes the pass suite testable: a pass is a file in and a file out, with no PyTorch,
// no GPU and no fixture. It is also the fastest way to see what a pass does to a graph you wrote
// by hand.

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/text.h"
#include "tessiq/passes/pass.h"

namespace {

constexpr std::string_view kUsage = R"(usage: tessiq-opt [options] [input.tsr]

Reads textual IR from a file or stdin, runs a pass pipeline, and prints the result.

  --passes=a,b,c   passes to run, in order
  --list           list the registered passes and exit
  --verify-only    parse and verify, print nothing
  --stats          report per-pass timings and whether each changed anything
  -o FILE          write to FILE instead of stdout
  -h, --help       this message

  tessiq-opt --passes=const-fold,dce,cse graph.tsr
)";

struct Options {
  std::string input;
  std::string output;
  std::string pipeline;
  bool list = false;
  bool verify_only = false;
  bool stats = false;
  bool help = false;
};

tessiq::Result<Options> parse_args(const std::vector<std::string_view>& args) {
  Options options;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "-h" || arg == "--help") {
      options.help = true;
    } else if (arg == "--list") {
      options.list = true;
    } else if (arg == "--verify-only") {
      options.verify_only = true;
    } else if (arg == "--stats") {
      options.stats = true;
    } else if (arg.rfind("--passes=", 0) == 0) {
      options.pipeline = std::string(arg.substr(9));
    } else if (arg == "-o") {
      if (i + 1 >= args.size()) {
        return tessiq::invalid_argument("-o needs a file name");
      }
      options.output = std::string(args[++i]);
    } else if (!arg.empty() && arg.front() == '-') {
      return tessiq::invalid_argument("unknown option '" + std::string(arg) + "'; run with --help");
    } else if (options.input.empty()) {
      options.input = std::string(arg);
    } else {
      return tessiq::invalid_argument("more than one input file given");
    }
  }
  return options;
}

tessiq::Result<std::string> read_input(const std::string& path) {
  if (path.empty()) {
    std::ostringstream buffer;
    buffer << std::cin.rdbuf();
    return buffer.str();
  }
  std::ifstream file(path);
  if (!file) {
    return tessiq::not_found("cannot open '" + path + "'");
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

int run(const std::vector<std::string_view>& args) {
  auto options = parse_args(args);
  if (!options) {
    std::cerr << "tessiq-opt: " << options.error().message() << "\n";
    return 2;
  }

  if (options->help) {
    std::cout << kUsage;
    return 0;
  }

  if (options->list) {
    for (const std::string& name : tessiq::passes::PassRegistry::instance().names()) {
      auto pass = tessiq::passes::PassRegistry::instance().create(name);
      std::cout << "  " << name << std::string(name.size() < 14 ? 14 - name.size() : 1, ' ')
                << (*pass)->description() << "\n";
    }
    return 0;
  }

  auto text = read_input(options->input);
  if (!text) {
    std::cerr << "tessiq-opt: " << text.error().message() << "\n";
    return 1;
  }

  auto graph = tessiq::ir::parse(*text);
  if (!graph) {
    std::cerr << "tessiq-opt: " << graph.error().message() << "\n";
    return 1;
  }

  if (options->verify_only) {
    return 0;
  }

  tessiq::passes::PassManager manager;
  if (!options->pipeline.empty()) {
    if (const auto added = manager.add_from_pipeline(options->pipeline); !added) {
      std::cerr << "tessiq-opt: " << added.error().message() << "\n";
      return 2;
    }
  }

  auto reports = manager.run(**graph);
  if (!reports) {
    std::cerr << "tessiq-opt: " << reports.error().message() << "\n";
    return 1;
  }

  if (options->stats) {
    for (const auto& report : *reports) {
      std::cerr << "  " << report.pass << ": " << (report.changed ? "changed" : "no change") << ", "
                << report.elapsed_us << "us\n";
    }
  }

  const std::string printed = tessiq::ir::print(**graph);
  if (options->output.empty()) {
    std::cout << printed;
  } else {
    std::ofstream out(options->output);
    if (!out) {
      std::cerr << "tessiq-opt: cannot write '" << options->output << "'\n";
      return 1;
    }
    out << printed;
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  // main may not let an exception escape. Nothing here is expected to throw, but Result's
  // accessors can if a precondition is violated, and a terminate with no message helps nobody.
  try {
    std::vector<std::string_view> args;
    args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) {
      args.emplace_back(argv[i]);
    }
    return run(args);
  } catch (const std::exception& error) {
    std::cerr << "tessiq-opt: " << error.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "tessiq-opt: unknown error\n";
    return 1;
  }
}
