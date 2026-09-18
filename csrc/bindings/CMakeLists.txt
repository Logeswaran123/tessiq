# Copyright 2026 The Tessera Authors.
# SPDX-License-Identifier: Apache-2.0

# FindPython defaults to the highest version it can locate, not the active environment. In a conda
# layout that means the base install rather than the env on PATH, so the extension gets built
# against one interpreter's headers and imported into another. LOCATION takes the first on PATH,
# and an explicit -DPython_EXECUTABLE still wins over both.
# STANDARD, not the default FIRST: FIRST trusts CONDA_PREFIX, which points at the base
# environment whenever a conda env is on PATH without having been activated in this shell. With
# STANDARD the search follows PATH, so an activated venv still wins by being first.
set(Python_FIND_VIRTUALENV STANDARD)
set(Python_FIND_STRATEGY LOCATION)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "tessera: building the extension for ${Python_EXECUTABLE} (${Python_VERSION})")
# torch and pybind11 ship their CMake config inside the Python packages, where CMake does not
# look by default. Ask the interpreter rather than making every caller pass -DCMAKE_PREFIX_PATH:
# a plain `cmake -S . -B build` should work in an environment that can already import them.
function(tessera_prepend_python_cmake_path module expression)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "${expression}"
    OUTPUT_VARIABLE path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE status
    ERROR_QUIET)
  if(status EQUAL 0 AND EXISTS "${path}")
    list(PREPEND CMAKE_PREFIX_PATH "${path}")
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
    message(STATUS "tessera: found ${module} cmake config at ${path}")
  else()
    message(STATUS "tessera: ${Python_EXECUTABLE} cannot import ${module}")
  endif()
endfunction()

tessera_prepend_python_cmake_path(pybind11 "import pybind11; print(pybind11.get_cmake_dir())")
tessera_prepend_python_cmake_path(torch "import torch; print(torch.utils.cmake_prefix_path)")

find_package(pybind11 CONFIG REQUIRED)

if(NOT Torch_FOUND)
  find_package(Torch QUIET)
endif()
if(NOT Torch_FOUND)
  message(FATAL_ERROR
          "The Python extension needs libtorch, and ${Python_EXECUTABLE} cannot import torch.\n"
          "Install it (pip install torch), or build the C++ core on its own with "
          "-DTESSERA_BUILD_PYTHON=OFF, which needs neither torch nor a GPU.")
endif()

pybind11_add_module(_C MODULE
  module.cpp
  ../tessera/torch/backend.cpp
  ../tessera/torch/ops.cpp)

target_link_libraries(_C PRIVATE
  tessera::runtime
  tessera::support
  ${TORCH_LIBRARIES})

# torch's headers have moved past C++17: recent releases use std::strong_ordering and requires
# clauses, which do not parse under -std=c++17 no matter what we write. Only the code that
# includes them is built this way; the core stays at C++17 so it keeps compiling anywhere.
target_compile_features(_C PRIVATE cxx_std_20)

# torch's Python bindings live in libtorch_python, which TORCH_LIBRARIES does not include.
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
target_link_libraries(_C PRIVATE "${TORCH_PYTHON_LIBRARY}")

# Must match the ABI torch itself was built with, or every std::string crossing the boundary is
# silently misinterpreted. Read from torch rather than assumed.
target_compile_definitions(_C PRIVATE
  _GLIBCXX_USE_CXX11_ABI=$<IF:$<BOOL:${TORCH_CXX_FLAGS}>,1,1>)

# The extension links libtorch, which lives in the torch package next to ours in site-packages.
# Without an RPATH, importing tessera before torch fails with a bare "libtorch.so: cannot open
# shared object file" that says nothing about the cause.
set_target_properties(_C PROPERTIES
  INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib;${TORCH_INSTALL_PREFIX}/lib"
  BUILD_WITH_INSTALL_RPATH ON)

tessera_set_warnings(_C)
tessera_enable_sanitizers(_C)

install(TARGETS _C DESTINATION tessera)
