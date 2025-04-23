# ~~~
# SPDX-FileCopyrightText: Michael Popoloski
# SPDX-License-Identifier: MIT
# ~~~

add_executable(
  netlist2_unittests
  ../../../tests/unittests/main.cpp
  ../../../tests/unittests/Test.cpp
  DepthFirstSearchTests.cpp
  DirectedGraphTests.cpp)

target_link_libraries(netlist2_unittests PRIVATE slang::slang Catch2::Catch2)

target_compile_definitions(netlist2_unittests PRIVATE UNITTESTS)

target_include_directories(netlist2_unittests PRIVATE ../include
                                                     ../../../tests/unittests)

add_test(NAME netlist2_unittests COMMAND netlist2_unittests)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  target_sources(netlist2_unittests
                 PRIVATE ${PROJECT_SOURCE_DIR}/scripts/win32.manifest)
endif()
