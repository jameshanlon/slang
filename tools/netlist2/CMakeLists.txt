# ~~~
# SPDX-FileCopyrightText: Michael Popoloski
# SPDX-License-Identifier: MIT
# ~~~

add_executable(slang_netlist2 netlist.cpp
                              source/Netlist.cpp)
add_executable(slang::netlist2 ALIAS slang_netlist2)

target_link_libraries(slang_netlist2 PRIVATE slang::slang)
set_target_properties(slang_netlist2 PROPERTIES OUTPUT_NAME "slang-netlist2")
target_include_directories(slang_netlist2 PRIVATE include)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  target_sources(slang_netlist2
                 PRIVATE ${PROJECT_SOURCE_DIR}/scripts/win32.manifest)
endif()

if(SLANG_INCLUDE_INSTALL)
  install(TARGETS slang_netlist2 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(SLANG_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
