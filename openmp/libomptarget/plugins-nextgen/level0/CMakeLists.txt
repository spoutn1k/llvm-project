##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for an LEVEL0 machine if available.
#
##===----------------------------------------------------------------------===##

################################################################################
set(LIBOMPTARGET_BUILD_LEVEL0_PLUGIN TRUE CACHE BOOL "Whether to build LEVEL0 plugin")
if (NOT LIBOMPTARGET_BUILD_LEVEL0_PLUGIN)
  libomptarget_say("Not building LEVEL0 NextGen offloading plugin: LIBOMPTARGET_BUILD_LEVEL0_PLUGIN is false")
  return()
endif()

if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building LEVEL0 NextGen plugin: only support LEVEL0 in Linux x86_64, ppc64le, or aarch64 hosts")
  return()
endif()

################################################################################
# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=LEVEL0)

# Define debug prefix. TODO: This should be automatized in the Debug.h but it
# requires changing the original plugins.
add_definitions(-DDEBUG_PREFIX="TARGET LEVEL0 RTL")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc64le)|(aarch64)$")
   add_definitions(-DLITTLEENDIAN_CPU=1)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DDEBUG)
endif()

set(LIBOMPTARGET_EXTRA_SOURCE)
set(LIBOMPTARGET_DEP_LIBRARIES "ze_loader")

if(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  # On FreeBSD, the 'environ' symbol is undefined at link time, but resolved by
  # the dynamic linker at runtime. Therefore, allow the symbol to be undefined
  # when creating a shared library.
  set(LDFLAGS_UNDEFINED "-Wl,--allow-shlib-undefined")
else()
  set(LDFLAGS_UNDEFINED "-Wl,-z,defs")
endif()

add_llvm_library(omptarget.rtl.level0.nextgen SHARED
  src/rtl.cpp
  ${LIBOMPTARGET_EXTRA_SOURCE}

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}

  LINK_COMPONENTS
  Support
  Object

  LINK_LIBS
  PRIVATE
  elf_common
  MemoryManager
  PluginInterface
  ${LIBOMPTARGET_DEP_LIBRARIES}
  ${OPENMP_PTHREAD_LIB}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
  ${LDFLAGS_UNDEFINED}

  NO_INSTALL_RPATH
)

target_include_directories(
  omptarget.rtl.level0.nextgen
  PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.level0.nextgen LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
set_target_properties(omptarget.rtl.level0.nextgen PROPERTIES
  INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_VISIBILITY_PRESET protected)
