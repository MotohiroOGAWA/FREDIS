cmake_minimum_required(VERSION 3.21)

project(msentity
  VERSION 0.1.0
  DESCRIPTION "Mass-spectrometry dataset core library (MSDS/MSP/MGF/TSV/CSV)"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Tests and tools are built by default only when msentity is the top-level
# project (not when added with add_subdirectory / FetchContent).
option(MSENTITY_BUILD_TESTS "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(MSENTITY_BUILD_TOOLS "Build the msentity-tool command-line utility" ${PROJECT_IS_TOP_LEVEL})
option(MSENTITY_BUILD_C_API "Build the shared C API library (msentity_c) used by the bindings" ON)
option(MSENTITY_TEST_BINDINGS "Add the Python/Java/C# binding tests to ctest" ON)
option(MSENTITY_STATIC_DEPS
  "Link Arrow, Parquet, and HDF5 statically (self-contained msentity_c for packaging)" OFF)

if(MSVC)
  # Sources and tests contain UTF-8 literals.
  add_compile_options(/utf-8)
endif()
if(WIN32)
  # Keep DLLs next to the executables that load them (tests, tools).
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

include(GNUInstallDirs)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MsEntityDependencies.cmake)

add_library(msentity
  src/value.cpp
  src/internal/text.cpp
  src/internal/table.cpp
  src/internal/hdf5.cpp
  src/internal/json.cpp
  src/internal/parquet.cpp
  src/core/peak_series.cpp
  src/core/ms_dataset.cpp
  src/io/item_parser.cpp
  src/io/reader_context.cpp
  src/io/msp.cpp
  src/io/mgf.cpp
  src/io/delimited.cpp
  src/io/msds.cpp
  src/io/load.cpp
  src/processing/id.cpp
)
add_library(msentity::msentity ALIAS msentity)

target_include_directories(msentity
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Arrow is part of the public API (metadata tables are arrow::Table); HDF5,
# Parquet, and nlohmann_json are implementation details.
target_link_libraries(msentity
  PUBLIC
    ${MSENTITY_ARROW_TARGET}
  PRIVATE
    ${MSENTITY_PARQUET_TARGET}
    ${MSENTITY_HDF5_TARGETS}
    nlohmann_json::nlohmann_json)

set_target_properties(msentity PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(msentity PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Internal helpers (src/internal: Arrow tables, HDF5, Parquet, JSON, text) for
# extension libraries that add msentity to their own build with FetchContent /
# add_subdirectory, such as msentity-similarity. Not installed; not a stable API.
add_library(msentity_internal INTERFACE)
add_library(msentity::internal ALIAS msentity_internal)
target_include_directories(msentity_internal INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(msentity_internal INTERFACE
  msentity ${MSENTITY_PARQUET_TARGET} ${MSENTITY_HDF5_TARGETS} nlohmann_json::nlohmann_json)

# Shared C API library for the language bindings. It contains the whole core
# (linked statically) and exports only the msentity_* C functions.
if(MSENTITY_BUILD_C_API)
  add_library(msentity_c SHARED src/c/msentity_c.cpp)
  add_library(msentity::msentity_c ALIAS msentity_c)
  target_link_libraries(msentity_c PRIVATE msentity ${MSENTITY_ARROW_TARGET})
  target_include_directories(msentity_c
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(msentity_c PRIVATE MSENTITY_C_BUILD)
  set_target_properties(msentity_c PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  if(UNIX AND NOT APPLE)
    target_link_options(msentity_c PRIVATE "LINKER:--exclude-libs,ALL")
    if(MSENTITY_STATIC_DEPS AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # Do not require a newer libstdc++ than the target system has.
      target_link_options(msentity_c PRIVATE -static-libstdc++ -static-libgcc)
    endif()
  endif()
  install(TARGETS msentity_c EXPORT msentityTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(MSENTITY_BUILD_TOOLS)
  add_executable(msentity-tool tools/msentity_tool.cpp)
  target_link_libraries(msentity-tool PRIVATE msentity nlohmann_json::nlohmann_json)
endif()

if(MSENTITY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
  if(MSENTITY_BUILD_C_API AND MSENTITY_TEST_BINDINGS)
    add_subdirectory(bindings)
  endif()
endif()

if(TARGET msentity_hdf5)
  install(TARGETS msentity_hdf5 EXPORT msentityTargets)
endif()
install(TARGETS msentity EXPORT msentityTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/msentity DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(MSENTITY_BUILD_TOOLS)
  install(TARGETS msentity-tool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

include(CMakePackageConfigHelpers)
install(EXPORT msentityTargets
  NAMESPACE msentity::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/msentity)
configure_package_config_file(cmake/msentityConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/msentityConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/msentity)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/msentityConfigVersion.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/msentityConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/msentityConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/msentity)
