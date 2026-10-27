cmake_minimum_required(VERSION 3.21)

project(msentity_similarity
  VERSION 0.1.0
  DESCRIPTION "Spectrum similarity and the .mssim result format, built on msentity"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MSENTITY_SIMILARITY_BUILD_TESTS "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(MSENTITY_SIMILARITY_BUILD_TOOLS "Build the mssim-tool command-line utility" ${PROJECT_IS_TOP_LEVEL})
option(MSENTITY_SIMILARITY_BUILD_C_API
  "Build the shared C API library (msentity_similarity_c) used by the bindings" ON)
option(MSENTITY_SIMILARITY_TEST_BINDINGS "Add the Python/Java/C# binding tests to ctest" ON)
option(MSENTITY_STATIC_DEPS
  "Link Arrow, Parquet, and HDF5 statically (self-contained C libraries for packaging)" OFF)

if(MSVC)
  add_compile_options(/utf-8)
endif()
if(WIN32)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

include(GNUInstallDirs)

# ---- msentity --------------------------------------------------------------
# Pinned to a commit of https://github.com/MotohiroOGAWA/msentity. For local
# development use a checkout instead:
#   cmake -DFETCHCONTENT_SOURCE_DIR_MSENTITY=../msentity ...
include(FetchContent)
FetchContent_Declare(msentity
  GIT_REPOSITORY https://github.com/MotohiroOGAWA/msentity.git
  GIT_TAG 18c29fc8eada8ebb6e674cdb2c59a63fc0de78be)
set(MSENTITY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MSENTITY_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MSENTITY_TEST_BINDINGS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(msentity)

# ---- C++ library -------------------------------------------------------------
add_library(msentity_similarity
  src/calculation.cpp
  src/similarity_dataset.cpp
  src/calculate.cpp)
add_library(msentity::similarity ALIAS msentity_similarity)
target_include_directories(msentity_similarity
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# msentity::internal provides msentity's HDF5/Parquet/table helpers.
target_link_libraries(msentity_similarity
  PUBLIC msentity::msentity
  PRIVATE msentity::internal)
set_target_properties(msentity_similarity PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(msentity_similarity PRIVATE Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(msentity_similarity PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- C API for the bindings ----------------------------------------------------
if(MSENTITY_SIMILARITY_BUILD_C_API)
  add_library(msentity_similarity_c SHARED src/c/msentity_similarity_c.cpp)
  add_library(msentity::similarity_c ALIAS msentity_similarity_c)
  target_link_libraries(msentity_similarity_c PRIVATE msentity_similarity msentity::internal)
  target_include_directories(msentity_similarity_c
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${msentity_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_compile_definitions(msentity_similarity_c PRIVATE MSSIM_C_BUILD)
  set_target_properties(msentity_similarity_c PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
  if(UNIX AND NOT APPLE)
    target_link_options(msentity_similarity_c PRIVATE "LINKER:--exclude-libs,ALL")
    if(MSENTITY_STATIC_DEPS AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      target_link_options(msentity_similarity_c PRIVATE -static-libstdc++ -static-libgcc)
    endif()
  endif()
endif()

if(MSENTITY_SIMILARITY_BUILD_TOOLS)
  add_executable(mssim-tool tools/mssim_tool.cpp)
  target_link_libraries(mssim-tool PRIVATE msentity_similarity)
endif()

if(MSENTITY_SIMILARITY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
  if(MSENTITY_SIMILARITY_BUILD_C_API AND MSENTITY_SIMILARITY_TEST_BINDINGS)
    add_subdirectory(bindings)
  endif()
endif()

install(TARGETS msentity_similarity
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/msentity DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(TARGET msentity_similarity_c)
  install(TARGETS msentity_similarity_c
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
