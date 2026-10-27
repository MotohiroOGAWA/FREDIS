# Locate Arrow, Parquet, HDF5, and nlohmann_json and pick their link targets.
#
# Works with system packages (shared libraries, e.g. Apache Arrow's APT
# repository + libhdf5-dev) and with vcpkg (static libraries on every OS).
# With MSENTITY_STATIC_DEPS=ON the static variants are required, so that
# libmsentity_c is self-contained and can be shipped inside the Python, Java,
# and C# packages.
#
# Results:
#   MSENTITY_ARROW_TARGET    Arrow::arrow_shared | Arrow::arrow_static
#   MSENTITY_PARQUET_TARGET  Parquet::parquet_shared | Parquet::parquet_static
#   MSENTITY_HDF5_TARGETS    targets for the HDF5 C and high-level libraries
#   MSENTITY_HDF5_FROM_CONFIG  ON when HDF5 came from its CMake package config

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)

if(MSENTITY_STATIC_DEPS OR NOT TARGET Arrow::arrow_shared)
  set(MSENTITY_ARROW_TARGET Arrow::arrow_static)
else()
  set(MSENTITY_ARROW_TARGET Arrow::arrow_shared)
endif()
if(MSENTITY_STATIC_DEPS OR NOT TARGET Parquet::parquet_shared)
  set(MSENTITY_PARQUET_TARGET Parquet::parquet_static)
else()
  set(MSENTITY_PARQUET_TARGET Parquet::parquet_shared)
endif()

# HDF5: prefer the package config (vcpkg, CMake-built HDF5), else FindHDF5
# (Debian/Ubuntu libhdf5-dev).
set(MSENTITY_HDF5_FROM_CONFIG OFF)
find_package(hdf5 CONFIG QUIET)
if(hdf5_FOUND)
  if(MSENTITY_STATIC_DEPS OR NOT TARGET hdf5::hdf5-shared)
    set(_kind static)
  else()
    set(_kind shared)
  endif()
  if(TARGET hdf5::hdf5-${_kind} AND TARGET hdf5::hdf5_hl-${_kind})
    set(MSENTITY_HDF5_TARGETS hdf5::hdf5_hl-${_kind} hdf5::hdf5-${_kind})
    set(MSENTITY_HDF5_FROM_CONFIG ON)
  endif()
endif()

if(NOT MSENTITY_HDF5_FROM_CONFIG)
  if(MSENTITY_STATIC_DEPS)
    set(HDF5_USE_STATIC_LIBRARIES ON)
  endif()
  find_package(HDF5 REQUIRED COMPONENTS C HL)
  # Wrap the variables in a target so it can be linked and exported like the
  # imported targets above.
  add_library(msentity_hdf5 INTERFACE)
  target_include_directories(msentity_hdf5 INTERFACE ${HDF5_INCLUDE_DIRS})
  target_compile_definitions(msentity_hdf5 INTERFACE ${HDF5_DEFINITIONS})
  target_link_libraries(msentity_hdf5 INTERFACE ${HDF5_HL_LIBRARIES} ${HDF5_C_LIBRARIES})
  set(MSENTITY_HDF5_TARGETS msentity_hdf5)
endif()

message(STATUS "msentity: Arrow=${MSENTITY_ARROW_TARGET} Parquet=${MSENTITY_PARQUET_TARGET} "
               "HDF5=${MSENTITY_HDF5_TARGETS} (static deps: ${MSENTITY_STATIC_DEPS})")
