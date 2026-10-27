# Tests of the Python, Java, and C# bindings, run by ctest when the toolchain
# for a language is found. Each suite uses the msentity_similarity_c library
# just built, and the msentity binding and msentity_c library from the
# msentity sources this build uses (${msentity_SOURCE_DIR}).

if(NOT TARGET msentity_c)
  message(STATUS "Binding tests disabled (msentity_c not built; set MSENTITY_BUILD_C_API=ON)")
  return()
endif()

set(_libs "MSSIM_C_LIBRARY=$<TARGET_FILE:msentity_similarity_c>"
          "MSENTITY_C_LIBRARY=$<TARGET_FILE:msentity_c>"
          "MSENTITY_TEST_DATA_DIR=${msentity_SOURCE_DIR}/tests/data")
set(_msentity_bindings ${msentity_SOURCE_DIR}/bindings)

# Python: ctypes + NumPy (pyarrow optional)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                  RESULT_VARIABLE _numpy_missing OUTPUT_QUIET ERROR_QUIET)
  if(NOT _numpy_missing)
    add_test(NAME binding_python
      COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
    set(_pythonpath "${CMAKE_CURRENT_SOURCE_DIR}/python")
    if(WIN32)
      string(APPEND _pythonpath "\\;${_msentity_bindings}/python")
    else()
      string(APPEND _pythonpath ":${_msentity_bindings}/python")
    endif()
    set_tests_properties(binding_python PROPERTIES
      ENVIRONMENT "${_libs};PYTHONPATH=${_pythonpath}")
  else()
    message(STATUS "Python binding tests disabled (NumPy not found)")
  endif()
endif()

# Java: Maven + JDK 22 or later. The msentity artifact is installed into the
# local Maven repository first (fixture binding_java_msentity).
find_program(MSSIM_MVN mvn)
if(MSSIM_MVN)
  add_test(NAME binding_java_msentity
    COMMAND ${MSSIM_MVN} -q -B install -DskipTests
    WORKING_DIRECTORY ${_msentity_bindings}/java)
  set_tests_properties(binding_java_msentity PROPERTIES FIXTURES_SETUP msentity_java)
  add_test(NAME binding_java
    COMMAND ${MSSIM_MVN} -q -B test
      -Dmssim.library=$<TARGET_FILE:msentity_similarity_c>
      -Dmsentity.library=$<TARGET_FILE:msentity_c>
      -Dmsentity.data=${msentity_SOURCE_DIR}/tests/data
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/java)
  set_tests_properties(binding_java PROPERTIES FIXTURES_REQUIRED msentity_java)
else()
  message(STATUS "Java binding tests disabled (mvn not found)")
endif()

# C#: .NET SDK 8 or later; MsEntity is referenced as a project of the msentity sources.
find_program(MSSIM_DOTNET dotnet HINTS $ENV{HOME}/.dotnet $ENV{DOTNET_ROOT})
if(MSSIM_DOTNET)
  add_test(NAME binding_csharp
    COMMAND ${MSSIM_DOTNET} test MsEntity.Similarity.slnx --nologo -v quiet
      -p:MsEntityProject=${_msentity_bindings}/csharp/MsEntity/MsEntity.csproj
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/csharp)
  set_tests_properties(binding_csharp PROPERTIES
    ENVIRONMENT "${_libs};DOTNET_CLI_TELEMETRY_OPTOUT=1;DOTNET_NOLOGO=1")
else()
  message(STATUS "C# binding tests disabled (dotnet not found)")
endif()
