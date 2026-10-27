# Tests of the Python, Java, and C# bindings, run by ctest when the toolchain
# for a language is found. Each suite uses the msentity_c library just built.

set(_lib "MSENTITY_C_LIBRARY=$<TARGET_FILE:msentity_c>")

# Python: ctypes + NumPy (pyarrow/pandas optional)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                  RESULT_VARIABLE _numpy_missing OUTPUT_QUIET ERROR_QUIET)
  if(NOT _numpy_missing)
    add_test(NAME binding_python
      COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
    set_tests_properties(binding_python PROPERTIES
      ENVIRONMENT "${_lib};PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python")
  else()
    message(STATUS "Python binding tests disabled (NumPy not found)")
  endif()
endif()

# Java: Maven + JDK 22 or later (Foreign Function & Memory API)
find_program(MSENTITY_MVN mvn)
if(MSENTITY_MVN)
  add_test(NAME binding_java
    COMMAND ${MSENTITY_MVN} -q -B test -Dmsentity.library=$<TARGET_FILE:msentity_c>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/java)
else()
  message(STATUS "Java binding tests disabled (mvn not found)")
endif()

# C#: .NET SDK 8 or later
find_program(MSENTITY_DOTNET dotnet HINTS $ENV{HOME}/.dotnet $ENV{DOTNET_ROOT})
if(MSENTITY_DOTNET)
  add_test(NAME binding_csharp
    COMMAND ${MSENTITY_DOTNET} test MsEntity.slnx --nologo -v quiet
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/csharp)
  set_tests_properties(binding_csharp PROPERTIES
    ENVIRONMENT "${_lib};DOTNET_CLI_TELEMETRY_OPTOUT=1;DOTNET_NOLOGO=1")
else()
  message(STATUS "C# binding tests disabled (dotnet not found)")
endif()
