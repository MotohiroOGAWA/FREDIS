find_package(GTest REQUIRED)

add_executable(msentity_tests
  test_text.cpp
  test_peak_series.cpp
  test_ms_dataset.cpp
  test_item_parser.cpp
  test_msp_mgf.cpp
  test_delimited.cpp
  test_msds.cpp
  $<$<TARGET_EXISTS:msentity_c>:test_c_api.cpp>
)
target_link_libraries(msentity_tests PRIVATE msentity GTest::gtest_main)
# Tests also exercise internal helpers.
target_include_directories(msentity_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(msentity_tests PRIVATE ${MSENTITY_HDF5_TARGETS} nlohmann_json::nlohmann_json)
target_compile_definitions(msentity_tests PRIVATE
  MSENTITY_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  MSENTITY_EXAMPLES_DIR="${PROJECT_SOURCE_DIR}/docs/examples")
if(TARGET msentity_c)
  target_link_libraries(msentity_tests PRIVATE msentity_c)
endif()

include(GoogleTest)
gtest_discover_tests(msentity_tests)
