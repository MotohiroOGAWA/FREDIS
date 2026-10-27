find_package(GTest REQUIRED)

add_executable(msentity_similarity_tests
  test_calculation.cpp
  test_similarity_dataset.cpp
  $<$<TARGET_EXISTS:msentity_similarity_c>:test_c_api.cpp>)
target_link_libraries(msentity_similarity_tests PRIVATE
  msentity_similarity msentity::internal GTest::gtest_main)
if(TARGET msentity_similarity_c)
  target_link_libraries(msentity_similarity_tests PRIVATE msentity_similarity_c)
endif()
target_compile_definitions(msentity_similarity_tests PRIVATE
  MSSIM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  MSENTITY_TEST_DATA_DIR="${msentity_SOURCE_DIR}/tests/data"
  MSENTITY_EXAMPLES_DIR="${msentity_SOURCE_DIR}/docs/examples")

include(GoogleTest)
gtest_discover_tests(msentity_similarity_tests)
