FILE(GLOB TEST_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" test_*.cpp test_*.cu vtest_*.cpp vtest_*.cu)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(CUDA_NVCC_FLAGS "-arch=sm_10")
EXPAND_TESTS("${TEST_SOURCES}")
