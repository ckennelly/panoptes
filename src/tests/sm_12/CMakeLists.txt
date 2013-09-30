FILE(GLOB TEST_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" test_*.cpp test_*.cu vtest_*.cpp vtest_*.cu)

SET(CUDA_NVCC_FLAGS "-arch=sm_12")
EXPAND_TESTS("${TEST_SOURCES}")

# Atomic helper library
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
CUDA_ADD_LIBRARY(k_atomic_kernels "k_atomic_kernels.cu")
TARGET_LINK_LIBRARIES(vtest_k_atomic_global k_atomic_kernels)
TARGET_LINK_LIBRARIES(vtest_k_atomic_shared k_atomic_kernels)
