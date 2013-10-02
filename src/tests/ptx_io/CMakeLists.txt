SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

ADD_LIBRARY(test_utilities test_utilities.cpp)

# Parser tests
ADD_EXECUTABLE(test_parser test_parser.cpp)
TARGET_LINK_LIBRARIES(test_parser gtest ptx_io test_utilities)

ADD_TEST(NAME RUNNER_test_parser COMMAND "$<TARGET_FILE:test_parser>")
ADD_TEST(NAME VRUNNER_test_parser COMMAND valgrind --error-exitcode=1 --leak-check=full "$<TARGET_FILE:test_parser>")
