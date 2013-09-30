CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Boost
FIND_PACKAGE(Boost REQUIRED thread)

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD on)
INCLUDE_DIRECTORIES("${CUDA_INCLUDE_DIRS}")

# Warnings
SET(WARNINGS "-Wall -Wextra -Wshadow -Wpointer-arith -Wcast-align -Wwrite-strings -Wmissing-declarations -Wredundant-decls -Winline -Wno-long-long -Wuninitialized -Wconversion")
SET(CMAKE_C_FLAGS "${WARNINGS} -Wmissing-prototypes -Wnested-externs -Wstrict-prototypes")
SET(CMAKE_CXX_FLAGS "${WARNINGS} -Wabi -Wnon-virtual-dtor -fno-rtti")

# Build targets
ENABLE_TESTING()
ADD_SUBDIRECTORY(src)
