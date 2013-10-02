# Internal headers
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# Exported headers
INCLUDE_DIRECTORIES(include)

# Build PTX I/O library
ADD_SUBDIRECTORY(ptx_io)

# Build main Panoptes library
ADD_SUBDIRECTORY(panoptes)

# Build Panoptes launcher
ADD_SUBDIRECTORY(launcher)

# Tests
ADD_SUBDIRECTORY(tests)
