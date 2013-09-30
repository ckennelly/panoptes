# Exported headers
INCLUDE_DIRECTORIES(include)

# Build main Panoptes library
ADD_SUBDIRECTORY(panoptes)

# Build Panoptes launcher
ADD_SUBDIRECTORY(launcher)

# Tests
ADD_SUBDIRECTORY(tests)
