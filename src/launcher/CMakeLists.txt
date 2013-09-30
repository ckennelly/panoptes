FILE(GLOB launcher_SOURCES *.c)

ADD_EXECUTABLE(panoptes_launcher ${launcher_SOURCES})
SET_TARGET_PROPERTIES(panoptes_launcher PROPERTIES OUTPUT_NAME panoptes)
ADD_DEPENDENCIES(panoptes_launcher panoptes)
