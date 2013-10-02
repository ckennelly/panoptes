FIND_PACKAGE(BISON REQUIRED)
FIND_PACKAGE(FLEX REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

SET(GRAMMAR_FILE "ptx_grammar.tab.cc")
BISON_TARGET(grammar "ptx_grammar.yy" "${GRAMMAR_FILE}")

SET(LEXER_FILE "ptx_lex.lex.cc")
FLEX_TARGET(lexer "ptx_lex.ll" "${LEXER_FILE}")

# Kill off spurious warnings that are due to the generated code.
SET_SOURCE_FILES_PROPERTIES(${GRAMMAR_FILE} ${LEXER_FILE} PROPERTIES COMPILE_FLAGS "-Wno-conversion")

FILE(GLOB ptx_io_SOURCES *.cpp)

ADD_LIBRARY(ptx_io STATIC "${GRAMMAR_FILE}" "${LEXER_FILE}" ${ptx_io_SOURCES})
ADD_DEPENDENCIES(ptx_io lexer grammar)
