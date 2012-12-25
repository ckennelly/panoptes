CC=gcc
CXX=g++
LEX=flex
RM=rm
NVCC=nvcc
YACC=bison
WARNINGS := -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-align \
			-Wwrite-strings -Wmissing-declarations -Wredundant-decls \
			-Winline -Wno-long-long -Wuninitialized -Wconversion
CWARNINGS := $(WARNINGS) -Wmissing-prototypes -Wnested-externs -Wstrict-prototypes
CXXWARNINGS := $(WARNINGS) -Wabi -Wnon-virtual-dtor -fno-rtti
CFLAGS := -g -fPIC -std=c99 $(CWARNINGS)
CXXFLAGS := -g $(CXXWARNINGS)
CUDAINC=/opt/cuda/include

LEXOBJS := $(patsubst %.ll,%.lex.cc,$(wildcard *.ll))
LEXHEADERS := $(patsubst %.ll,%.lex.h,$(wildcard *.ll))
YACCOBJS := $(patsubst %.yy,%.tab.cc,$(wildcard *.yy))
YACCHEADERS := $(patsubst %.yy,%.tab.hh,$(wildcard %.yy))
SRCOBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(patsubst %.cu,%.o,$(wildcard *.cu)) \
	$(patsubst %.tab.cc,%.o,$(YACCOBJS)) $(patsubst %.lex.cc,%.o,$(LEXOBJS))

all: libpanoptes.so test

libpanoptes.so: $(SRCOBJS)
	gcc -shared -Wl,-soname,libcudart.so -o libpanoptes.so $^ -ldl -lboost_thread -lcuda

%.lex.cc: %.ll Makefile
	flex --header-file=$(patsubst %.lex.cc,%.lex.hh,$@) -o $@ -c $<

%.tab.cc: %.yy Makefile
	bison -d $<

# Blindly depend on all headers
%.o: %.c Makefile  $(wildcard *.h)
	$(CC) $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

%.o: %.cpp Makefile $(wildcard *.h)
	$(CXX) $(CXXFLAGS) -fPIC -MMD -MP -I$(CUDAINC) -fvisibility=hidden -c $< -o $@

%.o: %.cu Makefile
	$(NVCC) -Xcompiler -fPIC -g $(INCLUDEDIR) -c $< -o $@

%.o: %.tab.cc Makefile $(wildcard *.h)
	$(CXX) -fPIC -MMD -MP -fvisibility=hidden -c $< -o $@

%.o: %.lex.cc Makefile $(wildcard *.h) $(YACCOBJS)
	$(CXX) -fPIC -MMD -MP -fvisibility=hidden -c $< -o $@

interposer.o: interposer.cpp Makefile $(wildcard *.h)
	$(CXX) $(CXXFLAGS) -fPIC -MMD -MP -I$(CUDAINC) -fvisibility=default -c $< -o $@

ptx_ir.o: ptx_ir.cpp ptx_grammar.tab.cc $(wildcard *.h)

ptx_lex.o: ptx_lex.lex.cc $(wildcard *.h)

ptx_lexer.o: ptx_lexer.cpp Makefile ptx_grammar.tab.cc $(wildcard *.h)
	$(CXX) -fPIC -MMD -MP -fvisibility=hidden -c $< -o $@

ptx_parser_state.o: ptx_parser_state.cpp Makefile ptx_grammar.tab.cc $(wildcard *.h)
	$(CXX) -fPIC -MMD -MP -fvisibility=hidden -c $< -o $@

clean:
	-$(RM) -f $(SRCOBJS) $(patsubst %.o,%.d,$(SRCOBJS)) $(LEXOBJS) $(YACCOBJS) $(LEXHEADERS) $(YACCHEADERS) libpanoptes.so ptx_grammar.tab.hh ptx_lex.lex.hh *.gc*
	$(MAKE) -C tests clean

test: libpanoptes.so
	$(MAKE) -C tests

.PHONY: all clean
