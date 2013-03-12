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

LDFLAGS := -Wl,-L/usr/lib/boost-1_49 -ldl -lboost_thread -lcuda

all: libpanoptes.so test

libpanoptes.so: $(SRCOBJS)
	gcc -shared -Wl,-soname,libcudart.so -o libpanoptes.so $^ $(LDFLAGS)

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
	$(NVCC) $(CXXFLAGS) -Xcompiler -fPIC -g $(INCLUDEDIR) -c $< -o $@

%.o: %.tab.cc Makefile $(wildcard *.h)
	$(CXX) $(CXXFLAGS) -fPIC -MMD -MP -fvisibility=hidden -c $< -o $@

%.o: %.lex.cc Makefile $(wildcard *.h) $(YACCOBJS)
	$(CXX) $(CXXFLAGS) -fPIC -MMD -MP -fvisibility=hidden -c $< -o $@

interposer.o: interposer.cpp Makefile $(wildcard *.h)
	$(CXX) $(CXXFLAGS) -fPIC -MMD -MP -I$(CUDAINC) -Werror=missing-declarations -fvisibility=default -c $< -o $@

ptx_ir.o: ptx_ir.cpp ptx_grammar.tab.cc $(wildcard *.h)

ptx_lex.o: ptx_lex.lex.cc $(wildcard *.h)

ptx_lexer.o: ptx_lexer.cpp Makefile ptx_grammar.tab.cc $(wildcard *.h)
	$(CXX) $(CXXFLAGS) -fPIC -MMD -MP -fvisibility=hidden -c $< -o $@

ptx_parser_state.o: ptx_parser_state.cpp Makefile ptx_grammar.tab.cc $(wildcard *.h)
	$(CXX) $(CXXFLAGS) -fPIC -MMD -MP -fvisibility=hidden -c $< -o $@

clean:
	-$(RM) -f $(SRCOBJS) $(patsubst %.o,%.d,$(SRCOBJS)) $(LEXOBJS) $(YACCOBJS) $(LEXHEADERS) $(YACCHEADERS) libpanoptes.so ptx_grammar.tab.hh ptx_lex.lex.hh *.gc* coverage.info
	-$(RM) -rf coverage
	$(MAKE) -C tests clean

# Based on systemtap's coverage gcov implementation:
# http://sourceware.org/git/gitweb.cgi?p=systemtap.git;a=blob;f=Makefile.am;h=c05e6c725b1fbe87f13e3656b851818a2274115a;hb=HEAD
lcov:
	find . -name '*.gc' | xargs rm -f
	$(MAKE) clean
	$(MAKE) CXXFLAGS="-g -fprofile-arcs -ftest-coverage" CFLAGS="-g -fprofile-arcs -ftest-coverage" LDFLAGS="$(LDFLAGS) -lgcov" libpanoptes.so
	$(MAKE) -C tests lcov
	lcov -c --no-recursion -d . -o coverage.info
	lcov -r coverage.info '/tmp/*' '/usr/*' '/opt/*' 'tests' -o coverage.info
	genhtml -o coverage -p $(abspath .) coverage.info

test: libpanoptes.so
	$(MAKE) -C tests

.PHONY: all clean
