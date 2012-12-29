CC=gcc
CXX=g++
NVCC=nvcc
RM=rm
WARNINGS := -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-align \
			-Wwrite-strings -Wmissing-declarations -Wredundant-decls \
			-Winline -Wno-long-long -Wuninitialized -Wconversion
CWARNINGS := $(WARNINGS) -Wmissing-prototypes -Wnested-externs -Wstrict-prototypes
CXXWARNINGS := $(WARNINGS)
CFLAGS := -g -fPIC -std=c99 $(CWARNINGS)
CXXFLAGS := -g $(CXXWARNINGS)
LDFLAGS := -lcuda -lgtest -L/usr/lib/boost-1_49
CUDAINC=/opt/cuda/include
VALGRINDFLAGS := --error-exitcode=1 --leak-check=full --suppressions=valgrind.suppressions

all: test exe

lcov:
	$(MAKE) LDFLAGS="$(LDFLAGS) -lgcov" exe test

# When building CUDA files (as all of these test cases are), nvcc does a better
# job at linking.
test_%: test_%.o
	nvcc $(LDFLAGS) -o $@ $<
vtest_%: vtest_%.o
	nvcc $(LDFLAGS) -o $@ $<

# This builds the test runner via dependency and runs it with and without
# Panoptes
run_%: %
	(test .$<.norm_passed -nt $<                                              ) || (                                        				 ./$< && touch .$<.norm_passed)
	(test .$<.pano_passed -nt $< && test .$<.pano_passed -nt ../libpanoptes.so) || (LD_PRELOAD="../libpanoptes.so"          				 ./$< && touch .$<.pano_passed)

# This builds the test runner (via the run_%) dependency and then runs it with
# Valgrind as well
vrun_%: % run_%
	(test .$<.valg_passed -nt $< && test .$<.valg_passed -nt ../libpanoptes.so) || (LD_PRELOAD="../libpanoptes.so" valgrind $(VALGRINDFLAGS) ./$< && touch .$<.valg_passed)

arun_%: %
	(test .$<.apassed -nt $<) || (./$< && touch .$<.apassed)

# Anything prefixed with test_ gets tested.  Anything prefixed with vtest_ is
# tested and run under Valgrind.  Tests which deliberately cause memory
# errors should not be run under Valgrind.
#
# atest: "Additional" standalone tests.
RUNNERS := $(patsubst %.cu,%,$(wildcard *test_*.cu)) $(patsubst %.cpp,%,$(wildcard *test_*.cpp))

exe : $(RUNNERS)

test: $(patsubst %.cu,run_%,$(wildcard test_*.cu)) $(patsubst %.cu,vrun_%,$(wildcard vtest_*.cu)) $(patsubst %.cpp,run_%,$(wildcard test_*.cpp)) $(patsubst %.cpp,vrun_%,$(wildcard vtest_*.cpp)) $(patsubst %,arun_%,$(wildcard atest_*))

%.o: %.c Makefile
	$(CC) $(CFLAGS) -fPIC -MMD -MP -c $< -o $@

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -fPIC -MMD -MP -I$(CUDAINC) -c $< -o $@

%.o: %.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@

test_k_assert.o: test_k_assert.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

k_atomic_kernels.o: k_atomic_kernels.cu vtest_k_atomic.h Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_12

k_atomic_kernels_generic.o: k_atomic_kernels_generic.cu vtest_k_atomic.h vtest_k_atomic_generic.h Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_atomic_generic.o: vtest_k_atomic.h vtest_k_atomic_generic.h
vtest_k_atomic_global.o: vtest_k_atomic.h
vtest_k_atomic_shared.o: vtest_k_atomic.h

vtest_k_atomic_generic: vtest_k_atomic_generic.o k_atomic_kernels_generic.o
	nvcc -o vtest_k_atomic_generic $^ $(LDFLAGS)

vtest_k_atomic_global: vtest_k_atomic_global.o k_atomic_kernels.o
	nvcc -o vtest_k_atomic_global $^ $(LDFLAGS)

vtest_k_atomic_shared: vtest_k_atomic_shared.o k_atomic_kernels.o
	nvcc -o vtest_k_atomic_shared $^ $(LDFLAGS)

vtest_k_atomic_shared_offsets.o: vtest_k_atomic_shared_offsets.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_13

vtest_k_ballot.o: vtest_k_ballot.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_bfe.o: vtest_k_bfe.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_bfi.o: vtest_k_bfi.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_bfind.o: vtest_k_bfind.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_blas.o: vtest_k_blas.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_block_reductions.o: vtest_k_block_reductions.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_brev.o: vtest_k_brev.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_brkpt.o: vtest_k_brkpt.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_11

vtest_k_carry.o: vtest_k_carry.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_clz.o: vtest_k_clz.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_copysign.o: vtest_k_copysign.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_fma.o: vtest_k_fma.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_global_reductions.o: vtest_k_global_reductions.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_irregular.o: vtest_k_irregular.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_isspacep.o: vtest_k_isspacep.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_mad.o: vtest_k_mad.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_mulhi.o: vtest_k_mulhi.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_prefetch.o: vtest_k_prefetch.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_pmevent.o: vtest_k_pmevent.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_printf.o: vtest_k_printf.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_prmt.o: vtest_k_prmt.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_popc.o: vtest_k_popc.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_rsqrt.o: vtest_k_rsqrt.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_testp.o: vtest_k_testp.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_threadfence.o: vtest_k_threadfence.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_tex1d_sm20.o: vtest_k_tex1d_sm20.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_uniform.o: vtest_k_uniform.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_k_vote.o: vtest_k_vote.cu Makefile
	$(NVCC) -g $(INCLUDEDIR) -c $< -o $@ -arch sm_20

vtest_threads: vtest_threads.o Makefile
	nvcc $(LDFLAGS) -lboost_thread -o $@ $<

clean:
	-$(RM) -f *.d *.o .*.*_passed $(RUNNERS) .*.apassed *.gc*

.PHONY: all clean
