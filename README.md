Panoptes - A Binary Translation Framework for CUDA
(c) 2012-2013 - Chris Kennelly (chris@ckennelly.com)

Overview
========

Panoptes intercepts library calls to the GPU in order to maintain bookkeeping
information about the state of the GPU, including device code.  This permits
on-the-fly instrumentation of existing programs without recompilation.

This functionality is currently demonstrated by providing a memory checking
functionality similar to Valgrind's memcheck tool for CUDA that instruments
device code so that it can continue to run on the GPU, maintaining the
parallelism that may have necessitated the use of GPUs in the first place.

Panoptes is open source software, licensed under the GPLv3.  For more
information see COPYING.

Building
========

The Panoptes interposer depends on Boost, CUDA, cmake, and Valgrind (for its
hooks).  The testsuite shares the same dependencies as well as Google's
googletest framework (http://code.google.com/p/googletest/).

CMake should locate the appropriate packages to generate the Makefiles
necessary to build Panoptes.  A working CUDA-compatible GPU is required for the
tests to work.

Using Panoptes
==============

To run a CUDA program under Panoptes (for demonstration purposes, named
"`my_cuda_program`"):

    panoptes ./my_cuda_program

libpanoptes.so needs to be in the ordinary library search path (`LD_LIBRARY_PATH`).

Limitations
===========

Panoptes is a research code base that has not achieved a complete
implementation of CUDA.  Notable limitatations (and the rationale for them)
currently include:

* Mapped memory accesses from the GPU.  Panoptes currently reports to callers
  of cudaGetDeviceProperties the flag canMapHostMemory to be zero.

  Supporting direct access requires we maintain two sets of validity bits, one
  for the device and one for the host, keeping the state of the two sets
  reasonably consistent.

  We could make the host "authoritative," exposing the validity bits for mapped
  regions stored by Valgrind directly to the device.  Doing so would require
  tight coupling with Valgrind's internals as well as likely patching Valgrind
  to disable its compression technique for validity bits (as Panoptes uses 1:1
  bit level shadowing).

  We could make the device authoritative.  Upon a kernel launch, we would need
  to speculatively transfer any dirty, host-stored validity bits out of
  Valgrind and onto the device.  Upon a host access, we would have to load the
  validity bits off of the device and place them into Valgrind.

* Instruction support.  Not all parts of the PTX instruction set are supported.
  Further, parts of the PTX instruction set that are supported have largely
  been tested by generating kernels written in C/C++ with nvcc.  It is possible
  that there are untested edge cases that would only be exposed by use of
  inline PTX.

  * Surfaces:           Surface support is currently being tested, but is not
                        released.

  * Video instructions: The video instructions (vadd, vsub, vabsdiff, vmin,
                        vmax, vshl, vshr, vmad, vset) are currently not
                        instrumented.

Future Work
===========

With the `--tool` flag, Panoptes provides support for alternate instrumentation
modes.

* `hostgpu`: This instrumentation mode translates CUDA to parallel host code
  on-the-fly.  By using vectorized SSE/AVX instructions where possible,
  transformed code is able to maximize the computational throughput available
  from the host processor. This approach allows for a write-once, run-anywhere
  approach to parallel software development, exploiting the hardware resources
  available.

  These extensions add a series of passes during the processing of PTX code
  during the instrumentation process. Following basic block extraction, we
  identify opportunities to use native host vector instructions to provide 4-
  or 8-wide SIMD operations simultaneously. Where possible, code is scheduled
  to minimize register spills, executing the most instructions possible for a
  pseudowarp of 4 or 8 threads before performing a lightweight context switch
  to another pseudowarp. Combining host-based threads with vectorized
  instructions allows efficient use of host processors for parallel programs.

* These instrumentation capabilities can be extended to data race detection,
  coverage analysis, and instruction-level performance measurements.
