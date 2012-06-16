Thread Safety
=============

Panoptes aims to replicate CUDA's native levels of thread safety.  For most
calls, this requires holding a context-level lock to protect internal data
structures.

Kernel Launches
===============

The sequence of calls required launch a kernel with the CUDA Runtime API
presents an opportunity for races to occur if the call stack is na?vely
implemented.  Consider the example:

    my_kernel<<<grid, block, 0>>>(my_argument);

NVCC translates this into several calls into the Runtime API:

    cudaConfigureCall(grid, block, 0);
    cudaSetupArgument(&my_argument, sizeof(my_argument), 0);
    cudaLaunch(my_kernel);

If the internal state of the call stack is shared across threads that are using
the same context, we will inevitably race as one thread starts to uses (and
pops) the call stack that another thread configured.

Call-level locking does not resolve this problem:  Races, albeit at a coarser
level can still occur.  Empirically, given that `vtest_threads` does not fail
when running natively, it appears that CUDA keeps the call stack in thread
local storage to protect each thread's call stack from other threads sharing
the same context.
