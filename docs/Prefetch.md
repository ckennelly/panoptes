Prefetch Instrumentation
========================

Instrumenting the `prefetch` and `prefetchu` instructions requires striking a
delicate balance between accuracy and practicality.  Much of Panoptes'
instrumentation for these instructions is based on the `vtest_k_prefetch` test,
used to probe the behavior of the device for in- and out-of-bounds prefetch
operations.

Observations
------------

1.  Prefetch operations in explicit spaces (`.local`, `.global`) do not cause
    faults when they are out-of-bounds.  Panoptes omits instrumentation from
    these instructions.

2.  There is some amount of slack permitted beyond the true allocation:
    Prefetch operations at the bounds of (and even a small distance beyond)
    memory location succeed.  For `.local` addresses, there seems to be 2K of
    slack.  For `.global` addresses, there can be up to 1M of slack.  Since
    this is likely due to an overallocation on the part of `cudaMalloc` and
    because this crosses several (16) chunks, Panoptes permits 1K of slack by
    rounding down to the nearest 1K boundary and checking the allocation.
    Since allocations from `cudaMalloc` are aligned, this adjusted address will
    generally be in-bounds.

3.  Misaligned `.global` accesses in the generic space with `prefetch` (but not
    `prefetchu`) fail.  Misaligned `.local` addresses in the generic space fail
    with either prefetch instruction.  Misaligned accesses in explicit spaces
    succeed.

Instrumentation Procedure
-------------------------

Panoptes validates prefetched addresses in four stages.

1.  Alignment is checked by examining the lower bits of the address obtained by
    masking with `sizeof(void *) - 1`.  If the instruction is a `prefetchu`
    instruction, Panoptes suppresses this error if `isspacep.global` is true
    for the address.

2.  If `isspacep.local`, the address is converted to a `.local` address and
    compared against `0xFFFCA0`.  Empirically, given a kernel with the PTX
    `.local .b8 t[N]`, the address satisifies `t + N == 0xFFFCA0`.  This bound
    is checked in the `Local.UpperBound` test of `vtest_k_prefetch`.

3.  If `isspacep.local` is false, the address is checked against the globl
    address space.  The address is rounded down to a 1K boundary to permit
    allocation slack.

4.  The validity bits of the address are checked.

If any stage fails, the global read-only address is prefetched instead of the
original addresss.  Prefetches against this address will always succeed.  This
prevents the kernel launch from failing, allowing the error processing code to
read the error data off of the GPU, display relevant errors, and then mark the
stream as having failed as the return value to `cudaStreamSycnchronize`, giving
the appearance of identical behavior as native CUDA execution.
