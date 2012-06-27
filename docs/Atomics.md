Atomic Instrumentation
======================

The `atom` and `red` instructions are instrumented precisely, that is,
addresses must be completely in-bounds.  Panoptes performs its instrumentation
of these instructions in `instrument_atom` of `global_context_memcheck.cpp`.

Validity Bit Computation
------------------------

Consider the two canonical atomic operations (for `OP != .cas`):

    red.OP.TYPE [a], b;
    red.cas.TYPE [a], b, c;

For the `.and`, `.or`, `.xor`, and `.exch` atomic operations, the validity bits
of the input `b` passthrough to the final validity bits of `*a`.

For the `.inc`, `.dec`, `.add`, `.min`, and `.max` operations, if these
operations were performed non-atomically the validity bits of `*a` and `b`
could potentially form a wholly invalid result in `*a`.  As a result, Panoptes
pessimistically assumes that any invalid bits in `b` propagate throughout the
entire input value as ensuring fast atomic operation does not permit precise
computation of the result.

For the `.cas` operation, the invalid bits of `b` propagate throughout the
entire resulting value as any single (invalid) bit could affect whether the
value of `c` is stored in `*a`.  Since `c` is stored unmodified (if it is
stored at all), its validity bits passthruogh to the final value of `*a`.

Explicit `.shared` Instrumentation
----------------------------------

`.shared` space operations can occur on two types of addresses, fixed symbols
and variables.  Per the `atest_negative_offsets` test, `ptxas` does not
assemble atomic operations against fixed symbols at negative offsets.

Operations at fixed, symbolic addresses should always succeed.  The validity
bits for these addresses will occur at a known shadow symbol.

Operations at variable addresses are checked for underflow and overflow.

Since there is no universal write target for the `.shared` space, `.shared`
operations are predicated.  If the operation does not occur, the destination
validity bits of `atom` instructions are invalidated.

Explicit `.global` Instrumentation
----------------------------------

Addresses are checked with the address space lookup tables.  For invalid
addresses, the atomic operation itself is performed against the global
write-only symbol.  For `atom` instructions, the results will be nonsensical
because the symbol lacks initialization and is the dumping ground for invalid
write operations.  Nonetheless, the validity bits are computed against the
validity bits stored in the address space lookup table.  For invalid addresses,
these bits will be invalid so the result of any `atom` operation will be
invalid.

Implicit `.generic` Instrumentation
-----------------------------------

Input values are checked according to the rules of `.shared` addresses if
`isspacep.shared` is true, otherwise, the address space lookup tables are
consulted as with `.global` addresses.

Since the instrumentation for `.shared` addresses requires atomic operations be
predicated, generic address-based atomic operations are converted to predicated
instructions.
