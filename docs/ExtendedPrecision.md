Extended Precision Instrumention
================================

The `addc`, `subc`, and `madc` instructions, coupled with the carry-out flag
(`.cc`) enable extended-precision mathematical operations to be performed.  All
instructions using the carry-in or carry-out functionality operate only on u32
and s32 types (as of PTX ISA 3.0).  This simplifies the handling of the
validity state of the carry flag:  Since all operations will be 4 bytes wide,
the validity state can be kept as a 4 byte integer without any further
conversions.  (If a mix of widths was supported, a canonical width would have
to be chosen and operations in noncanonical sizes would require conversion.)

At all times, the carry flag (stored in register `__panoptes_vcarry`) is either
`0` (valid) or `0xFFFFFFFF` (invalid).

Carry-In Functionality
----------------------

We perform a bitwise OR of the carry validity bits with our input arguments.
If the carry flag is invalid, the entire output of the `addc/subc/madc`
operation will be invalid as the carried bit would propagate left.  (This left
propagation has been precomputed by how this flag is stored, i.e. all-1's.)

Carry-Out Functionality
-----------------------

For compile-time constants, the carry-flag is cleared.

For validity of the most significant output bit determines validity of the
carry flag.  To propagate the validity across the entire carry flag, we take
the resulting validity bits and perform a _signed_ right shift by 31.
