/**
 * Panoptes - A Binary Translation Framework for CUDA
 * (c) 2011-2012 Chris Kennelly <chris@ckennelly.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __PANOPTES__PTX_IR_H__
#define __PANOPTES__PTX_IR_H__

#include <boost/bimap.hpp>
#include <list>
#include <string>
#include <vector>

namespace panoptes {

typedef std::string string_t;

enum type_t {
    b8_type, b16_type, b32_type, b64_type,
    u8_type, u16_type, u32_type, u64_type,
    s8_type, s16_type, s32_type, s64_type,
             f16_type, f32_type, f64_type,
    pred_type, texref_type, invalid_type
};

enum space_t {
    const_space, global_space, local_space, shared_space, generic_space,
    param_space, reg_space, invalid_space
};

enum vector_t {
    v1 = 1, v2 = 2, v3 = 3, v4 = 4
};

enum linkage_t {
    linkage_default, linkage_extern, linkage_visible
};

enum variant_type_t {
    variant_integer, variant_single, variant_double
};

struct variant_t {
    variant_type_t  type;
    union {
        double      d;
        float       f;
        uint64_t    u;
    } data;
};

struct variable_t {
    variable_t();

    linkage_t       linkage;
    space_t         space;
    vector_t        vector_size;
    type_t          type;
    bool            is_ptr;
    bool            has_align;
    size_t          alignment;
    std::string     name;
    bool            has_suffix;
    int             suffix;
    bool            is_array;
    unsigned        array_dimensions;
    bool            array_flexible;
    size_t          array_size[3];

    bool            has_initializer;
    bool            initializer_vector;
    typedef std::vector<variant_t> variant_vt;
    variant_vt      initializer;

    void set_token(int token);
    size_t size() const;
};

struct param_t : public variable_t {
    param_t() {
        alignment = 4u;
    }
};

enum op_t {
    op_abs,      op_add,      op_addc,      op_and,      op_atom,
    op_bar,      op_bfe,      op_bfi,       op_bfind,    op_bra,
    op_brev,     op_brkpt,    op_call,      op_clz,      op_cnot,
    op_copysign, op_cos,      op_cvt,       op_cvta,     op_div,
    op_ex2,      op_exit,     op_fma,       op_isspacep, op_ld,
    op_ldu,      op_lg2,      op_mad,       op_mad24,    op_madc,
    op_max,      op_membar,   op_min,       op_mov,      op_mul,
    op_mul24,    op_neg,      op_not,       op_or,       op_pmevent,
    op_popc,     op_prefetch, op_prefetchu, op_prmt,     op_rcp,
    op_red,      op_rem,      op_ret,       op_rsqrt,    op_sad,
    op_selp,     op_set,      op_setp,      op_shl,      op_shr,
    op_sin,      op_slct,     op_sqrt,      op_st,       op_sub,
    op_subc,     op_suld,     op_suq,       op_sured,    op_sust,
    op_testp,    op_tex,      op_tld4,      op_trap,     op_txq,
    op_vabsdiff, op_vadd,     op_vmad,      op_vmax,     op_vmin,
    op_vote,     op_vset,     op_vshl,      op_vshr,     op_vsub,
    op_xor,      op_invalid
};

enum cache_t {
    cache_ca, cache_cg, cache_cs, cache_cv, cache_lu, cache_wb, cache_wt,
    cache_default, cache_invalid
};

enum barrier_op_t {
    barrier_sync, barrier_arrive, barrier_reduce, barrier_invalid
};

enum rounding_t {
    rounding_default, rounding_rn,     rounding_rz,   rounding_rm,
    rounding_rp,      rounding_rni,    rounding_rzi,  rounding_rmi,
    rounding_rpi,     rounding_invalid
};

enum op_mul_width_t {
    width_default, width_hi, width_lo, width_wide
};

enum op_set_cmp_t {
    cmp_eq,  cmp_ne,  cmp_lt,  cmp_le,  cmp_gt,  cmp_ge,
    cmp_lo,  cmp_ls,  cmp_hi,  cmp_hs,  cmp_equ, cmp_neu,
    cmp_ltu, cmp_leu, cmp_gtu, cmp_geu, cmp_num, cmp_nan,
    cmp_invalid
};

enum op_set_bool_t {
    bool_none, bool_and, bool_or, bool_xor, bool_popc
};

enum geom_t {
    geom_1d, geom_2d, geom_3d, geom_a1d, geom_a2d, geom_cube, geom_acube,
    geom_invalid
};

enum atom_op_t {
    atom_and, atom_or,   atom_xor, atom_inc,
    atom_dec, atom_add,  atom_min, atom_max,
    atom_cas, atom_exch, atom_invalid
};

enum field_t {
    field_x, field_y, field_z, field_w, field_none
};

enum operand_type_t {
    operand_identifier, operand_addressable, operand_indexed, operand_constant,
    operand_float,      operand_double,      invalid_operand
};

struct operand_t {
    operand_t();
    void reset();

    static operand_t make_addressable(const std::string & id,
        int64_t offset);
    static operand_t make_identifier(const std::string & id);
    static operand_t make_iconstant(int64_t i);
    static operand_t make_fconstant(float   f);
    static operand_t make_dconstant(double  d);

    bool operator==(const operand_t & r) const;
    bool operator!=(const operand_t & r) const;
    bool is_constant() const;

    operand_type_t op_type;
    bool negated;

    typedef std::vector<std::string> string_vt;
    string_vt   identifier;

    typedef std::vector<bool> constness_vt;
    constness_vt constness;

    void push_field(int token);
    typedef std::vector<field_t> field_vt;
    field_vt    field;

    union {
        int64_t     offset;
        float       fvalue;
        double      dvalue;
    };
};

enum approximation_t {
    default_approximation, approximate, full_approximation
};

enum barrier_scope_t {
    invalid_barrier_scope, barrier_cta, barrier_gl, barrier_sys
};

enum vote_mode_t {
    invalid_vote_mode, vote_all, vote_any, vote_uniform, vote_ballot
};

enum testp_op_t {
    invalid_testp_op, testp_finite, testp_infinite, testp_number,
    testp_nan, testp_normal, testp_subnormal
};

enum prmt_mode_t {
    invalid_prmt_mode, prmt_default, prmt_f4e, prmt_b4e, prmt_rc8, prmt_ecl,
    prmt_ecr, prmt_rc16
};

enum prefetch_cache_t {
    invalid_cache, cache_L1, cache_L2
};

struct statement_t {
    statement_t();
    void reset();
    void set_width(int token);
    void set_token(int token);
    void set_atomic_op(int token);
    void set_geometry(int token);
    void set_operands(const std::vector<operand_t> & operands);

    bool            has_predicate;
    bool            is_negated;
    string_t        predicate;
    bool            has_ppredicate;
    string_t        ppredicate;
    bool            has_qpredicate;
    string_t        qpredicate;
    bool            saturating;
    bool            uniform;
    op_t            op;
    bool            is_volatile;
    bool            carry_out;
    approximation_t approximation;
    rounding_t      rounding;
    op_mul_width_t  width;
    barrier_op_t    barrier;
    barrier_scope_t barrier_scope;
    space_t         space;
    cache_t         cache;
    type_t          type;
    type_t          type2;
    op_set_cmp_t    cmp;
    op_set_bool_t   bool_op;
    atom_op_t       atomic_op;
    vote_mode_t     vote_mode;
    geom_t          geometry;
    bool            ftz;
    bool            is_to;
    bool            has_return_value;
    bool            mask;
    bool            shiftamt;
    testp_op_t      testp_op;
    prmt_mode_t     prmt_mode;
    prefetch_cache_t prefetch_cache;

    vector_t        vector;

    /* We always go in order of operands without regard for naming */
    typedef std::vector<operand_t> operand_vt;
    operand_vt      operands;
};

struct scope_t;
struct label_t;

enum block_type_t {
    block_scope, block_statement, block_label, block_invalid
};

struct function_t;

struct block_t : boost::noncopyable {
    block_t() : block_type(block_invalid), parent(NULL), fparent(NULL),
        invalid(NULL) {}
    block_t(block_type_t t) : block_type(t), parent(NULL), fparent(NULL),
        invalid(NULL) {}
    ~block_t();

    block_type_t block_type;
    block_t *    parent;
    function_t * fparent;
    union {
        scope_t * scope;
        statement_t * statement;
        label_t * label;
        void * invalid;
    };
};

struct scope_t {
    scope_t() { }
    ~scope_t();

    typedef std::vector<variable_t> variable_vt;
    variable_vt     variables;
    typedef std::list<block_t *> block_vt;
    block_vt        blocks;
};

struct label_t {
    std::string label;
};

struct ptx_t;

struct function_t {
    function_t();

    linkage_t       linkage;
    bool            entry;
    std::string     entry_name;

    ptx_t *         parent;

    bool            has_return_value;
    param_t         return_value;

    typedef std::vector<param_t> param_vt;
    param_vt        params;
    bool            no_body;
    block_t         scope;
};

enum sm_t {
    SM10, SM11, SM12, SM13, SM20, SM21, DEBUG
};

struct texture_t {
    texture_t();

    type_t type;

    typedef std::vector<std::string> string_vt;
    string_vt names;
};

struct ptx_t : boost::noncopyable {
    ~ptx_t();

    unsigned version_major;
    unsigned version_minor;

    sm_t sm;
    bool map_f64_to_f32;
    unsigned address_size;

    typedef std::vector<texture_t> texture_vt;
    texture_vt      textures;

    typedef std::vector<variable_t> variable_vt;
    variable_vt     variables;

    typedef std::vector<function_t *> entry_vt;
    entry_vt        entries;
};

} // end namespace panoptes

#endif // __PANOPTES__PTX_IR_H__
