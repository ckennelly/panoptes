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

#include <stdint.h>
#include "ptx_grammar.tab.hh"
#include "ptx_ir.h"

using namespace panoptes;

block_t::~block_t() {
    switch (block_type) {
        case block_scope:
            delete scope;
            return;
        case block_statement:
            delete statement;
            return;
        case block_label:
            delete label;
            return;
        case block_invalid:
            return;
    }
}

function_t::function_t() : parent(NULL) { }

scope_t::~scope_t() {
    for (block_vt::iterator it = blocks.begin(); it != blocks.end(); ++it) {
        delete *it;
    }
}

ptx_t::~ptx_t() {
    for (size_t i = 0; i < entries.size(); i++) {
        delete entries[i];
    }
}

statement_t::statement_t() {
    reset();
}

void statement_t::reset() {
    has_predicate   = false;
    is_negated      = false;
    has_ppredicate  = false;
    has_qpredicate  = false;
    saturating      = false;
    uniform         = false;
    op              = op_invalid;
    type2           = invalid_type;
    is_volatile     = false;
    carry_out       = false;
    approximation   = default_approximation;
    rounding        = rounding_default;
    width           = width_default;
    barrier         = barrier_invalid;
    barrier_scope   = invalid_barrier_scope;
    space           = invalid_space;
    cache           = cache_default;
    type            = invalid_type;
    cmp             = cmp_invalid;
    bool_op         = bool_none;
    atomic_op       = atom_invalid;
    vote_mode       = invalid_vote_mode;
    geometry        = geom_invalid;
    ftz             = false;
    is_to           = false;
    has_return_value = false;
    mask            = false;
    shiftamt        = false;
    testp_op        = invalid_testp_op;
    prmt_mode       = invalid_prmt_mode;
    prefetch_cache  = invalid_cache;
    vector          = v1;
    operands.clear();
}

bool operand_t::is_constant() const {
    switch (op_type) {
        case operand_addressable:
        case operand_indexed:
        case operand_identifier:
            return false;
        case operand_constant:
        case operand_float:
        case operand_double:
            return true;
        case invalid_operand:
            assert(0 && "Invalid operand type.");
            return false;
    }

    __builtin_unreachable();
    return false;
}

bool operand_t::operator==(const operand_t & r) const {
    if (r.op_type != op_type) {
        return false;
    }

    if (r.negated != negated) {
        return false;
    }

    switch (op_type) {
        case operand_addressable:
        case operand_indexed:
            if (r.offset != offset) {
                return false;
            }
            /* Fall through. */
        case operand_identifier:
            if (r.identifier != identifier ||
                    r.field != field) {
                return false;
            }
            break;
        case operand_constant:
            return r.offset == offset;
        case operand_float:
            return r.fvalue == fvalue;
        case operand_double:
            return r.dvalue == dvalue;
        case invalid_operand:
            break;
    }

    return true;
}

bool operand_t::operator!=(const operand_t & r) const {
    if (r.op_type != op_type) {
        return true;
    }

    if (r.negated != negated) {
        return true;
    }

    switch (op_type) {
        case operand_addressable:
        case operand_indexed:
            if (r.offset != offset) {
                return true;
            }
            /* Fall through. */
        case operand_identifier:
            if (r.identifier != identifier ||
                    r.field != field) {
                return true;
            }
            break;
        case operand_constant:
            return r.offset != offset;
        case operand_float:
            return r.fvalue != fvalue;
        case operand_double:
            return r.dvalue != dvalue;
        case invalid_operand:
            break;
    }

    return false;
}

operand_t::operand_t() { reset(); }

operand_t operand_t::make_addressable(const std::string & id,
        int64_t offset) {
    operand_t ret;
    ret.op_type = operand_addressable;
    ret.identifier.push_back(id);
    ret.field.push_back(field_none);
    ret.offset = offset;
    return ret;
}

operand_t operand_t::make_identifier(const std::string & id) {
    operand_t ret;
    ret.op_type = operand_identifier;
    ret.identifier.push_back(id);
    ret.field.push_back(field_none);
    return ret;
}

operand_t operand_t::make_iconstant(int64_t i) {
    operand_t ret;
    ret.op_type = operand_constant;
    ret.offset  = i;
    return ret;
}

operand_t operand_t::make_fconstant(float   f) {
    operand_t ret;
    ret.op_type = operand_float;
    ret.fvalue  = f;
    return ret;
}

operand_t operand_t::make_dconstant(double  d) {
    operand_t ret;
    ret.op_type = operand_double;
    ret.dvalue  = d;
    return ret;
}

void operand_t::reset() {
    op_type = invalid_operand;
    identifier.clear();
    field.clear();
    offset = 0;
    negated = false;
}

void operand_t::push_field(int token) {
    switch (token) {
        case TOKEN_X:
            field.push_back(field_x);
            return;
        case TOKEN_Y:
            field.push_back(field_y);
            return;
        case TOKEN_Z:
            field.push_back(field_z);
            return;
        case TOKEN_W:
            field.push_back(field_w);
                return;
        default:
            assert(0 && "Unknown field accessor token.");
            break;
    }
}

void statement_t::set_atomic_op(int token) {
    switch (token) {
        case TOKEN_AND:
            atomic_op = atom_and;
            return;
        case TOKEN_OR:
            atomic_op = atom_or;
            return;
        case TOKEN_XOR:
            atomic_op = atom_xor;
            return;
        case TOKEN_CAS:
            atomic_op = atom_cas;
            return;
        case TOKEN_EXCH:
            atomic_op = atom_exch;
            return;
        case TOKEN_ADD:
            atomic_op = atom_add;
            return;
        case TOKEN_INC:
            atomic_op = atom_inc;
            return;
        case TOKEN_DEC:
            atomic_op = atom_dec;
            return;
        case TOKEN_MIN:
            atomic_op = atom_min;
            return;
        case TOKEN_MAX:
            atomic_op = atom_max;
            return;
    }

    assert(0 && "Unknown atomic operation token.");
}

void statement_t::set_geometry(int token) {
    switch (token) {
        case TOKEN_1D:
            geometry = geom_1d;
            return;
        case TOKEN_2D:
            geometry = geom_2d;
            return;
        case TOKEN_3D:
            geometry = geom_3d;
            return;
        case TOKEN_A1D:
            geometry = geom_a1d;
            return;
        case TOKEN_A2D:
            geometry = geom_a2d;
            return;
        case TOKEN_CUBE:
            geometry = geom_cube;
            return;
        case TOKEN_ACUBE:
            geometry = geom_acube;
            return;
    }

    assert(0 && "Unknown geometry token.");
}

void statement_t::set_token(int token) {
    switch (token) {
        case OPCODE_ABS:
            op = op_abs;
            return;
        case OPCODE_ADD:
            op = op_add;
            return;
        case OPCODE_ADDC:
            op = op_addc;
            return;
        case OPCODE_AND:
            op = op_and;
            return;
        case OPCODE_ATOM:
            op = op_atom;
            return;
        case OPCODE_BAR:
            op = op_bar;
            return;
        case OPCODE_BFE:
            op = op_bfe;
            return;
        case OPCODE_BFI:
            op = op_bfi;
            return;
        case OPCODE_BFIND:
            op = op_bfind;
            return;
        case OPCODE_BRA:
            op = op_bra;
            return;
        case OPCODE_BREV:
            op = op_brev;
            return;
        case OPCODE_BRKPT:
            op = op_brkpt;
            return;
        case OPCODE_CALL:
            op = op_call;
            return;
        case OPCODE_CLZ:
            op = op_clz;
            return;
        case OPCODE_CNOT:
            op = op_cnot;
            return;
        case OPCODE_COPYSIGN:
            op = op_copysign;
            return;
        case OPCODE_COS:
            op = op_cos;
            return;
        case OPCODE_CVT:
            op = op_cvt;
            return;
        case OPCODE_CVTA:
            op = op_cvta;
            return;
        case OPCODE_DIV:
            op = op_div;
            return;
        case OPCODE_EX2:
            op = op_ex2;
            return;
        case OPCODE_EXIT:
            op = op_exit;
            return;
        case OPCODE_FMA:
            op = op_fma;
            return;
        case OPCODE_ISSPACEP:
            op = op_isspacep;
            return;
        case OPCODE_LD:
            op = op_ld;
            return;
        case OPCODE_LDU:
            op = op_ldu;
            return;
        case OPCODE_LG2:
            op = op_lg2;
            return;
        case OPCODE_MAD:
            op = op_mad;
            return;
        case OPCODE_MAD24:
            op = op_mad24;
            return;
        case OPCODE_MADC:
            op = op_madc;
            return;
        case OPCODE_MAX:
            op = op_max;
            return;
        case OPCODE_MEMBAR:
            op = op_membar;
            return;
        case OPCODE_MIN:
            op = op_min;
            return;
        case OPCODE_MOV:
            op = op_mov;
            return;
        case OPCODE_MUL:
            op = op_mul;
            return;
        case OPCODE_MUL24:
            op = op_mul24;
            return;
        case OPCODE_NEG:
            op = op_neg;
            return;
        case OPCODE_NOT:
            op = op_not;
            return;
        case OPCODE_OR:
            op = op_or;
            return;
        case OPCODE_PMEVENT:
            op = op_pmevent;
            return;
        case OPCODE_POPC:
            op = op_popc;
            return;
        case OPCODE_PREFETCH:
            op = op_prefetch;
            return;
        case OPCODE_PREFETCHU:
            op = op_prefetchu;
            return;
        case OPCODE_PRMT:
            op = op_prmt;
            return;
        case OPCODE_RCP:
            op = op_rcp;
            return;
        case OPCODE_RED:
            op = op_red;
            return;
        case OPCODE_REM:
            op = op_rem;
            return;
        case OPCODE_RET:
            op = op_ret;
            return;
        case OPCODE_RSQRT:
            op = op_rsqrt;
            return;
        case OPCODE_SAD:
            op = op_sad;
            return;
        case OPCODE_SELP:
            op = op_selp;
            return;
        case OPCODE_SET:
            op = op_set;
            return;
        case OPCODE_SETP:
            op = op_setp;
            return;
        case OPCODE_SHL:
            op = op_shl;
            return;
        case OPCODE_SHR:
            op = op_shr;
            return;
        case OPCODE_SIN:
            op = op_sin;
            return;
        case OPCODE_SLCT:
            op = op_slct;
            return;
        case OPCODE_SQRT:
            op = op_sqrt;
            return;
        case OPCODE_ST:
            op = op_st;
            return;
        case OPCODE_SUB:
            op = op_sub;
            return;
        case OPCODE_SUBC:
            op = op_subc;
            return;
        case OPCODE_SULD:
            op = op_suld;
            return;
        case OPCODE_SUQ:
            op = op_suq;
            return;
        case OPCODE_SURED:
            op = op_sured;
            return;
        case OPCODE_SUST:
            op = op_sust;
            return;
        case OPCODE_TESTP:
            op = op_testp;
            return;
        case OPCODE_TEX:
            op = op_tex;
            return;
        case OPCODE_TLD4:
            op = op_tld4;
            return;
        case OPCODE_TRAP:
            op = op_trap;
            return;
        case OPCODE_TXQ:
            op = op_txq;
            return;
        case OPCODE_VABSDIFF:
            op = op_vabsdiff;
            return;
        case OPCODE_VADD:
            op = op_vadd;
            return;
        case OPCODE_VMAD:
            op = op_vmad;
            return;
        case OPCODE_VMAX:
            op = op_vmax;
            return;
        case OPCODE_VMIN:
            op = op_vmin;
            return;
        case OPCODE_VOTE:
            op = op_vote;
            return;
        case OPCODE_VSET:
            op = op_vset;
            return;
        case OPCODE_VSHL:
            op = op_vshl;
            return;
        case OPCODE_VSHR:
            op = op_vshr;
            return;
        case OPCODE_VSUB:
            op = op_vsub;
            return;
        case OPCODE_XOR:
            op = op_xor;
            return;
        case TOKEN_CONST:
            space = const_space;
            return;
        case TOKEN_GLOBAL:
            space = global_space;
            return;
        case TOKEN_LOCAL:
            space = local_space;
            return;
        case TOKEN_PARAM:
            space = param_space;
            return;
        case TOKEN_SHARED:
            space = shared_space;
            return;
        case TOKEN_GENERIC:
            space = generic_space;
            return;
        case TOKEN_CA:
            cache = cache_ca;
            return;
        case TOKEN_CG:
            cache = cache_cg;
            return;
        case TOKEN_CS:
            cache = cache_cs;
            return;
        case TOKEN_CV:
            cache = cache_cv;
            return;
        case TOKEN_LU:
            cache = cache_lu;
            return;
        case TOKEN_V2:
            vector = v2;
            return;
        case TOKEN_V4:
            vector = v4;
            return;
        case TOKEN_EQ:
            cmp = cmp_eq;
            return;
        case TOKEN_NE:
            cmp = cmp_ne;
            return;
        case TOKEN_LT:
            cmp = cmp_lt;
            return;
        case TOKEN_LE:
            cmp = cmp_le;
            return;
        case TOKEN_GT:
            cmp = cmp_gt;
            return;
        case TOKEN_GE:
            cmp = cmp_ge;
            return;
        case TOKEN_LO:
            cmp = cmp_lo;
            return;
        case TOKEN_LS:
            cmp = cmp_ls;
            return;
        case TOKEN_HI:
            cmp = cmp_hi;
            return;
        case TOKEN_HS:
            cmp = cmp_hs;
            return;
        case TOKEN_EQU:
            cmp = cmp_equ;
            return;
        case TOKEN_NEU:
            cmp = cmp_neu;
            return;
        case TOKEN_LTU:
            cmp = cmp_ltu;
            return;
        case TOKEN_GTU:
            cmp = cmp_gtu;
            return;
        case TOKEN_GEU:
            cmp = cmp_geu;
            return;
        case TOKEN_NUM:
            cmp = cmp_num;
            return;
        case TOKEN_NAN:
            cmp = cmp_nan;
            return;
        case TOKEN_AND:
            bool_op = bool_and;
            return;
        case TOKEN_OR:
            bool_op = bool_or;
            return;
        case TOKEN_XOR:
            bool_op = bool_xor;
            return;
        case TOKEN_POPC:
            bool_op = bool_popc;
            return;
        case TOKEN_RNI:
            rounding = rounding_rni;
            return;
        case TOKEN_RZI:
            rounding = rounding_rzi;
            return;
        case TOKEN_RMI:
            rounding = rounding_rmi;
            return;
        case TOKEN_RPI:
            rounding = rounding_rpi;
            return;
        case TOKEN_RN:
            rounding = rounding_rn;
            return;
        case TOKEN_RZ:
            rounding = rounding_rz;
            return;
        case TOKEN_RM:
            rounding = rounding_rm;
            return;
        case TOKEN_RP:
            rounding = rounding_rp;
            return;
        case TOKEN_SYNC:
            barrier = barrier_sync;
            return;
        case TOKEN_ARRIVE:
            barrier = barrier_arrive;
            return;
        case TOKEN_RED:
            barrier = barrier_reduce;
            return;
        case TOKEN_FULL:
            approximation = full_approximation;
            return;
        case TOKEN_APPROX:
            approximation = approximate;
            return;
        case TOKEN_MCTA:
            barrier_scope = barrier_cta;
            return;
        case TOKEN_MGL:
            barrier_scope = barrier_gl;
            return;
        case TOKEN_MSYS:
            barrier_scope = barrier_sys;
            return;
        case TOKEN_ALL:
            vote_mode       = vote_all;
            return;
        case TOKEN_ANY:
            vote_mode       = vote_any;
            return;
        case TOKEN_BALLOT:
            vote_mode       = vote_ballot;
            return;
        case TOKEN_UNI:
            vote_mode       = vote_uniform;
            return;
        case TOKEN_FINITE:
            testp_op = testp_finite;
            return;
        case TOKEN_INFINITE:
            testp_op = testp_infinite;
            return;
        case TOKEN_NUMBER:
            testp_op = testp_number;
            return;
        case TOKEN_NOTANUMBER:
            testp_op = testp_nan;
            return;
        case TOKEN_NORMAL:
            testp_op = testp_normal;
            return;
        case TOKEN_SUBNORMAL:
            testp_op = testp_subnormal;
            return;
        case TOKEN_F4E:
            prmt_mode = prmt_f4e;
            return;
        case TOKEN_B4E:
            prmt_mode = prmt_b4e;
            return;
        case TOKEN_RC8:
            prmt_mode = prmt_rc8;
            return;
        case TOKEN_ECL:
            prmt_mode = prmt_ecl;
            return;
        case TOKEN_ECR:
            prmt_mode = prmt_ecr;
            return;
        case TOKEN_RC16:
            prmt_mode = prmt_rc16;
            return;
        case TOKEN_L1:
            prefetch_cache = cache_L1;
            return;
        case TOKEN_L2:
            prefetch_cache = cache_L2;
            return;
        default:
            assert(0 && "Unknown token.");
            return;
    }
}

void statement_t::set_width(int token) {
    switch (token) {
        case TOKEN_LO:
            width    = width_lo;
            return;
        case TOKEN_HI:
            width    = width_hi;
            return;
        case TOKEN_WIDE:
            width    = width_wide;
            return;
        default:
            assert(0 && "Unknown token.");
            return;
    }
}

void statement_t::set_operands(const std::vector<operand_t> & ops) {
    operands = ops;
}

texture_t::texture_t() : type(invalid_type) { }

variable_t::variable_t() : linkage(linkage_default),
        space(invalid_space), vector_size(v1), type(invalid_type),
        is_ptr(false), has_align(false), alignment(1),
        has_suffix(false), suffix(-1), is_array(false),
        array_dimensions(0), array_flexible(false),
        has_initializer(false), initializer_vector(false) { }

size_t variable_t::size() const {
    size_t base_size;
    switch (type) {
        case b8_type:
        case s8_type:
        case u8_type:
            base_size = 1u;
            break;
        case b16_type:
        case f16_type:
        case s16_type:
        case u16_type:
            base_size = 2u;
            break;
        case b32_type:
        case f32_type:
        case s32_type:
        case u32_type:
            base_size = 4u;
            break;
        case b64_type:
        case f64_type:
        case s64_type:
        case u64_type:
            base_size = 8u;
            break;
        case pred_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type for computing the size.");
            return 0;
    }

    switch (vector_size) {
        case v1:
            base_size *= 1u;
            break;
        case v2:
            base_size *= 2u;
            break;
        case v3:
            base_size *= 3u;
            break;
        case v4:
            base_size *= 4u;
            break;
    }

    size_t scale = 1u;
    if (is_array) {
        assert(!(array_flexible) && "TODO Support flexible arrays");

        for (unsigned i = 0; i < array_dimensions; i++) {
            scale *= array_size[i];
        }
    }

    return base_size * scale;
}

void variable_t::set_token(int token) {
    switch (token) {
        case TOKEN_CONST:   space = const_space; break;
        case TOKEN_GLOBAL:  space = global_space; break;
        case TOKEN_LOCAL:   space = local_space;  break;
        case TOKEN_REG:     space = reg_space; break;
        case TOKEN_SHARED:  space = shared_space; break;
        case TOKEN_PARAM:   space = param_space; break;
        default:
            assert(0 && "Unknown variable token.");
            break;
    }
}
