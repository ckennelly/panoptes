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

#include "context_memcheck.h"
#include "context_memcheck_internal.h"
#include <cstdio>
#include "global_context_memcheck.h"
#include "logger.h"
#include "ptx_formatter.h"

using namespace panoptes;

typedef boost::unique_lock<boost::mutex> scoped_lock;

using internal::__master_symbol;
const char * panoptes::internal::__master_symbol = "__panoptes_master_symbol";

static const char * __global_ro       = "__panoptes_global_ro";
static const char * __global_wo       = "__panoptes_global_wo";
static const char * __validity_prefix = "__panoptes_v";
static const char * __texture_prefix  = "__panoptes_t";
static const char * __errors_register = "__panoptes_lerrors";
static const char * __has_errors      = "__panoptes_haserrors";
static const char * __vcarry          = "__panoptes_carry";
static const char * __reg_global_errors  = "__panoptes_gerrors";
static const char * __reg_global_errorsw = "__panoptes_gerrorsw";
static const char * __errors_address  = "__panoptes_eaddr";
static const char * __reg_error_count = "__panoptes_recount";
static const char * __zero_register8  = "__panoptes_zero8";
static const char * __zero_register16 = "__panoptes_zero16";
static const char * __zero_register32 = "__panoptes_zero32";
static const char * __zero_register64 = "__panoptes_zero64";

static const char * __shared_reg      = "__panoptes_shared_size";
static const char * __local_reg       = "__panoptes_local_size";

static const char * __param_shared    = "__panoptes_shared_sizep";
static const char * __param_error     = "__panoptes_errors";
static const char * __param_error_count = "__panoptes_error_count";

static const size_t metadata_size     = 3u * sizeof(void *);

static type_t pointer_type() {
    switch (sizeof(void *)) {
        case 4u: return b32_type;
        case 8u: return b64_type;
    }

    __builtin_unreachable();
}

static type_t upointer_type() {
    switch (sizeof(void *)) {
        case 4u: return u32_type;
        case 8u: return u64_type;
    }

    __builtin_unreachable();
}

static size_t sizeof_type(type_t t) {
    switch (t) {
        case b8_type:
        case s8_type:
        case u8_type:
            return 1u;
        case b16_type:
        case f16_type:
        case s16_type:
        case u16_type:
            return 2u;
        case b32_type:
        case f32_type:
        case s32_type:
        case u32_type:
            return 4u;
        case b64_type:
        case f64_type:
        case s64_type:
        case u64_type:
            return 8u;
        case pred_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            return 0;
    }

    __builtin_unreachable();
}

static type_t bitwise_type(type_t t) {
    switch (t) {
        case b8_type:
        case s8_type:
        case u8_type:
            return b8_type;
        case b16_type:
        case f16_type:
        case s16_type:
        case u16_type:
            return b16_type;
        case b32_type:
        case f32_type:
        case s32_type:
        case u32_type:
            return b32_type;
        case b64_type:
        case f64_type:
        case s64_type:
        case u64_type:
            return b64_type;
        case pred_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            return invalid_type;
    }

    __builtin_unreachable();
    return invalid_type;
}

static type_t signed_type(type_t t) {
    switch (t) {
        case b8_type:
        case s8_type:
        case u8_type:
            return s8_type;
        case b16_type:
        case f16_type:
        case s16_type:
        case u16_type:
            return s16_type;
        case b32_type:
        case f32_type:
        case s32_type:
        case u32_type:
            return s32_type;
        case b64_type:
        case f64_type:
        case s64_type:
        case u64_type:
            return s64_type;
        case pred_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            return invalid_type;
    }

    __builtin_unreachable();
    return invalid_type;
}

static type_t bitwise_of(size_t s) {
    if (s > 4u) {
        return b64_type;
    } else if (s > 2u) {
        return b32_type;
    } else if (s > 1u) {
        return b16_type;
    } else {
        return b8_type;
    }
}

static type_t signed_of(size_t s) {
    if (s > 4u) {
        return s64_type;
    } else if (s > 2u) {
        return s32_type;
    } else if (s > 1u) {
        return s16_type;
    } else {
        return s8_type;
    }
}

static type_t unsigned_of(size_t s) {
    if (s > 32u) {
        return u64_type;
    } else if (s > 16u) {
        return u32_type;
    } else if (s > 8u) {
        return u16_type;
    } else {
        return u8_type;
    }
}

static statement_t make_add(type_t type, const operand_t & dst,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op = op_add;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_add(type_t type, const operand_t & dst,
        const operand_t & a, const int64_t b) {
    statement_t ret;
    ret.op = op_add;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(operand_t::make_iconstant(b));
    return ret;
}

static statement_t make_and(type_t type, const operand_t & dst,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op = op_and;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_and(type_t type, const operand_t & dst,
        const operand_t & a, const int64_t b) {
    statement_t ret;
    ret.op = op_and;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(operand_t::make_iconstant(b));
    return ret;
}

static statement_t make_bfe(type_t type, const operand_t & d,
        const operand_t & a, const operand_t & b, const operand_t & c) {
    statement_t ret;
    ret.op   = op_bfe;
    ret.type = type;
    ret.operands.push_back(d);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    ret.operands.push_back(c);
    return ret;
}

static statement_t make_bfi(type_t type, const operand_t & f,
        const operand_t & a, const operand_t & b, const operand_t & c,
        const operand_t & d) {
    statement_t ret;
    ret.op   = op_bfi;
    ret.type = type;
    ret.operands.push_back(f);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    ret.operands.push_back(c);
    ret.operands.push_back(d);
    return ret;
}

static statement_t make_brev(type_t type, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op   = op_brev;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_cvt(type_t type, type_t type2, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op = op_cvt;
    ret.type = type;
    ret.type2 = type2;
    ret.saturating = false;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_cnot(type_t type, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op   = op_cnot;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_isspacep(space_t space, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op = op_isspacep;
    ret.space = space;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);

    return ret;
}

static statement_t make_ld(type_t type, space_t space, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op = op_ld;
    ret.space = space;
    ret.type = type;

    ret.operands.push_back(dst);
    ret.operands.push_back(src);

    return ret;
}

static statement_t make_min(type_t type, const operand_t & dst,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op   = op_min;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_mov(type_t type, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op   = op_mov;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_mov(type_t type, const operand_t & dst, int64_t src) {
    statement_t ret;
    ret.op   = op_mov;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(operand_t::make_iconstant(src));
    return ret;
}

static statement_t make_mul(type_t type, const op_mul_width_t width,
        const operand_t & dst, const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op    = op_mul;
    ret.type  = type;
    ret.width = width;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_neg(type_t type, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op   = op_neg;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_not(type_t type, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op   = op_not;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_or(type_t type, const operand_t & d,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op = op_or;
    ret.type = type;
    ret.operands.push_back(d);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_setp(type_t type, op_set_cmp_t cmp,
        const std::string & pred, const operand_t & lhs,
        const operand_t & rhs) {
    statement_t ret;
    ret.op   = op_setp;
    ret.type = type;
    ret.cmp  = cmp;
    ret.has_ppredicate = true;
    ret.ppredicate = pred;
    ret.operands.push_back(lhs);
    ret.operands.push_back(rhs);
    return ret;
}

static statement_t make_setp(type_t type, op_set_cmp_t cmp,
        const std::string & pred, const operand_t & lhs,
        const int64_t rhs) {
    statement_t ret;
    ret.op   = op_setp;
    ret.type = type;
    ret.cmp  = cmp;
    ret.has_ppredicate = true;
    ret.ppredicate = pred;
    ret.operands.push_back(lhs);
    ret.operands.push_back(operand_t::make_iconstant(rhs));
    return ret;
}

static statement_t make_setp(type_t type, op_set_cmp_t cmp,
        const std::string & pred, const std::string & qpred,
        const operand_t & lhs, const operand_t & rhs) {
    statement_t ret;
    ret.op   = op_setp;
    ret.type = type;
    ret.cmp  = cmp;
    ret.has_ppredicate = true;
    ret.ppredicate = pred;
    ret.has_qpredicate = true;
    ret.qpredicate = qpred;
    ret.operands.push_back(lhs);
    ret.operands.push_back(rhs);
    return ret;
}

static statement_t make_slct(type_t type, type_t type2, const operand_t & d,
        const operand_t & a, const operand_t & b, const operand_t & c) {
    statement_t ret;
    ret.op    = op_slct;
    ret.type  = type;
    ret.type2 = type2;
    ret.operands.push_back(d);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    ret.operands.push_back(c);
    return ret;
}

static statement_t make_shl(type_t type, const operand_t & dst,
        const operand_t & src, const int64_t shift) {
    statement_t ret;
    ret.op = op_shl;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    ret.operands.push_back(operand_t::make_iconstant(shift));
    return ret;
}

static statement_t make_shr(type_t type, const operand_t & dst,
        const operand_t & src, const operand_t & shift) {
    statement_t ret;
    ret.op = op_shr;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    ret.operands.push_back(shift);
    return ret;
}

static statement_t make_shr(type_t type, const operand_t & dst,
        const operand_t & src, const int64_t shift) {
    statement_t ret;
    ret.op = op_shr;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    ret.operands.push_back(operand_t::make_iconstant(shift));
    return ret;
}

static statement_t make_sub(type_t type, const operand_t & dst,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op = op_sub;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_sub(type_t type, const operand_t & dst,
        const operand_t & a, const int64_t b) {
    statement_t ret;
    ret.op = op_sub;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(operand_t::make_iconstant(b));
    return ret;
}

static statement_t make_selp(type_t type, const std::string & pred,
        const operand_t & dst, const operand_t & lhs, const operand_t & rhs) {
    statement_t ret;
    ret.op = op_selp;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(lhs);
    ret.operands.push_back(rhs);
    ret.operands.push_back(operand_t::make_identifier(pred));
    return ret;
}

static statement_t make_xor(type_t type, const operand_t & d,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op = op_xor;
    ret.type = type;
    ret.operands.push_back(d);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static std::string make_validity_symbol(const std::string & in) {
    assert(in.size() > 0);
    if (in[0] == '%') {
        return "%__panoptes_v" + in.substr(1);
    } else {
        return __validity_prefix + in;
    }
}

static operand_t make_validity_operand(const operand_t & in, size_t width) {
    operand_t ret;

    size_t ni, nf;
    switch (in.op_type) {
        case operand_indexed:
        case operand_identifier:
        case operand_addressable:
            /**
             * Addressable operands need not have fields.
             */
            ni = in.identifier.size();
            nf = in.field.size();
            assert(ni >= nf);

            ret.op_type = operand_identifier;
            for (size_t i = 0; i < ni; i++) {
                field_t f = i < nf ? in.field[i] : field_none;

                const std::string & id = in.identifier[i];
                const size_t id_size = id.size();
                bool constant = false;

                const char pm[]  = "%pm";
                const char env[] = "%envreg";

                if (id == "%tid" || id == "%ntid" || id == "%laneid" ||
                        id == "%warpid" || id == "%nwarpid" ||
                        id == "%ctaid" || id == "%nctaid" || id == "%smid" ||
                        id == "%nsmid" || id == "%gridid" ||
                        id == "%lanemask_eq" || id == "%lanemask_le" ||
                        id == "%lanemask_lt" || id == "%lanemask_ge" ||
                        id == "%lanemask_gt" || id == "%clock" ||
                        id == "%clock64" || id == "WARP_SZ") {
                    constant = true;
                } else {
                    if (memcmp(id.c_str(), pm, std::min(id_size,
                            sizeof(pm) - 1u)) == 0) {
                        int s;
                        int p = sscanf(id.c_str() + sizeof(pm) - 1u, "%d", &s);
                        if (p == 1 && s >= 0 && s <= 7) {
                            constant = true;
                        }
                    }

                    if (memcmp(id.c_str(), env, std::min(id_size,
                            sizeof(env) - 1u)) == 0) {
                        int s;
                        int p = sscanf(id.c_str() + sizeof(env) - 1u,
                            "%d", &s);
                        if (p == 1 && s >= 0 && s <= 31) {
                            constant = true;
                        }
                    }
                }

                if (constant) {
                    if (ni == 1) {
                        return operand_t::make_iconstant(0);
                    } else {
                        /**
                         * The zero registers are necessary as we cannot
                         * express {0, %foo} in PTX.  Instead, we must do
                         * {%zero, %foo}, where %zero is an appropriately sized
                         * register.
                         */
                        switch (width) {
                            case 1:
                                ret.identifier.push_back(__zero_register8);
                                ret.field.push_back(field_none);
                                break;
                            case 2:
                                ret.identifier.push_back(__zero_register16);
                                ret.field.push_back(field_none);
                                break;
                            case 3:
                            case 4:
                                ret.identifier.push_back(__zero_register32);
                                ret.field.push_back(field_none);
                                break;
                            case 5:
                            case 6:
                            case 7:
                            case 8:
                                ret.identifier.push_back(__zero_register64);
                                ret.field.push_back(field_none);
                                break;
                            case 0:
                            default:
                                assert(0 && "Unreachable.");
                                break;
                        }
                    }
                } else {
                    ret.identifier.push_back(make_validity_symbol(id));
                    ret.field.push_back(f);
                }
            }

            return ret;
        case operand_constant:
        case operand_float:
        case operand_double:
            /* Constant values are always defined. */
            return operand_t::make_iconstant(0);
        case invalid_operand:
            assert(0 && "Invalid operand type.");
            return in;
    }

    __builtin_unreachable();
    return ret;
}

global_context_memcheck::global_context_memcheck() :
    state_(new global_memcheck_state()) { }

global_context_memcheck::~global_context_memcheck() {
    /**
     * Cleanup instrumentation metadata.
     */
    for (entry_info_map_t::iterator it = entry_info_.begin();
            it != entry_info_.end(); ++it) {
        delete it->second.inst;
    }
}

cuda_context * global_context_memcheck::factory(int device,
        unsigned int flags) const {
    return new cuda_context_memcheck(
        const_cast<global_context_memcheck *>(this), device, flags);
}

static void analyze_block(size_t * fixed_shared_memory,
        size_t * local_memory, const block_t * block) {
    assert(block);

    const scope_t & scope = *block->scope;
    size_t vn;

    switch (block->block_type) {
        case block_scope:
            /* Scan for shared variables. */
            vn = scope.variables.size();
            for (size_t i = 0; i < vn; i++) {
                const variable_t & v = scope.variables[i];
                if (v.space == shared_space && !(v.array_flexible)) {
                    *fixed_shared_memory += v.size();
                } else if (v.space == local_space) {
                    *local_memory += v.size();
                }
            }

            /* Recurse. */
            for (scope_t::block_vt::const_iterator it = scope.blocks.begin();
                    it != scope.blocks.end(); ++it) {
                analyze_block(fixed_shared_memory, local_memory, *it);
            }

            break;
        case block_statement:
        case block_label:
        case block_invalid:
            /* Do nothing */
            break;
    }
}

void global_context_memcheck::analyze_entry(function_t * entry) {
    if (entry->linkage == linkage_extern) {
        /* Add to list of external entries. */
        external_entries_.insert(entry->entry_name);
        return;
    }

    if (entry->has_return_value) {
        assert(0 && "Return values are not supported.");
        return;
    }

    if (!(entry->entry)) {
        nonentries_.insert(entry->entry_name);
        return;
    }

    /* Determine size of parameter block. */
    size_t offset = 0;
    const size_t pcount = entry->params.size();
    for (size_t i = 0; i < pcount; i++) {
        const size_t align = std::max(entry->params[i].has_align ?
            entry->params[i].alignment : 1u, entry->params[i].size());
        /* Align offset so far. */
        offset = (offset + align - 1u) & ~(align - 1u);
        /* Add size of parameter. */
        offset += entry->params[i].size();
    }

    /* Note parameter block information. */
    entry_info_t e;
    e.user_params     = pcount;
    e.user_param_size = offset;

    analyze_block(&e.fixed_shared_memory, &e.local_memory, &entry->scope);
    entry_info_.insert(entry_info_map_t::value_type(entry->entry_name, e));
}

void global_context_memcheck::instrument_entry(function_t * entry) {
    if (entry->linkage == linkage_extern) {
        return;
    }

    if (!(entry->entry)) {
        assert(0 && "Non-entry methods not supported.");
        return;
    }

    if (entry->no_body) {
        return;
    }

    /* Instrument parameters. */
    const size_t pcount = entry->params.size();
    size_t metadata = 0;

    { /* Dynamic shared memory size */
        param_t ss;
        ss.space = param_space;
        ss.type  = upointer_type();
        ss.name  = __param_shared;

        entry->params.push_back(ss);
        metadata += ss.size();
    }

    { /* Error counter. */
        param_t ec;
        ec.space = param_space;
        ec.type  = pointer_type();
        ec.is_ptr = true;
        ec.name   = __param_error_count;

        entry->params.push_back(ec);
        metadata += ec.size();
    }

    { /* Error output buffer. */
        param_t eb;
        eb.space = param_space;
        eb.type  = pointer_type();
        eb.is_ptr = true;
        eb.name   = __param_error;

        entry->params.push_back(eb);
        metadata += eb.size();
    }

    assert(metadata == metadata_size);

    /* Validity bits for parameters. */
    for (size_t i = 0; i < pcount; i++) {
        param_t vp = entry->params[i];
        switch (vp.type) {
            case b8_type:
            case s8_type:
            case u8_type:
                vp.type = b8_type;
                break;
            case b16_type:
            case f16_type:
            case s16_type:
            case u16_type:
                vp.type = b16_type;
                break;
            case b32_type:
            case f32_type:
            case s32_type:
            case u32_type:
                vp.type = b32_type;
                break;
            case b64_type:
            case f64_type:
            case s64_type:
            case u64_type:
                vp.type = b64_type;
                break;
            case pred_type:
            case texref_type:
            case invalid_type:
                assert(0 && "Unsupported type.");
                break;
        }
        vp.name = make_validity_symbol(vp.name);
        entry->params.push_back(vp);
    }

    internal::instrumentation_t * inst = new
        internal::instrumentation_t();

    entry_info_map_t::iterator it = entry_info_.find(entry->entry_name);
    if (it == entry_info_.end()) {
        assert(0 && "The impossible happened.");
        delete inst;
        return;
    }

    instrument_block(&entry->scope, inst, it->second);

    /* Add entry-wide error count initialization after instrumentation
     * pass */
    {
        variable_t errors;
        errors.space = reg_space;
        errors.type  = u32_type;
        errors.name  = __errors_register;
        assert(entry->scope.block_type == block_scope);
        entry->scope.scope->variables.push_back(errors);

        variable_t has_errors;
        has_errors.space = reg_space;
        has_errors.type  = pred_type;
        has_errors.name  = __has_errors;
        entry->scope.scope->variables.push_back(has_errors);

        variable_t global_errors;
        global_errors.space = reg_space;
        global_errors.type  = u32_type;
        global_errors.name  = __reg_global_errors;
        entry->scope.scope->variables.push_back(global_errors);

        variable_t global_errorsw;
        global_errorsw.space = reg_space;
        global_errorsw.type  = pointer_type();
        global_errorsw.name  = __reg_global_errorsw;
        entry->scope.scope->variables.push_back(global_errorsw);

        variable_t error_count;
        error_count.space = reg_space;
        error_count.type  = pointer_type();
        error_count.name  = __reg_error_count;
        entry->scope.scope->variables.push_back(error_count);

        variable_t error_address;
        error_address.space = reg_space;
        error_address.type  = pointer_type();
        error_address.name  = __errors_address;
        entry->scope.scope->variables.push_back(error_address);

        /* Code to initialize errors. */
        {
        block_t * b = new block_t();
        b->block_type   = block_statement;
        b->parent       = &entry->scope;
        b->statement    = new statement_t(make_mov(u32_type,
            operand_t::make_identifier(__errors_register), 0));

        entry->scope.scope->blocks.push_front(b);
        }

        /* Initialize error_count. */
        {
        block_t * b = new block_t();
        b->block_type   = block_statement;
        b->parent       = &entry->scope;
        b->statement    =
            new statement_t(make_ld(pointer_type(), param_space,
                operand_t::make_identifier(__reg_error_count),
                operand_t::make_identifier(__param_error_count)));

        entry->scope.scope->blocks.push_front(b);
        }

        /* Initialize error_address. */
        {
        block_t * b = new block_t();
        b->block_type   = block_statement;
        b->parent       = &entry->scope;
        b->statement    =
            new statement_t(make_ld(pointer_type(), param_space,
                operand_t::make_identifier(__errors_address),
                operand_t::make_identifier(__param_error)));

        entry->scope.scope->blocks.push_front(b);
        }

        /* Load shared memory size. */
        {
            scope_t & scope = *entry->scope.scope;
            const type_t uptr = upointer_type();

            variable_t ss;
            ss.space = reg_space;
            ss.type  = uptr;
            ss.name  = __shared_reg;
            scope.variables.push_back(ss);

            const operand_t shared_reg =
                operand_t::make_identifier(__shared_reg);
            const operand_t shared_param =
                operand_t::make_identifier(__param_shared);

            /* We have to do this out of order because we are pushing things
             * onto the *front* of the instruction list. */
            block_t * b;
            if (it->second.fixed_shared_memory > 0) {
                b = new block_t();
                b->block_type   = block_statement;
                b->parent       = &entry->scope;
                b->statement    = new statement_t(
                    make_add(uptr, shared_reg, shared_reg,
                        (int) it->second.fixed_shared_memory));
                scope.blocks.push_front(b);
            }

            b = new block_t();
            b->block_type   = block_statement;
            b->parent       = &entry->scope;
            b->statement    = new statement_t(
                make_ld(uptr, param_space, shared_reg, shared_param));
            scope.blocks.push_front(b);
        }

        if (it->second.local_memory > 0) {
            /* Load local memory size. */
            scope_t & scope = *entry->scope.scope;
            const type_t uptr = upointer_type();

            variable_t ss;
            ss.space = reg_space;
            ss.type  = uptr;
            ss.name  = __local_reg;
            scope.variables.push_back(ss);

            const operand_t local_reg =
                operand_t::make_identifier(__local_reg);

            block_t * b = new block_t();
            b->block_type   = block_statement;
            b->parent       = &entry->scope;
            b->statement    = new statement_t(
                make_mov(uptr, local_reg, (int) it->second.local_memory));
            scope.blocks.push_front(b);
        }

        /* Initialize carry flag. */
        {
            scope_t & scope = *entry->scope.scope;

            variable_t vc;
            vc.space = reg_space;
            vc.type  = u32_type;
            vc.name  = __vcarry;

            scope.variables.push_back(vc);

            const operand_t vc_reg =
                operand_t::make_identifier(__vcarry);

            block_t * b = new block_t();
            b->block_type   = block_statement;
            b->parent       = &entry->scope;
            b->statement    = new statement_t(make_mov(u32_type, vc_reg, 0));
            scope.blocks.push_front(b);
        }

        /* Initialize zero registers. */
        {
            type_t       types[] = {u8_type, u16_type, u32_type, u64_type};
            const char * names[] = {__zero_register8, __zero_register16,
                                    __zero_register32, __zero_register64};

            scope_t & scope = *entry->scope.scope;
            for (int i = 0; i < 4; i++) {
                const type_t type = types[i];
                const char * name = names[i];

                variable_t v;
                v.space = reg_space;
                v.type  = type;
                v.name  = name;

                scope.variables.push_back(v);

                const operand_t vreg = operand_t::make_identifier(name);

                block_t * b = new block_t();
                b->block_type   = block_statement;
                b->parent       = &entry->scope;

                const statement_t s = type == u8_type ?
                    make_cvt(u8_type, u16_type, vreg,
                        operand_t::make_iconstant(0)) :
                    make_mov(type, vreg, 0);
                b->statement    = new statement_t(s);
                scope.blocks.push_front(b);
            }
        }
    }

    /* Copy instrumented function. */
    it->second.function = entry;
    it->second.inst = inst;
}

void global_context_memcheck::instrument(
        void **fatCubinHandle, ptx_t * target) {
    assert(target);

    /* We need sm_11 for atomic instructions. */
    if (target->sm == SM10) {
        target->sm = SM11;
    }

    typedef std::vector<variable_t> variable_vt;
    variable_vt new_globals;

    const size_t tcount = target->textures.size();
    for (size_t i = 0; i < tcount; i++) {
        const size_t ncount = target->textures[i].names.size();
        for (size_t j = 0; j < ncount; j++) {
            variable_t tv;
            tv.space = const_space;
            tv.type  = pointer_type();
            tv.name = __texture_prefix + target->textures[i].names[j];
            tv.has_initializer = true;
            tv.initializer_vector = false;

            variant_t v;
            v.type   = variant_integer;
            v.data.u = 0;

            tv.initializer.push_back(v);
            new_globals.push_back(tv);
        }
    }

    {
        variable_t m;
        m.space     = const_space;
        m.type      = pointer_type();
        m.name      = __master_symbol;
        new_globals.push_back(m);
    }

    { // global_ro
        variable_t m;
        m.space     = global_space;
        m.type      = b64_type;
        m.name      = __global_ro;
        m.is_array  = true;
        m.array_dimensions = 1;
        m.array_size[0] = 2;

        variant_t zero;
        zero.type = variant_integer;
        zero.data.u = 0;
        m.has_initializer = true;
        m.initializer_vector = true;
        m.initializer.push_back(zero);
        m.initializer.push_back(zero);

        new_globals.push_back(m);
    }

    { // global_wo
        variable_t m;
        m.space     = global_space;
        m.type      = pointer_type();
        m.name      = __global_wo;
        m.is_array  = true;
        m.array_dimensions = 1;
        m.array_size[0] = 2;
        new_globals.push_back(m);
    }

    const size_t vcount = target->variables.size();
    for (size_t i = 0; i < vcount; i++) {
        variable_handle_t h(fatCubinHandle, target->variables[i].name);
        variable_data_t d;
        d.ptx            = target->variables[i];
        d.hostVar        = NULL;
        variable_definitions_.insert(variable_definition_map_t::value_type(
            h, d));
    }

    const size_t entries = target->entries.size();
    for (size_t i = 0; i < entries; i++) {
        analyze_entry(target->entries[i]);
    }

    for (size_t i = 0; i < entries; i++) {
        instrument_entry(target->entries[i]);
    }

    target->variables.insert(target->variables.end(),
        new_globals.begin(), new_globals.end());
}

using panoptes::internal::temp_operand;
using panoptes::internal::temp_ptr;

void global_context_memcheck::instrument_abs(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    const size_t width  = sizeof_type(statement.type);

    const type_t stype  = signed_type(statement.type);
    const type_t btype  = bitwise_type(statement.type);

    const operand_t & d = statement.operands[0];
    const operand_t & a = statement.operands[1];

    const operand_t vd = make_validity_operand(d, 0);
    const operand_t va = make_validity_operand(a, 0);

    switch (statement.type) {
        case s16_type:
        case s32_type:
        case s64_type: {
            /**
             * Per http://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs,
             * we can compute abs(x) by:
             *
             * const int mask = x >> (sizeof(x) * CHAR_BIT - 1)
             * abs(x) = (x + mask) ^ mask
             *
             * Since we are shifting in the validity bits of the sign, we have
             * two cases for mask (assuming int32_t): 0xFFFFFFFF and
             * 0x00000000.  xor with an all invalid bits value produces a
             * completely invalid result, so we can simplify our validity
             * tracking to
             *
             * vabs(x) = vx | (vx >> (sizeof(vx) * CHAR_BIT - 1))
             */
            const temp_operand tmp(*auxillary, stype);
            aux->push_back(make_shr(stype, tmp, va,
                (int) width * CHAR_BIT - 1));
            aux->push_back(make_or(btype, vd, tmp, va));

            break; }
        case f32_type:
        case f64_type:
            /**
             * TODO We are ignoring ftz for now.
             *
             * The sign bit will always be cleared.
             */
            aux->push_back(make_and(btype, vd, va,
                (1 << (width * CHAR_BIT - 1)) - 1));
            break;
        case s8_type:
        case b8_type:
        case u8_type:
        case b16_type:
        case f16_type:
        case u16_type:
        case b32_type:
        case u32_type:
        case b64_type:
        case u64_type:
        case pred_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            break;
    }

    *keep = true;
}

void global_context_memcheck::instrument_add(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 3u);
    const operand_t & a = statement.operands[1];
    const operand_t & b = statement.operands[2];
    const operand_t & d = statement.operands[0];

    const operand_t & vd = make_validity_operand(d, 0);

    /**
     * mul24 instructions with types other than u32 and s32 are not
     * supported by PTX.
     */
    assert(statement.op != op_mul24 ||
        (statement.type == s32_type ||
         statement.type == u32_type));

    const operand_t va = make_validity_operand(a, 0);
    const operand_t vb = make_validity_operand(b, 0);

    const bool carry_in = (statement.op == op_addc) ||
        (statement.op == op_subc);

    const size_t width  = sizeof_type(statement.type);
    const type_t stype  = signed_of(width);
    const type_t btype  = bitwise_of(width);

    const operand_t vc = operand_t::make_identifier(__vcarry);

    switch (statement.type) {
        case f32_type:
        case f64_type:
            assert(!(carry_in));

            /**
             * If any bits are invalid, assume all are invalid.
             */
            if (va.is_constant() && vb.is_constant()) {
                aux->push_back(make_mov(btype, vd, va));
            } else if (va.is_constant() && !(vb.is_constant())) {
                const temp_operand tmp(*auxillary, stype);
                aux->push_back(make_cnot(btype, tmp, vb));
                aux->push_back(make_sub(stype, vd, tmp, 1));
            } else if (!(va.is_constant()) && vb.is_constant()) {
                const temp_operand tmp(*auxillary, stype);
                aux->push_back(make_cnot(btype, tmp, va));
                aux->push_back(make_sub(stype, vd, tmp, 1));
            } else {
                const temp_operand tmp(*auxillary, stype);
                aux->push_back(make_or(btype, tmp, va, vb));
                aux->push_back(make_cnot(btype, tmp, tmp));
                aux->push_back(make_sub(stype, vd, tmp, 1));
            }

            break;
        case s16_type:
        case s32_type:
        case s64_type:
        case u16_type:
        case u32_type:
        case u64_type: {
            /**
             * Per "Using Memcheck" (2005), undefinedness is
             * computed by OR'ing and then propagating to the left.
             */
            assert(!(carry_in) || width == 4u);
            std::vector<operand_t> vin;
            if (carry_in) {
                vin.push_back(vc);
            }

            if (!(va.is_constant())) {
                vin.push_back(va);
            }

            if (!(vb.is_constant())) {
                vin.push_back(vb);
            }

            bool constant = false;

            const size_t vsize = vin.size();
            if (vsize == 0) {
                aux->push_back(make_mov(btype, vd, 0));
                constant = true;
            } else if (vsize == 1 && carry_in) {
                /* Copy validity bits of carry flag. */
                aux->push_back(make_mov(btype, vd, vc));
                break;
            } else {
                operand_t blended;

                const temp_operand tmp(*auxillary, stype);

                if (vsize == 1u) {
                    blended = vin[0];
                } else {
                    assert(vsize >= 2u);
                    const temp_operand u(*auxillary, stype);

                    /* Blend inputs together. */
                    blended = u;
                    aux->push_back(make_or(btype, u, vin[0], vin[1]));
                    for (size_t i = 2; i < vsize; i++) {
                        aux->push_back(make_or(btype, u, u, vin[i]));
                    }
                }

                aux->push_back(make_neg(stype, tmp, blended));

                if (statement.op == op_mul &&
                        statement.width == width_wide) {
                    /**
                     * Sign extend result.
                     */
                    const type_t swtype  = signed_of(width * 2u);
                    aux->push_back(make_or(btype, tmp, tmp, blended));
                    aux->push_back(make_cvt(swtype, stype, vd, tmp));
                } else {
                    aux->push_back(make_or(btype, vd, tmp, blended));
                }
            }

            /* If instructed to carry-out, store the validity bits
             * of the carry flag by propagating the high bit. */
            if (statement.carry_out) {
                assert(statement.type == u32_type ||
                       statement.type == s32_type);
                assert(statement.op   == op_add  ||
                       statement.op   == op_addc ||
                       statement.op   == op_sub  ||
                       statement.op   == op_subc);

                if (constant) {
                    /* Carry-out is wholly valid. */
                    aux->push_back(make_mov(btype, vc, 0));
                } else {
                    aux->push_back(make_shr(stype, vc, vd, 31));
                }
            }

            break; }
        case s8_type:
        case b8_type:
        case u8_type:
        case b16_type:
        case f16_type:
        case b32_type:
        case b64_type:
        case pred_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            break;
    }

    *keep = true;
}

void global_context_memcheck::instrument_and(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 3u);
    const operand_t & a = statement.operands[1];
    const operand_t & b = statement.operands[2];
    const operand_t & d = statement.operands[0];

    const operand_t va = make_validity_operand(a, 0);
    const operand_t vb = make_validity_operand(b, 0);
    const operand_t vd = make_validity_operand(d, 0);

    const type_t type   = statement.type;
    const type_t btype  = type == pred_type ? b16_type : bitwise_type(type);

    switch (type) {
        case pred_type:
            /**
             * For simplicity, we assume the worst case and OR the validity
             * bits without regard for the underlying data.
             */
            if (va.is_constant() && vb.is_constant()) {
                aux->push_back(make_mov(b16_type, vd, va));
            } else if (va.is_constant() && !(vb.is_constant())) {
                aux->push_back(make_mov(b16_type, vd, vb));
            } else if (!(va.is_constant()) && vb.is_constant()) {
                aux->push_back(make_mov(b16_type, vd, va));
            } else {
                aux->push_back(make_or(b16_type, vd, va, vb));
            }

            assert(d.identifier.size() > 0);
            auxillary->inst->unchecked.insert(d.identifier[0]);
            break;
        case b16_type:
        case b32_type:
        case b64_type:
            if (va.is_constant() && vb.is_constant()) {
                aux->push_back(make_mov(btype, vd, va));
            } else if (va.is_constant() && !(vb.is_constant())) {
                const temp_operand tmp(*auxillary, btype);
                const temp_operand tmp2(*auxillary, btype);

                /**
                 * vd = AND(vb, NOR(b, vb))
                 */
                aux->push_back(make_or(btype, tmp, b, vb));
                aux->push_back(make_not(btype, tmp2, tmp));
                aux->push_back(make_and(btype, vd, tmp, vb));
            } else if (!(va.is_constant()) && vb.is_constant()) {
                const temp_operand tmp(*auxillary, btype);
                const temp_operand tmp2(*auxillary, btype);

                /**
                 * vd = AND(va, NOR(a, va))
                 */
                aux->push_back(make_or(btype, tmp, a, va));
                aux->push_back(make_not(btype, tmp2, tmp));
                aux->push_back(make_and(btype, vd, tmp, va));
            } else {
                const temp_operand tmp0(*auxillary, btype);
                const temp_operand tmp1(*auxillary, btype);
                const temp_operand tmp2(*auxillary, btype);

                aux->push_back(make_or(btype, tmp0, a, va));
                aux->push_back(make_not(btype, tmp1, tmp0));
                aux->push_back(make_or(btype, tmp0, b, vb));
                aux->push_back(make_not(btype, tmp2, tmp0));
                aux->push_back(make_and(btype, tmp0, tmp1, tmp2));
                aux->push_back(make_or(btype, tmp1, va, vb));
                aux->push_back(make_and(btype, vd, tmp0, tmp1));
            }

            break;
        case f32_type:
        case f64_type:
        case s16_type:
        case s32_type:
        case s64_type:
        case u16_type:
        case u32_type:
        case u64_type:
        case s8_type:
        case b8_type:
        case u8_type:
        case f16_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            break;
    }

    *keep = true;
}

/**
 * Searches for the variable in the block (and its parents), returning true
 * if the variable is found.
 *
 * If not NULL, f_ returns a pointer to the containing function.
 * If not NULL, v_ returns a pointer to the variable.
 */
static bool find_variable(const std::string & id, space_t space,
        const block_t * block, const function_t ** f_,
        const variable_t ** v_) {
    bool found            = false;
    const function_t * f  = NULL;
    const variable_t * vr = NULL;

    const size_t isize = id.size();
    while (block && !(found)) {
        assert(!(block->parent) || !(block->fparent));
        assert(block->block_type == block_scope);
        const scope_t * s = block->scope;

        const size_t vn = s->variables.size();
        for (size_t vi = 0; vi < vn; vi++) {
            const variable_t & v = s->variables[vi];
            if (v.space != space) {
                continue;
            }

            const size_t vsize = v.name.size();
            const size_t msize = std::min(vsize, isize);
            if (v.has_suffix && memcmp(v.name.c_str(), id.c_str(), msize) == 0) {
                /* Matched the prefix, now parse the suffix.*/
                int suffix;
                int p = sscanf(id.c_str() + msize, "%d", &suffix);
                if (p == 1 && suffix >= 0 && suffix < v.suffix) {
                    found = true;
                    vr = &v;
                }
            } else if (!(v.has_suffix) && id == v.name) {
                found = true;
                vr = &v;
                break;
            }

            /* Consistency check. */
            assert(!(v.has_suffix ^ (v.suffix >= 0)));
        }

        /* Move up. */
        f     = block->fparent;
        block = block->parent;
    }

    if (f_) {
        *f_ = f;
    }

    if (found && v_) {
        *v_ = vr;
    }

    return found;
}

/**
 * Like find_variable but conducts its search in the global scope.
 */
static bool find_variable_global(const std::string & id, space_t space,
        const ptx_t * ptx, const variable_t ** v_) {
    bool found = false;
    const variable_t * vr = NULL;

    const size_t vn = ptx->variables.size();
    const size_t isize = id.size();
    for (size_t vi = 0; vi < vn; vi++) {
        const variable_t & v = ptx->variables[vi];
        if (v.space != space) {
            continue;
        }

        const size_t vsize = v.name.size();
        const size_t msize = std::min(vsize, isize);
        if (v.has_suffix && memcmp(v.name.c_str(), id.c_str(), msize) == 0) {
            /* Matched the prefix, now parse the suffix.*/
            int s;
            int p = sscanf(id.c_str() + msize, "%d", &s);
            if (p == 1 && s >= 0 && s <= v.suffix) {
                found = true;
                vr = &v;
            }
        } else if (!(v.has_suffix) && id == v.name) {
            found = true;
            vr = &v;
            break;
        }
    }

    if (found && v_) {
        *v_ = vr;
    }

    return found;
}

void global_context_memcheck::instrument_atom(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert((statement.op == op_atom && (statement.operands.size() >= 3u ||
                                        statement.operands.size() <= 4u)) ||
           (statement.op == op_red  && (statement.operands.size() >= 2u ||
                                        statement.operands.size() <= 3u)));

    const bool reduce = statement.op == op_red;
    const bool cas    = statement.atomic_op == atom_cas;
    /**
     * TODO:  Examine validity bits of addr.
     */
    const operand_t & addr = statement.operands[!reduce];
    assert(addr.identifier.size() == 1u);

    const operand_t & b    = statement.operands[1 + !reduce];
    const operand_t & c    = statement.operands[1 + !reduce + cas];

    const operand_t vb     = make_validity_operand(b, 0);
    const operand_t vc     = make_validity_operand(c, 0);

    const operand_t vd     = reduce ? operand_t() :
        make_validity_operand(statement.operands[0], 0);

    /**
     * Atomics do not support vectors.
     */
    assert(statement.vector == v1);
    const size_t width = sizeof_type(statement.type);

    operand_t new_addr, new_vaddr;

    /**
     * Compute the vbits that we are mixing in.
     */
    const type_t btype = bitwise_type(statement.type);
    const type_t stype = signed_type(statement.type);
    const type_t ptr_t = pointer_type();
    const type_t uptr_t = upointer_type();
    const temp_operand mix_in(*auxillary, btype);

    const operand_t local_errors =
        operand_t::make_identifier(__errors_register);

    switch(statement.atomic_op) {
        case atom_and:
        case atom_or:
        case atom_xor:
        case atom_exch:
            /**
             * Invalid bits do not spread.
             */
            assert(b == c);
            aux->push_back(make_mov(btype, mix_in, vb));
            break;
        case atom_inc:
        case atom_dec:
        case atom_add:
        case atom_min:
        case atom_max:
            /**
             * Invalid bits spread.
             */
            assert(b == c);
            if (vb.is_constant()) {
                aux->push_back(make_mov(btype, mix_in, vb));
            } else {
                aux->push_back(make_cnot(btype, mix_in, vb));
                aux->push_back(make_sub(stype, mix_in, mix_in, 1));
            }
            break;
        case atom_cas:
            /**
             * Invalid bits of b spread, invalid bits of c do not.
             */
            assert(b != c);

            if (vb.is_constant()) {
                if (vc.is_constant()) {
                    aux->push_back(make_mov(btype, mix_in, vb));
                } else {
                    aux->push_back(make_mov(btype, mix_in, vc));
                }
            } else {
                aux->push_back(make_cnot(btype, mix_in, vb));
                aux->push_back(make_sub(stype, mix_in, mix_in, 1));

                if (!(vc.is_constant())) {
                    aux->push_back(make_or(btype, mix_in, mix_in, vc));
                }
            }
            break;
        case atom_invalid:
            assert(0 && "Invalid atomic operation.");
            return;
    }

    /**
     * TODO:  Check alignment.
     */

    /**
     * TODO:  Handle predicated atomics.
     */
    assert(!(statement.has_predicate));

    /* Align address. */
    const intptr_t ialignmask = -((intptr_t) width - 1);
    const operand_t oalignmask = operand_t::make_iconstant(ialignmask);

    /* Constants for global bounds. */
    const size_t chunk_size = 1u << lg_chunk_bytes;
    const size_t chunk_mask = (1u << (lg_max_memory - lg_chunk_bytes)) - 1u;

    /**
     * chunk *** ptr = __master;
     */
    const operand_t master = operand_t::make_identifier(__master_symbol);

    typedef internal::instrumentation_t inst_t;
    if (statement.space == shared_space) {
        assert(addr.identifier.size() == 1u);
        const std::string & id = addr.identifier[0];

        /* Walk up scopes for identifiers. */
        const function_t * f = NULL;
        const variable_t * v = NULL;
        bool found = find_variable(id, shared_space, auxillary->block, &f, &v);

        if (!(found)) {
            assert(f);
            found = find_variable_global(id, shared_space, f->parent, &v);
        }

        const size_t size = found ? v->size() : 0u;
        if (found && !(v->array_flexible)) {
            /* We found a fixed symbol, verify we do not statically overrun
             * it. */
            const size_t end = width + (size_t) addr.offset;
            if (end > size) {
                std::stringstream ss;
                ss << statement;

                char msg[256];
                int ret = snprintf(msg, sizeof(msg),
                    "Shared atomic of %zu bytes at offset "
                    "%ld will overrun buffer:\n"
                    "Disassembly: %s\n", width,
                    addr.offset, ss.str().c_str());

                assert(ret < (int) sizeof(msg) - 1);
                logger::instance().print(msg);

                /* Cast off bits into the ether and mark result as invalid. */
                if (!(reduce)) {
                    aux->push_back(make_mov(btype, vd, -1));
                }

                return;
            } else {
                /* Map it to the validity symbol, preserving offset. */
                *keep = true;

                new_vaddr   = addr;
                new_vaddr.identifier.clear();
                new_vaddr.identifier.push_back(
                    make_validity_symbol(id));
                new_vaddr.field.clear();
                new_vaddr.field.push_back(field_none);

                statement_t vatomic;
                vatomic.op              = statement.op;
                vatomic.space           = shared_space;
                vatomic.atomic_op       = atom_or;
                vatomic.type            = btype;
                if (!(reduce)) {
                    vatomic.operands.push_back(vd);
                }
                vatomic.operands.push_back(new_vaddr);
                vatomic.operands.push_back(mix_in);
                aux->push_back(vatomic);
            }
        } else {
            /**
             * Verify address against the shared size parameter.  In practice,
             * CUDA can permit a bit of slack in the size of variables.  For
             * the sake of precision (and protecting validity bits), Panoptes
             * is exact.
             */
            const operand_t limit =
                operand_t::make_identifier(__shared_reg);
            const temp_ptr original_ptr(*auxillary);

            assert(addr.identifier.size() == 1u);
            aux->push_back(make_mov(ptr_t, original_ptr,
                operand_t::make_identifier(addr.identifier[0])));
            if (addr.op_type == operand_addressable &&
                    addr.offset != 0) {
                aux->push_back(make_add(uptr_t, original_ptr, original_ptr,
                    addr.offset));
            }

            const temp_operand valid_pred(*auxillary, pred_type);

            /* Check underflow. */
            aux->push_back(make_setp(uptr_t, cmp_ge, valid_pred,
                original_ptr, operand_t::make_iconstant(0)));

            /* Check overflow. */
            {
                const temp_ptr tmp_ptr(*auxillary);

                aux->push_back(make_add(uptr_t, tmp_ptr, original_ptr,
                    (int64_t) width));

                statement_t s;
                s.op    = op_setp;
                s.type  = uptr_t;
                s.cmp = cmp_le;
                s.bool_op = bool_and;
                s.has_ppredicate = true;
                s.ppredicate = valid_pred;
                s.operands.push_back(tmp_ptr);
                s.operands.push_back(limit);
                s.operands.push_back(valid_pred);
                aux->push_back(s);
            }

            statement_t new_atomic = statement;
            new_atomic.has_predicate = true;
            new_atomic.predicate     = valid_pred;
            aux->push_back(new_atomic);

            /**
             * vbits lie limit ahead of the actual address.
             */
            const temp_ptr vbits(*auxillary);
            aux->push_back(make_add(uptr_t, vbits, original_ptr, limit));

            statement_t vatomic;
            vatomic.has_predicate   = true;
            vatomic.predicate       = valid_pred;
            vatomic.op              = statement.op;
            vatomic.space           = shared_space;
            vatomic.atomic_op       = atom_or;
            vatomic.type            = btype;
            if (!(reduce)) {
                vatomic.operands.push_back(vd);
            }
            vatomic.operands.push_back(vbits);
            vatomic.operands.push_back(mix_in);
            aux->push_back(vatomic);

            if (!(reduce)) {
                /* If !(valid_pred), invalidate the destination. */
                aux->push_back(make_mov(btype, vd, -1));
                aux->back().has_predicate   = true;
                aux->back().is_negated      = true;
                aux->back().predicate       = valid_pred;
            }

            inst_t::error_desc_t desc_oob;
            desc_oob.type = inst_t::outofbounds_atomic_shared;
            desc_oob.orig = statement;
            auxillary->inst->errors.push_back(desc_oob);
            const size_t error_oob = auxillary->inst->errors.size();
            const operand_t op_oob =
                operand_t::make_iconstant((int64_t) error_oob);

            aux->push_back(make_selp(u32_type, valid_pred, local_errors,
                local_errors, op_oob));

            *keep = false;
        }

        return;
    } else if (statement.space == global_space) {
        /**
         * chunk ** ptr0 = *ptr0;
         */
        const temp_ptr ptr0(*auxillary);
        aux->push_back(make_ld(uptr_t, const_space, ptr0, master));

        /**
         * uintptr_t ptr1 = addr >> lg_chunk_bytes;
         * ptr1 &= max_chunks;
         */
        const temp_ptr ptr1(*auxillary);
        aux->push_back(make_shr(ptr_t, ptr1, addr,
            operand_t::make_iconstant(lg_chunk_bytes)));
        aux->push_back(make_and(ptr_t, ptr1, ptr1,
            operand_t::make_iconstant(chunk_mask)));

        /**
         * ptr1 *= sizeof(void *);
         */
        if (sizeof(void *) == 8) {
            aux->push_back(make_shl(ptr_t, ptr1, ptr1, 3));
        } else {
            aux->push_back(make_shl(ptr_t, ptr1, ptr1, 2));
        }

        /**
         * ptr0 += ptr1;
         */
        aux->push_back(make_add(uptr_t, ptr0, ptr0, ptr1));

        /**
         * chunk * ptr0 = *ptr0;
         */
        aux->push_back(make_ld(uptr_t, global_space, ptr0, ptr0));

        /*
         * Align:
         * ptr1 = addr & ~(sizeof(void *) - 1)
         *             & ~(chunk_size - 1);
         *
         * ptr2 = ptr0 + offset(vbits) + ptr1
         *
         * Ignore lower bits.
         * ptr1  >>= 3;
         */
        assert(chunk_size >= sizeof(void *));
        aux->push_back(make_and(ptr_t, ptr1, addr, (int64_t) chunk_size - 1));

        const temp_ptr ptr2(*auxillary);
        aux->push_back(make_add(uptr_t, ptr2, ptr0, ptr1));
        aux->push_back(make_add(uptr_t, ptr2, ptr2,
            offsetof(metadata_chunk, v_data)));

        aux->push_back(make_shr(ptr_t, ptr1, ptr1, 3));

        /**
         * uint8_t * ptr0 += ptr1;
         */
        aux->push_back(make_add(uptr_t, ptr0, ptr0, ptr1));

        /**
         * shift_ptr = ptr0 & 0x1;
         */
        const temp_ptr shift_ptr(*auxillary);
        aux->push_back(make_and(ptr_t, shift_ptr, ptr0, 0x1));

        /**
         * uint32_t shift_ptr32 = shift_ptr;
         */
        const operand_t shift_ptr32 =
            (sizeof(void *) == 4) ? shift_ptr :
            (operand_t) temp_operand(*auxillary, u32_type);
        if (sizeof(void *) == 8) {
            /* Convert. */
            assert(shift_ptr32 != shift_ptr);
            aux->push_back(make_cvt(u32_type, uptr_t, shift_ptr32, shift_ptr));
        }

        /**
         * shift_ptr32 *= width
         */
        {
            statement_t mul;
            mul.op = op_mul;
            mul.type = u32_type;
            mul.width = width_lo;
            mul.operands.push_back(shift_ptr32);
            mul.operands.push_back(shift_ptr32);
            mul.operands.push_back(operand_t::make_iconstant((int64_t) width));
            aux->push_back(mul);
        }

        /**
         * ptr0 &= ~0x1;
         */
        aux->push_back(make_and(ptr_t, shift_ptr, ptr0, ~((int64_t) 1)));

        /**
         * uint16_t tmp = *ptr;
         */
        const temp_operand tmp(*auxillary, u16_type);
        aux->push_back(make_ld(u16_type, global_space, tmp, ptr0));

        /**
         * Shift tmp accordingly, then mask.
         */
        aux->push_back(make_shr(u16_type, tmp, tmp, shift_ptr32));
        aux->push_back(make_and(b16_type, tmp, tmp, (1 << width) - 1));

        const temp_operand valid_atomic(*auxillary, pred_type);

        aux->push_back(make_setp(u16_type, cmp_eq, valid_atomic, tmp,
            operand_t::make_iconstant((1 << width) - 1)));

        /**
         * If !(valid_atomic), warn.
         */
        inst_t::error_desc_t desc_oob;
        desc_oob.type = inst_t::outofbounds_atomic_global;
        desc_oob.orig = statement;
        auxillary->inst->errors.push_back(desc_oob);
        const size_t error_oob = auxillary->inst->errors.size();
        const operand_t op_oob =
            operand_t::make_iconstant((int64_t) error_oob);

        aux->push_back(make_selp(u32_type, valid_atomic, local_errors,
            local_errors, op_oob));

        /* Apply atomic operation to global_wo if invalid */
        const operand_t global_wo = operand_t::make_identifier(__global_wo);
        aux->push_back(make_mov(ptr_t, ptr0, global_wo));
        aux->push_back(make_selp(ptr_t, valid_atomic, ptr0, addr, ptr0));

        *keep = false;
        statement_t new_atomic       = statement;
        new_atomic.operands[!reduce] = ptr0;
        aux->push_back(new_atomic);

        /**
         * Apply operation against validity bits.
         */
        statement_t vatomic;
        vatomic.op              = statement.op;
        vatomic.space           = global_space;
        vatomic.atomic_op       = atom_or;
        vatomic.type            = btype;
        if (!(reduce)) {
            vatomic.operands.push_back(vd);
        }
        vatomic.operands.push_back(ptr2);
        vatomic.operands.push_back(mix_in);
        aux->push_back(vatomic);
    } else {
        // Generic addressing
        *keep = false;
        const temp_operand is_good(*auxillary, pred_type);
        aux->push_back(make_mov(pred_type, is_good, 1));
        const temp_ptr vbits(*auxillary);

        const temp_operand is_shared(*auxillary, pred_type);
        aux->push_back(make_isspacep(shared_space, is_shared, addr));

        {
            const operand_t limit =
                operand_t::make_identifier(__shared_reg);
            const temp_ptr shared_ptr(*auxillary);

            {
                statement_t cvta;
                cvta.has_predicate  = true;
                cvta.predicate      = is_shared;
                cvta.op             = op_cvta;
                cvta.is_to          = true;
                cvta.space          = shared_space;
                cvta.type           = uptr_t;
                cvta.operands.push_back(shared_ptr);
                cvta.operands.push_back(addr);
                aux->push_back(cvta);
            }

            const temp_operand shared_invalid(*auxillary, pred_type);

            /* Check underflow. */
            {
                statement_t s;
                s.op             = op_setp;
                s.type           = uptr_t;
                s.cmp            = cmp_gt;
                s.bool_op        = bool_and;
                s.has_ppredicate = true;
                s.ppredicate     = shared_invalid;
                s.operands.push_back(shared_ptr);
                s.operands.push_back(operand_t::make_iconstant(0));
                s.operands.push_back(shared_invalid);
                aux->push_back(s);
            }

            /* Check overflow. */
            {
                const temp_ptr tmp_ptr(*auxillary);

                aux->push_back(make_add(uptr_t, tmp_ptr, addr,
                    (int64_t) width));

                statement_t s;
                s.op                = op_setp;
                s.type              = uptr_t;
                s.cmp               = cmp_le;
                s.bool_op           = bool_or;
                s.has_ppredicate    = true;
                s.ppredicate        = shared_invalid;
                s.operands.push_back(tmp_ptr);
                s.operands.push_back(limit);
                s.operands.push_back(shared_invalid);
                aux->push_back(s);
            }

            /* Merge */
            {
                statement_t s;
                s.has_predicate = true;
                s.predicate     = is_shared;
                s.op            = op_mov;
                s.type          = pred_type;
                s.operands.push_back(is_good);
                s.operands.push_back(shared_invalid);
                aux->push_back(s);
            }

            /* Initialize, possibly to throw away vbits. */
            aux->push_back(make_add(uptr_t, vbits, addr, limit));

            /* Mix in error. */
            {
                inst_t::error_desc_t desc_oob;
                desc_oob.type = inst_t::outofbounds_atomic_shared;
                desc_oob.orig = statement;
                auxillary->inst->errors.push_back(desc_oob);
                const size_t error_oob = auxillary->inst->errors.size();
                const operand_t op_oob =
                    operand_t::make_iconstant((int64_t) error_oob);

                aux->push_back(make_and(pred_type, shared_invalid,
                    shared_invalid, is_shared));
                aux->push_back(make_selp(u32_type, shared_invalid, local_errors,
                    op_oob, local_errors));
            }
        }

        /* Check global. */
        {
            /**
             * chunk ** ptr0 = *ptr0;
             */
            const temp_ptr ptr0(*auxillary);
            aux->push_back(make_ld(uptr_t, const_space, ptr0, master));
            aux->back().has_predicate = true;
            aux->back().is_negated    = true;
            aux->back().predicate     = is_shared;

            /**
             * uintptr_t ptr1 = addr >> lg_chunk_bytes;
             * ptr1 &= max_chunks;
             */
            const temp_ptr ptr1(*auxillary);
            aux->push_back(make_shr(ptr_t, ptr1, addr, lg_chunk_bytes));
            aux->push_back(make_and(ptr_t, ptr1, ptr1, chunk_mask));

            /**
             * ptr1 *= sizeof(void *);
             */
            if (sizeof(void *) == 8) {
                aux->push_back(make_shl(ptr_t, ptr1, ptr1, 3));
            } else {
                aux->push_back(make_shl(ptr_t, ptr1, ptr1, 2));
            }

            /**
             * ptr0 += ptr1;
             */
            aux->push_back(make_add(uptr_t, ptr0, ptr0, ptr1));

            /**
             * chunk * ptr0 = *ptr0;
             */
            aux->push_back(make_ld(uptr_t, global_space, ptr0, ptr0));
            aux->back().has_predicate = true;
            aux->back().is_negated    = true;
            aux->back().predicate     = is_shared;

            /*
             * Align:
             * ptr1 = addr & ~(sizeof(void *) - 1)
             *             & ~(chunk_size - 1);
             *
             * ptr2 = ptr0 + offset(vbits) + ptr1
             *
             * Ignore lower bits.
             * ptr1  >>= 3;
             */
            assert(chunk_size >= sizeof(void *));
            aux->push_back(make_and(ptr_t, ptr1, addr,
                (int64_t) chunk_size - 1));

            const temp_ptr ptr2(*auxillary);
            aux->push_back(make_add(uptr_t, ptr2, ptr0, ptr1));
            aux->push_back(make_add(uptr_t, ptr2, ptr2,
                offsetof(metadata_chunk, v_data)));

            const temp_ptr shift_ptr(*auxillary);
            if (sizeof(void *) == 8) {
                aux->push_back(make_and(ptr_t, shift_ptr, ptr1, 7));
                aux->push_back(make_shr(ptr_t, ptr1, ptr1, 3));
            } else {
                aux->push_back(make_and(ptr_t, shift_ptr, ptr1, 3));
                aux->push_back(make_shr(ptr_t, ptr1, ptr1, 2));
            }

            const temp_operand shift_ptr32(*auxillary, u32_type);
            aux->push_back(make_cvt(u32_type, uptr_t, shift_ptr32, shift_ptr));

            /**
             * uint8_t * ptr0 += ptr1;
             */
            aux->push_back(make_add(uptr_t, ptr0, ptr0, ptr1));

            /**
             * uint16_t tmp = *ptr;
             */
            const temp_operand tmp(*auxillary, u16_type);
            aux->push_back(make_ld(u8_type, global_space, tmp, ptr0));
            aux->back().has_predicate = true;
            aux->back().is_negated    = true;
            aux->back().predicate     = is_shared;

            /**
             * Shift and mask tmp accordingly.
             */
            aux->push_back(make_shr(b16_type, tmp, tmp, shift_ptr32));
            aux->push_back(make_and(b16_type, tmp, tmp, (1 << width) - 1));

            const temp_operand global_valid(*auxillary, pred_type);

            aux->push_back(make_setp(u16_type, cmp_eq, global_valid, tmp,
                operand_t::make_iconstant((1 << width) - 1)));

            /**
             * If !(global_valid || is_shared), warn.
             */
            {
                inst_t::error_desc_t desc_oob;
                desc_oob.type = inst_t::outofbounds_atomic_global;
                desc_oob.orig = statement;
                auxillary->inst->errors.push_back(desc_oob);
                const size_t error_oob = auxillary->inst->errors.size();
                const operand_t op_oob =
                    operand_t::make_iconstant((int64_t) error_oob);

                aux->push_back(make_or(pred_type, global_valid,
                    global_valid, is_shared));
                aux->push_back(make_selp(u32_type, global_valid, local_errors,
                    local_errors, op_oob));
            }

            /**
             * Mix-in global_valid into is_good.
             */
            aux->push_back(make_and(pred_type, is_good, is_good, global_valid));

            /**
             * Mix-in vbits for global
             */
            aux->push_back(make_selp(ptr_t, is_shared, vbits, vbits, ptr2));
        }

        /**
         * Apply atomic operation.
         */
        statement_t new_atomic      = statement;
        new_atomic.has_predicate    = true;
        new_atomic.predicate        = is_good;
        aux->push_back(new_atomic);

        /**
         * Apply operation against validity bits.
         */
        statement_t vatomic;
        vatomic.has_predicate   = true;
        vatomic.predicate       = is_good;
        vatomic.op              = statement.op;
        vatomic.space           = generic_space;
        vatomic.atomic_op       = atom_or;
        vatomic.type            = btype;
        if (!(reduce)) {
            vatomic.operands.push_back(vd);
        }
        vatomic.operands.push_back(vbits);
        vatomic.operands.push_back(mix_in);
        aux->push_back(vatomic);

        if (!(reduce)) {
            /* If !(is_good), invalidate the destination. */
            aux->push_back(make_mov(btype, vd, -1));
            aux->back().has_predicate   = true;
            aux->back().is_negated      = true;
            aux->back().predicate       = is_good;
        }
    }
}

void global_context_memcheck::instrument_bar(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    *keep = true;

    if (statement.barrier == barrier_sync ||
            statement.barrier == barrier_arrive) {
        /* No-op. */
        return;
    } else if (statement.barrier != barrier_reduce) {
        assert(0 && "Invalid barrier type.");
        return;
    }

    assert(statement.barrier == barrier_reduce);
    assert(statement.operands.size() >= 3u);
    assert(statement.operands.size() <= 4u);

    /*
     * The 4 argument variety is prone to problems where b < blockSize,
     * as we cannot use bar.red in succession and necessarily be
     * operating on the same threads.
     */
    const bool four_argument = (statement.operands.size() == 4u);

    const operand_t & d = statement.operands[0];
    const operand_t & a = statement.operands[1];
    /* If !(four_argument), b == c, but we do not access b. */
    const operand_t & b = statement.operands[2];
    const operand_t & c = statement.operands[four_argument ? 3 : 2];

    const operand_t vd  = make_validity_operand(d, 0);
    const operand_t vc  = make_validity_operand(c, 0);

    /* If c is a constant, the destination will be a good value. */
    if (vc.is_constant()) {
        /* popc has a u32 as its destination; all others are pred type. */
        const type_t vd_type =
            (statement.bool_op == bool_popc) ? b32_type : b16_type;
        aux->push_back(make_mov(vd_type, d, 0));
        return;
    }

    /* Temporaries. */
    const temp_operand tmp_pred(*auxillary, pred_type);

    /* Determine if c is valid. */
    aux->push_back(make_setp(u16_type, cmp_eq, tmp_pred, vc, 0));

    /**
     * OR across all vc's.  If any is invalid, the entirety of the
     * popc is invalid.
     */
    statement_t s;
    s.op      = op_bar;
    s.barrier = barrier_reduce;
    s.bool_op = bool_or;
    s.type    = pred_type;
    s.operands.push_back(tmp_pred);
    s.operands.push_back(a);
    if (four_argument) {
        s.operands.push_back(b);
    }
    s.operands.push_back(tmp_pred);
    aux->push_back(s);

    switch (statement.bool_op) {
        case bool_and:
        case bool_or:
            /**
             * If any bit is invalid, the resulting predicate is invalid.
             * We can mix in the actual value of c for an increase in
             * precision.  If any 0 input is valid to AND (for example), the
             * entire result is valid.
             */
            aux->push_back(make_selp(u16_type, tmp_pred, vd,
                operand_t::make_iconstant(0xFFFF),
                operand_t::make_iconstant(0)));
            break;
        case bool_popc:
            if (four_argument) {
                assert(b.is_constant() &&
                    "Number of thread in reduction must be constant.");
                uint32_t mask = static_cast<uint32_t>(b.offset);
                mask |= (mask - 1u);

                aux->push_back(make_selp(b32_type, tmp_pred, vd,
                    operand_t::make_iconstant(mask),
                    operand_t::make_iconstant(0)));
            } else {
                /**
                 * Determine the number of threads in the block and thus the
                 * maximum value of popc.  We cannot load special registers
                 * directly into the operands of mul, so we create several
                 * temporaries.
                 */
                const temp_operand block_size(*auxillary, u32_type);
                const temp_operand x(*auxillary, u32_type);
                const temp_operand y(*auxillary, u32_type);
                const temp_operand z(*auxillary, u32_type);

                aux->push_back(make_mov(u32_type, x,
                    operand_t::make_identifier("%ntid.x")));
                aux->push_back(make_mov(u32_type, y,
                    operand_t::make_identifier("%ntid.y")));
                aux->push_back(make_mov(u32_type, z,
                    operand_t::make_identifier("%ntid.z")));

                aux->push_back(make_mul(u32_type, width_lo, block_size, x, y));
                aux->push_back(make_mul(u32_type, width_lo, block_size,
                    block_size, z));

                /**
                 * Subtract 1 from the block_size to reduce it to all 1's, then
                 * OR this back into the block_size to determine which bits of
                 * the popc result will be invalid (as high bits corresponding
                 * to powers of 2 well in excess of the number of threads will
                 * always be 0).
                 */
                const temp_operand minus_one(*auxillary, u32_type);
                aux->push_back(make_sub(u32_type, minus_one, block_size, 1));
                aux->push_back(make_or(b32_type, block_size, block_size, minus_one));

                /**
                 * Map result of validity reduction to block_size or 0.
                 */
                aux->push_back(make_selp(b32_type, tmp_pred, vd, block_size,
                    operand_t::make_iconstant(0)));
            }

            break;
        case bool_none:
        case bool_xor:
            assert(0 && "Invalid boolean operation.");
            break;
    }
}

void global_context_memcheck::instrument_bfe(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 4u);

    assert(statement.type == u32_type ||
           statement.type == u64_type ||
           statement.type == s32_type ||
           statement.type == s64_type);

    /**
     * If either b or c have any invalid bits above their mask
     * (0xFF), we assume the worst and consider the entire
     * result invalid.
     */
    bool immed_constant;
    operand_t immed;

    std::vector<operand_t> source_validity;
    for (unsigned i = 2; i < 4; i++) {
        const operand_t & op   = statement.operands[i];
        const operand_t vop = make_validity_operand(op, 0);
        if (!(vop.is_constant())) {
            source_validity.push_back(vop);
        }
    }

    const size_t vargs = source_validity.size();
    const type_t btype = bitwise_type(statement.type);
    const type_t stype = signed_type(statement.type);

    const size_t width  = sizeof_type(statement.type);

    const temp_operand tmp0(*auxillary, b32_type);

    const operand_t zero = operand_t::make_iconstant(0);

    assert(vargs <= 2);
    switch (vargs) {
        case 0:
            /**
             * No variable arguments, result is defined.
             */
            immed = zero;
            immed_constant = true;
            break;
        case 1:
            /**
             * Propagate from source.
             */
            aux->push_back(make_and(b32_type, tmp0, source_validity[0], 0xFF));
            aux->push_back(make_cnot(b32_type, tmp0, tmp0));
            aux->push_back(make_sub(s32_type, tmp0, tmp0, 1));

            immed = tmp0;
            immed_constant = false;
            break;
        case 2:
            /**
             * OR validity bits of two results, then propagate.
             */
            aux->push_back(make_or(b32_type, tmp0,
                source_validity[0], source_validity[1]));
            aux->push_back(make_and(b32_type, tmp0, tmp0, 0xFF));
            aux->push_back(make_cnot(b32_type, tmp0, tmp0));
            aux->push_back(make_sub(s32_type, tmp0, tmp0, 1));

            immed = tmp0;
            immed_constant = false;
            break;
    }

    /**
     * Consider the argument to a.
     */
    const operand_t & a  = statement.operands[1];
    const operand_t   va = make_validity_operand(a, 0);
    const bool aconstant = va.is_constant();

    const operand_t & d  = statement.operands[0];
    const operand_t   vd = make_validity_operand(d, 0);

    if (aconstant && immed_constant) {
        /**
         * Result is constant.  We already have initialized immed to zero.
         */
        aux->push_back(make_mov(btype, vd, immed));
        return;
    }

    if (aconstant) {
        /**
         * We've computed all that we need to validity-bitwise in computing
         * immed.
         */
        if (width == 4u) {
            aux->push_back(make_mov(btype, vd, immed));
        } else if (width == 8u) {
            /**
             * Sign extend.
             */
            aux->push_back(make_cvt(stype, s32_type, vd, immed));
        } else {
            assert(0 && "Unsupported width.");
        }

        return;
    }

    if (immed_constant) {
        /**
         * Extract the validity bits using b and c.
         *
         * TODO:  This does not properly take into account the propagation of
         * the sign bit.
         */
        aux->push_back(make_bfe(statement.type, vd, va,
            statement.operands[2], statement.operands[3]));
    } else {
        /**
         * Extract, then OR with worst case validity results due to b and c.
         */
        assert((width == 4u || width == 8u) && "Unsupported width.");

        const temp_operand tmp1(*auxillary, btype);
        const temp_operand tmp2(*auxillary, btype);

        assert(immed != tmp1);

        /**
         * Sign extend.
         */
        if (width == 8u) {
            aux->push_back(make_cvt(s64_type, s32_type, tmp2, immed));
            assert(immed != tmp2);
            immed = tmp2;
        }

        aux->push_back(make_bfe(statement.type, tmp1, va,
            statement.operands[2], statement.operands[3]));
        aux->push_back(make_or(btype, vd, immed, tmp1));
    }

    *keep = true;
}

void global_context_memcheck::instrument_bfi(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 5u);
    assert(statement.type == b32_type || statement.type == b64_type);
    *keep = true;

    /**
     * If either c or d have any invalid bits above their mask (0xFF), we
     * assume the worst and consider the entire result invalid.
     */
    bool immed_constant;
    operand_t immed;

    std::vector<operand_t> source_validity;
    for (unsigned i = 3; i < 5; i++) {
        const operand_t & op   = statement.operands[i];
        const operand_t vop = make_validity_operand(op, 0);
        if (!(vop.is_constant())) {
            source_validity.push_back(vop);
        }
    }

    const size_t vargs = source_validity.size();
    const type_t btype = bitwise_type(statement.type);
    const size_t width  = sizeof_type(statement.type);

    const temp_operand tmp0(*auxillary, b32_type);

    assert(vargs <= 2);
    switch (vargs) {
        case 0:
            /**
             * No variable arguments, result is defined.
             */
            immed = operand_t::make_iconstant(0);
            immed_constant = true;
            break;
        case 1:
            /**
             * Propagate from source.
             */
            aux->push_back(make_and(b32_type, tmp0, source_validity[0], 0xFF));
            aux->push_back(make_cnot(b32_type, tmp0, tmp0));
            aux->push_back(make_sub(s32_type, tmp0, tmp0, 1));

            immed = tmp0;
            immed_constant = false;
            break;
        case 2:
            /**
             * OR validity bits of two results, then propagate.
             */
            aux->push_back(make_or(b32_type, tmp0,
                source_validity[0], source_validity[1]));
            aux->push_back(make_and(b32_type, tmp0, tmp0, 0xFF));
            aux->push_back(make_cnot(b32_type, tmp0, tmp0));
            aux->push_back(make_sub(s32_type, tmp0, tmp0, 1));
            immed = tmp0;
            immed_constant = false;
            break;
    }

    /**
     * Consider the argument to a and b.
     */
    const operand_t & a  = statement.operands[1];
    const operand_t   va = make_validity_operand(a, 0);
    const bool aconstant = va.is_constant();

    const operand_t & b  = statement.operands[2];
    const operand_t   vb = make_validity_operand(b, 0);
    const bool bconstant = vb.is_constant();

    const operand_t & vd = make_validity_operand(statement.operands[0], 0);

    const bool data_constant = aconstant && bconstant;

    if (data_constant && immed_constant) {
        /**
         * Result is constant.  We already have initialized immed
         * to zero.
         */
        aux->push_back(make_mov(btype, vd, immed));
        return;
    }

    if (data_constant) {
        /**
         * We've computed all that we need to validity-bitwise
         * in computing immed.
         *
         * Mark that we are using the tmp variable and sign extend
         * to the result, if necessary.
         */
        if (width == 4u) {
            aux->push_back(make_mov(btype, vd, immed));
        } else if (width == 8u) {
            aux->push_back(make_cvt(s64_type, s32_type, vd, immed));
        } else {
            assert(0 && "Unsupported width.");
        }

        return;
    }

    if (immed_constant) {
        /**
         * Insert the validity bits using c and d.
         */
        aux->push_back(make_bfi(statement.type, vd, va, vb,
            statement.operands[3], statement.operands[4]));
    } else {
        /**
         * Insert, then OR with worst case validity results
         * due to b and c.
         */
        const temp_operand tmp1(*auxillary, btype);
        const temp_operand tmp2(*auxillary, btype);

        assert(immed != tmp1);

        aux->push_back(make_bfi(statement.type, tmp1, va, vb,
            statement.operands[3], statement.operands[4]));

        if (width == 4u) {
            aux->push_back(make_or(btype, vd, immed, tmp1));
        } else if (width == 8u) {
            /* Sign extend immed. */
            assert(tmp2 != immed);

            aux->push_back(make_cvt(s64_type, s32_type, tmp2, immed));
            immed = tmp2;
            aux->push_back(make_or(btype, vd, immed, tmp1));
        } else {
            assert(0 && "Unsupported width.");
        }
    }
}

void global_context_memcheck::instrument_bit1(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);
    *keep = true;
    const type_t btype = bitwise_type(statement.type);

    const operand_t & a = statement.operands[1];
    const operand_t & d = statement.operands[0];

    const operand_t va  = make_validity_operand(a, 0);
    const operand_t vd  = make_validity_operand(d, 0);

    const bool constant = va.is_constant();

    if (constant) {
        /* Result is completely valid. */
        aux->push_back(make_mov(b32_type, vd, va));
        return;
    }

    /**
     * Suppose we compute the popc of the valid data bits, e.g.  AND(d,
     * NOT(v)), and label this result "ret."  The invalid bits make a
     * contribution of 0 to popc(v) to ret to obtain the popc in the result.
     * Per our rules for addition, since the least significant bit is invalid,
     * we must left propagate it as it could carry to more significant bits.
     */
    const temp_operand tmp(*auxillary, btype);
    operand_t ntmp;
    int    mret;

    switch (statement.type) {
        case b32_type:
        case s32_type: /* Due to bfind */
        case u32_type: /* Due to bfind */
            ntmp  = tmp;
            mret  = 32;
            break;
        case b64_type:
        case s64_type: /* Due to bfind */
        case u64_type: /* Due to bfind */
            /* The result must be narrowed. */
            ntmp    = temp_operand(*auxillary, b32_type);

            mret = 64;
            break;
        default:
            __builtin_unreachable();
            assert(0 && "Invalid type.");
            break;
    }

    /**
     * Map 0 -> 1, (everything else) -> 0
     */
    aux->push_back(make_cnot(btype, tmp, va));

    /**
     * Narrow, if applicable.
     */
    const size_t width = sizeof_type(statement.type);
    if (width == 8u) {
        aux->push_back(make_cvt(u32_type, u64_type, ntmp, tmp));
    }

    /**
     * Map 0 -> 0, (everything else) -> 0xFFFFFFFF
     */
    aux->push_back(make_sub(u32_type, ntmp, ntmp, 1));

    bool has_mask;
    int mask;
    switch (statement.op) {
        case op_bfind:
            /**
             * bfind can return a value between 0 and mret - 1 inclusive or it
             * can return 0xFFFFFFFF.
             */
            has_mask = false;
            break;
        case op_clz:
        case op_popc:
            /* Map 0 -> 0 (modulo 2 * mret). */
            has_mask = true;
            mask = 2 * mret - 1;
            break;
        default:
            assert(0 && "Unknown opcode.");
            break;
    }

    if (has_mask) {
        /*
         * The bits above those needed to represent the maximum result will be
         * zero.
         */
        aux->push_back(make_and(b32_type, vd, ntmp, mask));
    }
}

void global_context_memcheck::instrument_brev(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);
    const operand_t & a = statement.operands[1];
    const operand_t & d = statement.operands[0];

    const operand_t va = make_validity_operand(a, 0);
    const operand_t vd = make_validity_operand(d, 0);

    switch (statement.type) {
        case b32_type:
        case b64_type:
            if (va.is_constant()) {
                aux->push_back(make_mov(statement.type, vd, va));
            } else {
                /**
                 * Bitwise reverse the validity bits.
                 */
                aux->push_back(make_brev(statement.type, vd, va));
            }

            break;
        case b16_type:
        case f32_type:
        case f64_type:
        case s16_type:
        case s32_type:
        case s64_type:
        case u16_type:
        case u32_type:
        case u64_type:
        case s8_type:
        case b8_type:
        case u8_type:
        case f16_type:
        case pred_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            break;
    }

    *keep = true;
    (void) auxillary;
}

void global_context_memcheck::instrument_cnot(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    /**
     * We cannot reuse the popc codepath for cnot as it reduces the width of
     * its output to a u32 regardless of the instruction datatype.
     */
    assert(statement.operands.size() == 2u);
    *keep = true;

    const type_t btype = bitwise_type(statement.type);

    const operand_t & a = statement.operands[1];
    const operand_t & d = statement.operands[0];

    const operand_t & vd = make_validity_operand(d, 0);

    const operand_t va = make_validity_operand(a, 0);
    bool constant = va.is_constant();

    if (constant) {
        /* Result is completely valid. */
        aux->push_back(make_mov(btype, vd, va));
        return;
    }

    const temp_operand tmp(*auxillary, btype);

    /**
     * Map 0 -> 1, (everything else) -> 0
     */
    aux->push_back(make_cnot(btype, tmp, va));

    /**
     * Map 0 -> 0, (everything else) -> 1.
     */
    aux->push_back(make_xor(btype, vd, tmp, operand_t::make_iconstant(0x1)));
}

void global_context_memcheck::instrument_copysign(
        const statement_t & statement, statement_vt * aux, bool * keep,
        internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 3u);
    assert(statement.type == f32_type ||
           statement.type == f64_type);

    const operand_t & a = statement.operands[1];
    const operand_t & b = statement.operands[2];
    const operand_t & d = statement.operands[0];

    const operand_t va = make_validity_operand(a, 0);
    const operand_t vb = make_validity_operand(b, 0);
    const operand_t vd = make_validity_operand(d, 0);

    const type_t btype = bitwise_type(statement.type);

    /**
     * bfi and copysign both appear as part of sm_20 so bfi should be fair game
     * to use.
     */
    if (va.is_constant() && vb.is_constant()) {
        aux->push_back(make_mov(btype, vd, va));
    } else {
        /* One or both of the validity values is nonconstant, insert the sign
         * bit into the rest of the data. */
        aux->push_back(make_bfi(btype, vd, va, vb,
            operand_t::make_iconstant(
                (int) sizeof_type(statement.type) * CHAR_BIT - 1),
            operand_t::make_iconstant(1)));
    }

    *keep = true;
    (void) auxillary;
}

void global_context_memcheck::instrument_cvt(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);
    const operand_t & d = statement.operands[0];
    const operand_t & a = statement.operands[1];

    const operand_t vd = make_validity_operand(d, 0);
    const operand_t va = make_validity_operand(a, 0);

    const type_t dtype = statement.type;
    const type_t atype = statement.type2;

    *keep = true;

    if (!(statement.saturating)) {
        bool is_signed_d, is_signed_a;
        bool is_unsigned_d, is_unsigned_a;
        is_signed_d = is_signed_a = false;
        is_unsigned_d = is_unsigned_a = false;

        switch (dtype) {
            case u8_type:
            case u16_type:
            case u32_type:
            case u64_type:
                is_unsigned_d = true;
                break;
            case s8_type:
            case s16_type:
            case s32_type:
            case s64_type:
                is_signed_d = true;
                break;
            default:
                break;
        }

        switch (atype) {
            case u8_type:
            case u16_type:
            case u32_type:
            case u64_type:
                is_unsigned_a = true;
                break;
            case s8_type:
            case s16_type:
            case s32_type:
            case s64_type:
                is_signed_a = true;
                break;
            default:
                break;
        }

        assert(!(is_signed_d && is_unsigned_d));
        assert(!(is_signed_a && is_unsigned_a));
        if (is_signed_d && is_signed_a) {
            /* Sign extend the validity bits. */
            aux->push_back(make_cvt(dtype, atype, vd, va));
            return;
        } else if (is_unsigned_d && is_unsigned_a) {
            /* Zero extend the validity bits. */
            aux->push_back(make_cvt(dtype, atype, vd, va));
            return;
        }
    }

    /* An invalid bit anywhere propagates across the value. */
    const type_t abtype = bitwise_type(atype);
    const type_t astype = signed_type(atype);

    const temp_operand tmp(*auxillary, abtype);
    aux->push_back(make_cnot(abtype, tmp, va));
    aux->push_back(make_sub(astype, tmp, tmp, 1));
    /* Sign extend across target. */
    const type_t dstype = signed_type(dtype);
    aux->push_back(make_cvt(dstype, astype, vd, tmp));
}

void global_context_memcheck::instrument_fp1(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert((statement.type == f32_type ||
            statement.type == f64_type) && "Invalid type.");
    assert(statement.operands.size() == 2u);

    const operand_t & a  = statement.operands[1];
    const operand_t   va = make_validity_operand(a, 0);
    const bool constant  = va.is_constant();

    const operand_t & d  = statement.operands[0];
    const operand_t   vd = make_validity_operand(d, 0);

    const type_t btype   = bitwise_type(statement.type);
    const type_t stype   = signed_type(statement.type);

    if (constant) {
        /* Constant argument, so result is valid. */
        aux->push_back(make_mov(btype, vd, va));
    } else {
        /* Spread invalid bits. */
        const temp_operand tmp(*auxillary, btype);
        aux->push_back(make_cnot(btype, tmp, va));
        aux->push_back(make_sub(stype, vd, tmp, 1));
    }

    *keep = true;
}

void global_context_memcheck::instrument_fp3(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    *keep = true;

    assert((statement.type == f32_type ||
            statement.type == f64_type) && "Invalid type.");
    assert(statement.operands.size() == 4u);

    /**
     * If any bits are invalid, assume the worst.
     */
    std::vector<operand_t> source_validity;
    for (unsigned i = 1; i < 4; i++) {
        const operand_t & op  = statement.operands[i];
        const operand_t   vop = make_validity_operand(op, 0);
        if (!(vop.is_constant())) {
            source_validity.push_back(vop);
        }
    }

    const operand_t & d = statement.operands[0];
    const operand_t vd  = make_validity_operand(d, 0);
    const type_t btype  = bitwise_type(statement.type);
    const type_t stype  = signed_type(statement.type);

    const operand_t zero = operand_t::make_iconstant(0);

    const size_t vargs = source_validity.size();
    if (vargs == 0) {
        /* No nonconstant arguments, so result is valid. */
        aux->push_back(make_mov(btype, vd, 0));
    } else {
        /* One or more nonconstant arguments, fold into a temporary variable,
         * then spread invalid bits. */
        const temp_operand tmp(*auxillary, btype);
        if (vargs == 1u) {
            /* Move directly into temporary variable. */
            aux->push_back(make_mov(btype, tmp, source_validity[0]));
        } else {
            aux->push_back(make_or(btype, tmp,
                source_validity[0], source_validity[1]));
            for (size_t i = 2; i < vargs; i++) {
                aux->push_back(make_or(btype, tmp, tmp, source_validity[i]));
            }
        }

        /* Spread invalid bits. */
        aux->push_back(make_cnot(btype, tmp, tmp));
        aux->push_back(make_sub(stype, vd, tmp, 1));
    }
}

void global_context_memcheck::instrument_isspacep(
        const statement_t & statement, statement_vt * aux, bool * keep,
        internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);

    /* Narrow a pointer type to a b16. */
    const operand_t & p = statement.operands[0];
    const operand_t & a = statement.operands[1];

    const operand_t vp = make_validity_operand(p, 0);
    assert(!(vp.is_constant()));
    const operand_t va = make_validity_operand(a, 0);

    const operand_t zero = operand_t::make_iconstant(0);
    const operand_t one  = operand_t::make_iconstant(1);

    if (va.is_constant()) {
        aux->push_back(make_mov(b16_type, vp, 0));
    } else {
        const temp_ptr tmpptr(*auxillary);
        const temp_operand tmp(*auxillary, b16_type);

        /* Map 0 -> 1, (everything) -> 0 */
        aux->push_back(make_cnot(pointer_type(), tmpptr, va));

        /* Narrow to b16 */
        aux->push_back(make_cvt(u16_type, upointer_type(), tmp, tmpptr));

        /* Map 0 -> 0, (everything) -> 0xFFFF */
        aux->push_back(make_sub(s16_type, vp, tmp, one));
    }

    *keep = true;
}

void global_context_memcheck::instrument_ld(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);
    const operand_t & src = statement.operands[1];
    /**
     * We convert:
     *    dst   = *src;
     *
     * into:
     *    const size_t max_chunks =
     *      (1u << (lg_max_memory - lg_chunk_bytes));
     *    const size_t chunk_size =
     *      (1u << lg_chunk_bytes);
     *          void * global_wo;
     *    const void * global_ro;
     *
     *    uint16_t error = 0;
     *
     *    -- TODO Track the max chidx seen per thread before mask,
     *            warn if exceeded.
     *
     *    size_t chidx = (src >> lg_chunk_bytes) &
     *                   (max_chunks - 1u);
     *    size_t inidx = (src & (chunk_size - 1u)) >> 3u;
     *    const metadata_chunk * chunk = __master_symbol[chidx];
     *
     *    -- This is appropriately sized for vector+type combo
     *    typedef FOO read_t;
     *    read_t  a     = chunk->a_data[inidx];
     *    uint8_t apass = a ? 1 : 0;
     *    uint8_t npass = 1 - apass;
     *    -- BAR is based on the line of PTX
     *    error          = max(error, npass * BAR);
     *    -- There are some type casts here.
     *    void * new_src = src * apass + global_ro * npass;
     *    *dst  = new_src;
     *    *vdst = chunk->vdata[inidx];
     *
     * TODO:  Check alignment.
     */
    const temp_ptr original_ptr(*auxillary);
    const temp_ptr inidx(*auxillary);
    const temp_ptr chunk(*auxillary);
    const temp_ptr chunk_ptr(*auxillary);
    const temp_ptr data_ptr(*auxillary);
    const temp_ptr validity_ptr_src(*auxillary);
    const temp_ptr chidx(*auxillary);
    const temp_ptr global_ro_reg(*auxillary);
    const temp_ptr vidx(*auxillary);

    const operand_t master = operand_t::make_identifier(__master_symbol);
    const operand_t a_data_ptr = operand_t::make_addressable(chunk_ptr,
        offsetof(metadata_chunk, a_data));
    const operand_t validity_ptr = operand_t::make_addressable(validity_ptr_src,
        offsetof(metadata_chunk, v_data));
    const operand_t global_ro = operand_t::make_identifier(__global_ro);

    const temp_operand valid_pred(*auxillary, pred_type);

    const size_t chunk_size = 1u << lg_chunk_bytes;
    const size_t max_chunks = (1u << (lg_max_memory - lg_chunk_bytes)) - 1u;
    const size_t width = sizeof_type(statement.type) *
        (unsigned) statement.vector;
    const type_t read_t = unsigned_of(width);
    const temp_operand a_data(*auxillary, read_t);

    const type_t wread_t = width <= 8 ? u16_type : read_t;
    const temp_operand a_wdata(*auxillary, u16_type);
    const temp_operand a_data32(*auxillary, u32_type);
    const operand_t a_cdata = width <= 8 ? a_wdata : a_data;

    operand_t new_src, new_vsrc;

    unsigned log_ptr;
    switch (sizeof(void *)) {
        case 4u: log_ptr = 2; break;
        case 8u: log_ptr = 3; break;
    }

    bool drop_load = false;
    bool invalidate = false;
    bool validate = false;

    const type_t ptr_t  = pointer_type();
    const type_t uptr_t = upointer_type();

    const operand_t local_errors =
        operand_t::make_identifier(__errors_register);

    typedef internal::instrumentation_t inst_t;
    if (statement.space == param_space) {
        assert(src.identifier.size() == 1u);
        const std::string & id = src.identifier[0];

        /* Verify that this is a parameter to us and not a call-level
         * parameter. */
        const block_t    * b = auxillary->block;
        const function_t * f = NULL;
        while (b) {
            f = b->fparent;
            b = b->parent;
        }
        assert(f);

        bool found = false;
        for (function_t::param_vt::const_iterator jit =
                f->params.begin(); jit != f->params.end(); ++jit) {
            if (jit->name == id) {
                found = true;
            }
        }

        if (found) {
            new_src  = src;
            new_vsrc = operand_t::make_identifier(make_validity_symbol(id));
            if (src.op_type == operand_addressable) {
                new_vsrc.op_type = operand_addressable;
                new_vsrc.offset  = src.offset;
            }
        } else {
            /* Mark as valid.  Keep load as-is. */
            validate = true;
            drop_load = true;
            aux->push_back(statement);
        }
    } else if (statement.space == shared_space) {
        assert(src.identifier.size() == 1u);
        const std::string & id = src.identifier[0];

        /* Walk up scopes for identifiers. */
        const function_t * f = NULL;
        const variable_t * v = NULL;
        bool found = find_variable(id, shared_space, auxillary->block, &f, &v);

        if (!(found)) {
            assert(f);
            found = find_variable_global(id, shared_space, f->parent, &v);
        }

        const bool   flexible = found ? v->array_flexible : false;
        const size_t size     = found ? v->size() : 0;

        if (found && !(flexible)) {
            /* We found a fixed symbol, verify we do not statically overrun
             * it. */
            const size_t end = width + (size_t) src.offset;

            if (src.offset < 0) {
                std::stringstream ss;
                ss << statement;

                char msg[256];
                int ret = snprintf(msg, sizeof(msg),
                    "Shared load of %zu bytes at offset "
                    "%ld will underrun buffer:\n"
                    "Disassembly: %s\n", width, src.offset, ss.str().c_str());

                assert(ret < (int) sizeof(msg) - 1);
                logger::instance().print(msg);

                /* Cast off bits into the ether. */
                drop_load = true;
                invalidate = true;
            } else if (end > size) {
                std::stringstream ss;
                ss << statement;

                char msg[256];
                int ret = snprintf(msg, sizeof(msg),
                    "Shared load of %zu bytes at offset "
                    "%ld will overrun buffer:\n"
                    "Disassembly: %s\n", width, src.offset, ss.str().c_str());

                assert(ret < (int) sizeof(msg) - 1);
                logger::instance().print(msg);

                /* Cast off bits into the ether. */
                drop_load = true;
                invalidate = true;
            } else {
                /* Map it to the validity symbol. */
                new_src  = src;

                new_vsrc = src;
                new_vsrc.identifier.clear();
                new_vsrc.identifier.push_back(make_validity_symbol(id));
                new_vsrc.field.clear();
                new_vsrc.field.push_back(field_none);
            }
        } else {
            /* Verify address against the shared size parameter. TODO:  Verify
             * we don't overrun the buffer. */

            const operand_t limit = operand_t::make_identifier(__shared_reg);

            drop_load = true;
            invalidate = false;

            assert(src.identifier.size() == 1u);
            aux->push_back(make_mov(ptr_t, original_ptr,
                operand_t::make_identifier(src.identifier[0])));
            if (src.op_type == operand_addressable &&
                    src.offset != 0) {
                aux->push_back(make_add(uptr_t, original_ptr, original_ptr,
                    src.offset));
            }

            const temp_operand not_valid_pred(*auxillary, pred_type);

            aux->push_back(make_setp(uptr_t, cmp_ge, valid_pred,
                not_valid_pred, limit, original_ptr));

            statement_t new_load = statement;
            /**
             * TODO:  We need to propagate the predicate flags of predicated
             * loads.
             */
            assert(!(new_load.has_predicate));
            new_load.has_predicate = true;
            new_load.predicate = valid_pred;
            aux->push_back(new_load);

            const temp_ptr vsrc(*auxillary);

            aux->push_back(make_add(uptr_t, vsrc, original_ptr, limit));

            const type_t btype = bitwise_type(statement.type);

            const operand_t vdst = make_validity_operand(
                statement.operands[0], 0);

            statement_t new_vload = statement;
            new_vload.has_predicate = true;
            new_vload.predicate = valid_pred;
            new_vload.type = btype;
            new_vload.operands[0] = vdst;
            new_vload.operands[1] = vsrc;
            aux->push_back(new_vload);

            /* If we don't do the load, invalidate on the not_valid_pred
             * flag. */
            const size_t ni = vdst.identifier.size();
            const operand_t negone = operand_t::make_iconstant(-1);

            /* Flag the error condition if one exists. */
            inst_t::error_desc_t desc_oob;
            desc_oob.type = inst_t::outofbounds_ld_shared;
            desc_oob.orig = statement;
            auxillary->inst->errors.push_back(desc_oob);
            const size_t error_oob = auxillary->inst->errors.size();
            const operand_t op_oob =
                operand_t::make_iconstant((int64_t) error_oob);

            aux->push_back(make_selp(u32_type, not_valid_pred, local_errors,
                op_oob, local_errors));

            /**
             * nvcc does not seem to generate byte-sized registers, so while
             * the load is a byte-wide, we don't get clued in on the size of
             * our target register.  For now, widen the register (and hope this
             * behavior doesn't change).
             */
            const type_t bwtype = (btype == b8_type) ?  b16_type : btype;

            for (size_t i = 0; i < ni; i++) {
                operand_t o;
                o.op_type = vdst.op_type;
                o.identifier.push_back(vdst.identifier[i]);
                o.field.push_back(vdst.field[i]);

                statement_t s;
                s.op = op_mov;
                s.type = bwtype;
                s.has_predicate = true;
                s.predicate = not_valid_pred;
                s.operands.push_back(o);
                s.operands.push_back(negone);
                aux->push_back(s);
            }
        }
    } else if (statement.space == local_space) {
        const operand_t limit = operand_t::make_identifier(__local_reg);

        drop_load = true;
        invalidate = false;

        operand_t origin;

        /* Verify address against the local size parameter. */
        const temp_ptr tmp(*auxillary);
        if (src.is_constant()) {
            origin = src;

            aux->push_back(make_add(uptr_t, tmp, origin, (int) width));
            aux->push_back(make_setp(uptr_t, cmp_ge, valid_pred, limit, tmp));
            /* TODO:  Check for underrun. */
        } else {
            assert(src.identifier.size() == 1u);
            const std::string & id = src.identifier[0];

            aux->push_back(make_mov(ptr_t, original_ptr,
                operand_t::make_identifier(id)));
            if (src.op_type == operand_addressable &&
                    src.offset != 0) {
                aux->push_back(make_add(uptr_t, original_ptr, original_ptr,
                    src.offset));
            }

            origin = original_ptr;

            aux->push_back(make_add(uptr_t, tmp, origin, (int) width));
            aux->push_back(make_setp(uptr_t, cmp_ge, valid_pred,
                operand_t::make_iconstant(0xFFFCA0), tmp));
        }

        statement_t new_load = statement;
        /**
         * TODO:  We need to propagate the predicate flags of predicated loads.
         */
        assert(!(new_load.has_predicate));
        new_load.has_predicate = true;
        new_load.predicate = valid_pred;
        aux->push_back(new_load);

        const temp_ptr vsrc(*auxillary);

        aux->push_back(make_add(uptr_t, vsrc, origin, limit));

        const type_t btype = bitwise_type(statement.type);
        const operand_t vdst = make_validity_operand(statement.operands[0], 0);

        statement_t new_vload = statement;
        new_vload.has_predicate = true;
        new_vload.predicate = valid_pred;
        new_vload.type = btype;
        new_vload.operands[0] = vdst;
        new_vload.operands[1] = vsrc;
        aux->push_back(new_vload);

        /* If we don't do the load, invalidate on the not_valid_pred flag. */
        const size_t ni = vdst.identifier.size();
        const operand_t negone = operand_t::make_iconstant(-1);

        /* Flag the error condition if one exists. */
        inst_t::error_desc_t desc_oob;
        desc_oob.type = inst_t::outofbounds_ld_local;
        desc_oob.orig = statement;
        auxillary->inst->errors.push_back(desc_oob);
        const size_t error_oob = auxillary->inst->errors.size();
        const operand_t op_oob = operand_t::make_iconstant((int64_t) error_oob);

        aux->push_back(make_selp(u32_type, valid_pred, local_errors,
            local_errors, op_oob));

        /**
         * nvcc does not seem to generate byte-sized registers, so while the
         * load is a byte-wide, we don't get clued in on the size of our target
         * register.  For now, widen the register (and hope this behavior
         * doesn't change).
         */
        const type_t bwtype = (btype == b8_type) ?  b16_type : btype;

        for (size_t i = 0; i < ni; i++) {
            operand_t o;
            o.op_type = vdst.op_type;
            o.identifier.push_back(vdst.identifier[i]);
            o.field.push_back(vdst.field[i]);

            statement_t s;
            s.op = op_mov;
            s.type = bwtype;
            s.has_predicate = true;
            s.is_negated = true;
            s.predicate = valid_pred;
            s.operands.push_back(o);
            s.operands.push_back(negone);
            aux->push_back(s);
        }
    } else {
        new_src  = data_ptr;
        new_vsrc = validity_ptr_src;

        aux->push_back(make_mov(ptr_t, chidx,
            operand_t::make_identifier(src.identifier[0])));
        if (src.op_type == operand_addressable &&
                src.offset != 0) {
            aux->push_back(make_add(uptr_t, chidx, chidx, src.offset));
        }

        aux->push_back(make_mov(uptr_t, global_ro_reg, global_ro));
        aux->push_back(make_mov(ptr_t, original_ptr, chidx));
        aux->push_back(make_mov(ptr_t, inidx, chidx));

        aux->push_back(make_shr(ptr_t, chidx, chidx, lg_chunk_bytes));
        aux->push_back(make_and(ptr_t, chidx, chidx, max_chunks - 1));
        aux->push_back(make_shl(ptr_t, chidx, chidx, log_ptr));

        aux->push_back(make_and(ptr_t, inidx, inidx, chunk_size - 1));
        aux->push_back(make_mov(ptr_t, vidx, inidx));
        aux->push_back(make_shr(ptr_t, inidx, inidx, 3));

        aux->push_back(make_ld(ptr_t, const_space, chunk, master));
        aux->push_back(make_add(uptr_t, chunk, chidx, chunk));
        aux->push_back(make_ld(ptr_t, global_space, chunk_ptr, chunk));

        aux->push_back(make_add(uptr_t, validity_ptr_src, chunk_ptr, vidx));
        aux->push_back(make_add(uptr_t, chunk_ptr, chunk_ptr, inidx));
        aux->push_back(make_ld(read_t, global_space, a_data, a_data_ptr));
        if (width <= 8) {
            /* We cannot setp on a u8 */
            aux->push_back(make_cvt(wread_t, read_t, a_wdata, a_data));

            /**
             * Since we load a byte at a time (the address bits for 8 bytes)
             * even if we are performing a load of less than 8 bytes, we need
             * to shift and mask.
             *
             * Grab lower bits of the address and shift accordingly.  Shifts
             * take u32's as their shift argument.
             */
            const type_t bwread_t = bitwise_type(wread_t);
            switch (sizeof(void *)) {
                case 8u:
                aux->push_back(make_cvt(u32_type, uptr_t, a_data32, vidx));
                break;
                case 4u:
                aux->push_back(make_mov(u32_type, a_data32, vidx));
                break;
            }
            aux->push_back(make_and(b32_type, a_data32, a_data32,
                sizeof(void *) - 1));
            aux->push_back(make_shr(wread_t, a_wdata, a_wdata, a_data32));

            /**
             * Mask high bits.
             */
            aux->push_back(make_and(bwread_t, a_wdata, a_wdata,
                (1 << width) - 1));
        }

        aux->push_back(make_setp(wread_t, cmp_eq, valid_pred, a_cdata,
            operand_t::make_iconstant((1 << width) - 1)));
        aux->push_back(make_selp(ptr_t, valid_pred, data_ptr, original_ptr,
            global_ro_reg));

        /* Flag the error condition if one exists. */
        inst_t::error_desc_t desc_oob;
        desc_oob.type = inst_t::outofbounds_ld_global;
        desc_oob.orig = statement;
        auxillary->inst->errors.push_back(desc_oob);
        const size_t error_oob = auxillary->inst->errors.size();
        const operand_t op_oob = operand_t::make_iconstant((int64_t) error_oob);

        aux->push_back(make_selp(u32_type, valid_pred, local_errors,
            local_errors, op_oob));

        if (offsetof(metadata_chunk, v_data)) {
            aux->push_back(make_add(uptr_t, validity_ptr_src, validity_ptr_src,
                offsetof(metadata_chunk, v_data)));
        }
    }

    *keep = false;

    assert(!(invalidate) || !(validate));
    if (invalidate) {
        /**
         * TODO:  We may want to actually fill in the data with some data.
         *
         * Mark destinations as invalid.
         */
        const operand_t & dst = statement.operands[0];
        const size_t ni = dst.identifier.size();
        const type_t btype = bitwise_type(statement.type);
        const operand_t negone = operand_t::make_iconstant(-1);
        for (size_t i = 0; i < ni; i++) {
            aux->push_back(make_mov(btype, operand_t::make_identifier(
                    make_validity_symbol(dst.identifier[i])), -1));
        }
    } else if (validate) {
        const operand_t & dst = statement.operands[0];
        const size_t ni = dst.identifier.size();
        const type_t btype = bitwise_type(statement.type);
        const operand_t zero = operand_t::make_iconstant(0);
        for (size_t i = 0; i < ni; i++) {
            aux->push_back(make_mov(btype, operand_t::make_identifier(
                    make_validity_symbol(dst.identifier[i])), 0));
        }
    }

    if (!(drop_load)) {
        statement_t new_load = statement;
        new_load.operands[1] = new_src;
        aux->push_back(new_load);

        statement_t new_vload = statement;
        new_vload.type = bitwise_type(new_vload.type);
        operand_t & vdst = new_vload.operands[0];
        assert(vdst.op_type == operand_identifier);
        assert(vdst.identifier.size() == vdst.field.size());
        for (size_t i = 0; i < vdst.identifier.size(); i++) {
            vdst.identifier[i] = make_validity_symbol(
                vdst.identifier[i]);
        }
        new_vload.operands[1] = new_vsrc;
        aux->push_back(new_vload);
    }
}

void global_context_memcheck::instrument_mad(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 4u);
    *keep = true;

    std::vector<operand_t> source_validity;
    for (unsigned i = 1; i < 4; i++) {
        const operand_t & op   = statement.operands[i];
        const operand_type_t t = op.op_type;
        switch (t) {
            case operand_identifier:
                source_validity.push_back(
                    make_validity_operand(op, 0));
                break;
            case operand_addressable:
            case operand_indexed:
            case invalid_operand:
                assert(0 && "Invalid operand type.");
                break;
            case operand_constant:
            case operand_float:
            case operand_double:
                /* Skip. */
                break;
        }
    }

    const operand_t & d = statement.operands[0];
    const operand_t vd  = make_validity_operand(d, 0);

    const type_t btype  = bitwise_type(statement.type);
    const type_t stype  = signed_type(statement.type);

    const temp_operand tmp(*auxillary, btype);
    const temp_operand tmp1(*auxillary, btype);

    const operand_t vc = operand_t::make_identifier(__vcarry);
    const operand_t zero = operand_t::make_iconstant(0);

    const size_t vargs = source_validity.size();
    const bool carry_in = (statement.op == op_madc);
    if (vargs == 0) {
        if (carry_in) {
            /* Copy carry flag validity bits. */
            aux->push_back(make_mov(btype, vd, vc));

            /* Validity of carry-out is validity carry-in, so do
             * nothing. */
        } else {
            /* No nonconstant arguments, so result is valid. */
            aux->push_back(make_mov(btype, vd, 0));

            /* Clear carry out. */
            if (statement.carry_out) {
                aux->push_back(make_mov(btype, vc, 0));
            }
        }

        return;
    } else {
        /* One or more nonconstant arguments, fold into a
         * temporary variable, then spread invalid bits. */
        if (vargs == 1u) {
            /* Move directly into temporary variable. */
            aux->push_back(make_mov(btype, tmp, source_validity[0]));
        } else {
            aux->push_back(make_or(btype, tmp,
                source_validity[0], source_validity[1]));
            for (size_t i = 2; i < vargs; i++) {
                aux->push_back(make_or(btype, tmp, tmp, source_validity[i]));
            }
        }

        /* Fold-in the carry bits. */
        if (carry_in) {
            aux->push_back(make_or(btype, tmp, tmp, vc));
        }
    }

    switch (statement.type) {
        case u16_type:
        case u32_type:
        case u64_type:
        case s16_type:
        case s32_type:
        case s64_type:
            /**
             * In the worst case, invalid bits are propagated to
             * bits as or more significant as the original invalid
             * bits.
             */

            /**
             * Spread left.
             */
            aux->push_back(make_neg(stype, tmp1, tmp));
            aux->push_back(make_or(btype, vd, tmp, tmp1));

            /**
             * Carry out.
             */
            if (statement.carry_out) {
                assert(statement.type == u32_type ||
                       statement.type == s32_type);
                aux->push_back(make_shr(stype, vc, vd, 31));
            }

            break;
        case f32_type:
        case f64_type:
            assert(!(statement.carry_out));

            /* Spread invalid bits across type. */
            aux->push_back(make_cnot(btype, tmp, tmp));
            aux->push_back(make_sub(stype, vd, tmp, 1));
            break;
        case b8_type:
        case b16_type:
        case b32_type:
        case b64_type:
        case u8_type:
        case s8_type:
        case f16_type:
        case pred_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Invalid type.");
            break;
    }
}

void global_context_memcheck::instrument_math2(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    /**
     * We work under the assumption that these operations are all
     * or nothing w.r.t. validity.
     */
    assert(statement.operands.size() == 3u);

    std::vector<operand_t> source_validity;
    for (unsigned i = 1; i < 3; i++) {
        const operand_t & op  = statement.operands[i];
        const operand_t   vop = make_validity_operand(op, 0);
        if (!(vop.is_constant())) {
            source_validity.push_back(vop);
        }
    }

    const size_t vargs = source_validity.size();
    const type_t btype = bitwise_type(statement.type);
    const type_t stype = signed_type(statement.type);

    const operand_t & d = statement.operands[0];
    const operand_t vd  = make_validity_operand(d, 0);

    assert(vargs <= 2);
    switch (vargs) {
        case 0:
            /**
             * No variable arguments, result is defined.
             */
            aux->push_back(make_mov(btype, vd, 0));
            break;
        case 1: {
            /**
             * Propagate from source.
             */
            const temp_operand tmp0(*auxillary, btype);

            aux->push_back(make_cnot(btype, tmp0, source_validity[0]));
            aux->push_back(make_sub(stype, vd, tmp0, 1));
            break; }
        case 2: {
            /**
             * OR validity bits of two results, then propagate.
             */
            const temp_operand tmp0(*auxillary, btype);

            aux->push_back(make_or(btype, tmp0,
                source_validity[0], source_validity[1]));
            aux->push_back(make_cnot(btype, tmp0, tmp0));
            aux->push_back(make_sub(stype, vd, tmp0, 1));
            break; }
    }

    *keep = true;
}

void global_context_memcheck::instrument_mov(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);

    const operand_t & a = statement.operands[1];
    const operand_t & d = statement.operands[0];
    const size_t awidth = statement.type == pred_type ? 0 :
        (sizeof_type(statement.type) /
         (a.is_constant() ? 1 : a.identifier.size()));

    *keep = true;

    /* Short cut for the simple case, constants/sreg's */
    operand_t va = make_validity_operand(a, awidth);
    if (!(va.is_constant())) {
        const char * zero_register = NULL;
        switch (awidth) {
            case 1:
                zero_register = __zero_register8;
                break;
            case 2:
                zero_register = __zero_register16;
                break;
            case 4:
                zero_register = __zero_register32;
                break;
            case 8:
                zero_register = __zero_register64;
                break;
            default:
                assert(0 && "Unsupported width.");
                break;
        }

        const size_t n = va.identifier.size();
        std::vector<bool> done(n, false);
        std::vector<bool> fixed(n, false);
        size_t rem = n;
        for (size_t i = 0; i < n; i++) {
            if (va.identifier[i] == zero_register) {
                done[i] = true;
                fixed[i] = true;
                rem--;
            }
        }

        /* Walk up scopes for identifiers. */
        const block_t * b = auxillary->block;
        const function_t * f = NULL;
        while (b) {
            assert(!(b->parent) || !(b->fparent));

            if (rem == 0) {
                break;
            }

            if (b->block_type == block_scope) {
                const scope_t * s = b->scope;

                for (size_t i = 0; i < n; i++) {
                    /* We're done with this identifier. */
                    if (done[i]) { continue; }

                    const size_t vn = s->variables.size();
                    for (size_t vi = 0; vi < vn; vi++) {
                        const variable_t & v = s->variables[vi];
                        if (a.identifier[i] == v.name &&
                                !(v.has_suffix) &&
                                v.space != reg_space) {
                            /* This is a fixed symbol. */
                            done[i] = true;
                            fixed[i] = true;
                            rem--;
                            break;
                        }

                        /* TODO: Consider marking off variables that we've
                         * resolved to registers. */
                    }
                }
            }

            /* Move up. */
            f = b->fparent;
            b = b->parent;
        }

        if (rem > 0) {
            assert(f);

            /* TODO:  Check off parameters. */

            const ptx_t * p = f->parent;
            for (size_t i = 0; i < n; i++) {
                /* We're done with this identifier. */
                if (done[i]) { continue; }

                const size_t vn = p->variables.size();
                for (size_t vi = 0; vi < vn; vi++) {
                    const variable_t & v = p->variables[vi];
                    if (a.identifier[i] == v.name &&
                            !(v.has_suffix) &&
                            v.space != reg_space) {
                        /* This is a fixed symbol. */
                        done[i] = true;
                        fixed[i] = true;
                        rem--;
                        break;
                    }
                }
            }
        }

        for (size_t i = 0; i < n; i++) {
            if (fixed[i])  {
                va.identifier[i] = zero_register;
                va.field[i]      = field_none;
            }
        }
    }

    const size_t dwidth = statement.type == pred_type ? 1u :
        (sizeof_type(statement.type) /
         (d.is_constant() ? 1 : d.identifier.size()));
    const operand_t vd = make_validity_operand(d, dwidth);
    statement_t copy = statement;
    copy.op = op_mov;
    if (copy.type == pred_type) {
        copy.type = u16_type;
    } else {
        copy.type = bitwise_type(copy.type);
    }

    copy.operands.clear();
    copy.operands.push_back(vd);
    copy.operands.push_back(va);
    aux->push_back(copy);
}

void global_context_memcheck::instrument_neg(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    const type_t btype  = bitwise_type(statement.type);

    const operand_t & d = statement.operands[0];
    const operand_t & a = statement.operands[0];

    const operand_t vd  = make_validity_operand(d, 0);
    const operand_t va  = make_validity_operand(a, 0);

    switch (statement.type) {
        case s16_type:
        case s32_type:
        case s64_type: {
            const type_t stype  = statement.type;
            const temp_operand tmp(*auxillary, stype);

            aux->push_back(make_neg(stype, tmp, va));
            aux->push_back(make_or(btype, vd, tmp, va));
            break; }
        case f32_type:
        case f64_type:
            /**
             * TODO:  Do not ignore ftz.
             *
             * The resulting validity is the same as the source validity as we
             * are only flipping the sign bit.
             *
             * For now, we ignore that NaN's produce unspecified NaNs (per PTX
             * ISA 3.0).
             */

            aux->push_back(make_mov(btype, vd, va));
            break;
        case s8_type:
        case b8_type:
        case u8_type:
        case b16_type:
        case f16_type:
        case u16_type:
        case b32_type:
        case u32_type:
        case b64_type:
        case u64_type:
        case pred_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            break;
    }

    *keep = true;
}

void global_context_memcheck::instrument_not(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);
    const operand_t & a = statement.operands[1];
    const operand_t & d = statement.operands[0];

    const operand_t va = make_validity_operand(a, 0);
    const operand_t vd = make_validity_operand(d, 0);

    switch (statement.type) {
        case pred_type:
            /**
             * Validity information is simply transfered.
             */
            aux->push_back(make_mov(b16_type, vd, va));

            assert(d.identifier.size() > 0);
            auxillary->inst->unchecked.insert(d.identifier[0]);
            break;
        case b16_type:
        case b32_type:
        case b64_type:
            aux->push_back(make_mov(statement.type, vd, va));
            break;
        case f32_type:
        case f64_type:
        case s16_type:
        case s32_type:
        case s64_type:
        case u16_type:
        case u32_type:
        case u64_type:
        case s8_type:
        case b8_type:
        case u8_type:
        case f16_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            break;
    }

    *keep = true;
    (void) auxillary;
}

void global_context_memcheck::instrument_or(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    *keep = true;

    assert(statement.operands.size() == 3u);
    const operand_t & a = statement.operands[1];
    const operand_t & b = statement.operands[2];
    const operand_t & d = statement.operands[0];

    const operand_t va = make_validity_operand(a, 0);
    const operand_t vb = make_validity_operand(b, 0);
    const operand_t vd = make_validity_operand(d, 0);

    switch (statement.type) {
        case pred_type:
            /**
             * For simplicity, we assume the worst case and OR the
             * validity bits without regard for the underlying
             * data.
             */
            if (va.is_constant() && vb.is_constant()) {
                aux->push_back(make_mov(b16_type, vd, va));
            } else if (va.is_constant() && !(vb.is_constant())) {
                aux->push_back(make_mov(b16_type, vd, vb));
            } else if (!(va.is_constant()) && vb.is_constant()) {
                aux->push_back(make_mov(b16_type, vd, va));
            } else {
                aux->push_back(make_or(b16_type, vd, va, vb));
            }

            assert(d.identifier.size() > 0);
            auxillary->inst->unchecked.insert(d.identifier[0]);
            break;
        case b16_type:
        case b32_type:
        case b64_type:
            if (va.is_constant() && vb.is_constant()) {
                aux->push_back(make_mov(statement.type, vd, va));
            } else if (va.is_constant() && !(vb.is_constant())) {
                aux->push_back(make_mov(statement.type, vd, vb));
            } else if (!(va.is_constant()) && vb.is_constant()) {
                aux->push_back(make_mov(statement.type, vd, va));
            } else {
                aux->push_back(make_or(statement.type, vd, va, vb));
            }

            break;
        case f32_type:
        case f64_type:
        case s16_type:
        case s32_type:
        case s64_type:
        case u16_type:
        case u32_type:
        case u64_type:
        case s8_type:
        case b8_type:
        case u8_type:
        case f16_type:
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            break;
    }
}

void global_context_memcheck::instrument_ret(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    *keep = false;

    const operand_t local_errors =
        operand_t::make_identifier(__errors_register);

    /* setp.ne.u32 __has_errors, local_errors, 0 */
    statement_t setp;
    setp.op      = op_setp;
    setp.type    = u32_type;
    setp.cmp     = cmp_ne;
    setp.bool_op = bool_none;
    setp.has_ppredicate = true;
    setp.ppredicate = __has_errors;

    setp.operands.push_back(local_errors);

    const operand_t op_zero = operand_t::make_iconstant(0);
    setp.operands.push_back(op_zero);

    aux->push_back(setp);

    /* @__has_errors atom.global.inc.u32 global_errors,
     *   [reg_error_count], 0xFFFFFFFF */
    statement_t atom;

    atom.has_predicate = true;
    atom.predicate = __has_errors;

    atom.op        = op_atom;
    atom.space     = global_space;
    atom.atomic_op = atom_inc;
    atom.type      = u32_type;

    const operand_t op_ge =
        operand_t::make_identifier(__reg_global_errors);
    atom.operands.push_back(op_ge);

    const operand_t op_error_count =
        operand_t::make_identifier(__reg_error_count);
    atom.operands.push_back(op_error_count);

    const operand_t op_max_u32 =
        operand_t::make_iconstant(0x3FFFFFFF);
    atom.operands.push_back(op_max_u32);

    aux->push_back(atom);

    /**
     * We want the count to grow without limit rather than hit the
     * edge case of wrapping around to precisely zero, but we need
     * to bound the number of values we write out to the size of
     * our buffer.
     *
     * min.u32 global_errors, global_errors, (buffer size)
     */
    aux->push_back(make_min(u32_type, op_ge, op_ge,
        operand_t::make_iconstant(
        (sizeof(error_buffer_t) / sizeof(uint32_t)) - 1)));

    /* shl.b32 global_errors, global_errors, 2 */
    statement_t shl;
    shl.op  = op_shl;
    shl.type = b32_type;
    shl.operands.push_back(op_ge);
    shl.operands.push_back(op_ge);

    const operand_t op_two =
        operand_t::make_iconstant(2);
    shl.operands.push_back(op_two);

    aux->push_back(shl);

    /* TODO:  This is not necessary when sizeof(void *) == 4 */
    /* cvt.PTRTYPE.u32 global_errorsw, global_errors */
    statement_t cvt;
    cvt.op = op_cvt;

    const type_t uptr_t = upointer_type();
    cvt.type = uptr_t;
    cvt.type2 = u32_type;

    const operand_t op_gew =
        operand_t::make_identifier(__reg_global_errorsw);

    cvt.operands.push_back(op_gew);
    cvt.operands.push_back(op_ge);

    aux->push_back(cvt);

    /* add.PTRTYPE error_address, error_address, global_errorsw */
    statement_t add;
    add.op = op_add;
    add.type = uptr_t;

    const operand_t op_error_address =
        operand_t::make_identifier(__errors_address);
    add.operands.push_back(op_error_address);
    add.operands.push_back(op_error_address);
    add.operands.push_back(op_gew);

    aux->push_back(add);

    /* @__has_errors st.global.u32 [error_address],
     *   local_errors */
    statement_t st;
    st.has_predicate = true;
    st.predicate = __has_errors;
    st.op    = op_st;
    st.space = global_space;
    st.type  = u32_type;
    st.operands.push_back(op_error_address);
    st.operands.push_back(local_errors);
    aux->push_back(st);

    aux->push_back(statement);
    (void) auxillary;
}

void global_context_memcheck::instrument_prefetch(
        const statement_t & statement, statement_vt * aux, bool * keep,
        internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 1u);

    const operand_t local_errors =
        operand_t::make_identifier(__errors_register);

    /**
     * From the vtest_k_prefetch test, we make the following
     * observations:
     * 1.  Prefetch operations in explicit spaces do not seem to cause faults
     *     when they are out of bounds.
     * 2.  There is some amount of slack:  Prefetch operations at the bounds of
     *     a memory allocation succeed without errors (even after accounting for
     *     cache line size).  For local operations, there seems to be 2k of
     *     slack.  For global operations, there can be up to 1M of slack (from
     *     empirical experiments with the Prefetch.GlobalEdgeOfBounds test).
     *     Since allocations are commonly aligned, Panoptes permits 1k of slack.
     * 3.  Misaligned global accesses in the generic space with prefetch (but
     *     not prefetchu) fail.  Misaligned local accesses in the generic space
     *     fail with either prefetch operation.
     *
     *     Misaligned accesses in explicit spaces succeed.
     *
     * From this, we implement the following checks.
     * 1.  Operations with explicit space specifications are not checked.
     * 2.  Generic operations are checked for alignment and bounds.
     */
    if (statement.space == generic_space) {
        /**
         * The "good" state of the predicates is true if:
         * -> has_predicate is false
         * -> has_predicate is true and is_negated is false
         */
        const bool good = !(statement.has_predicate) ||
            !(statement.is_negated);
        const operand_t up = operand_t::make_identifier(
            statement.predicate);

        const temp_operand op0(*auxillary, pred_type);
        const temp_operand op1(*auxillary, pred_type);
        const temp_operand op2(*auxillary, pred_type);

        const temp_ptr ptr(*auxillary);

        const operand_t & a = statement.operands[0];
        operand_t clean_a = a;
        assert(clean_a.op_type == operand_identifier ||
               clean_a.op_type == operand_addressable ||
               clean_a.op_type == operand_constant);
        if (clean_a.op_type == operand_addressable) {
            assert(clean_a.offset == 0 &&
                "Offset preloads are not supported.");
            clean_a.op_type = operand_identifier;
            clean_a.field.push_back(field_none);
        }

        const operand_t va  = make_validity_operand(clean_a, 0);
        const operand_t zero = operand_t::make_iconstant(0);

        /* If everything is okay, the predicates are set to good. */

        /* Check alignment. */
        const operand_t ptrmask = operand_t::make_iconstant(
            sizeof(void *) - 1);
        const type_t ptr_t = pointer_type();

        aux->push_back(make_and(ptr_t, ptr, clean_a, ptrmask));
        aux->push_back(make_setp(ptr_t,
            good ? cmp_eq : cmp_ne, op0, ptr, zero));

        /* If prefetchu and the address is global, suppress this error. */
        if (statement.op == op_prefetchu) {
            aux->push_back(make_isspacep(global_space, op2, clean_a));
            if (good) {
                aux->push_back(make_or(pred_type, op0, op0, op2));
            } else {
                aux->push_back(make_and(pred_type, op0, op0, op2));
            }
        }

        /* If has_predicate, mix in user predicate. */
        if (statement.has_predicate) {
            if (statement.is_negated) {
                aux->push_back(make_or(pred_type, op0, op0, up));
            } else {
                aux->push_back(make_and(pred_type, op0, op0, up));
            }
        }

        typedef internal::instrumentation_t inst_t;
        inst_t::error_desc_t desc_align;
        desc_align.type = inst_t::misaligned_prefetch;
        desc_align.orig = statement;
        auxillary->inst->errors.push_back(desc_align);
        const size_t error_align = auxillary->inst->errors.size();
        const operand_t op_align =
            operand_t::make_iconstant((int64_t) error_align);

        aux->push_back(make_selp(u32_type, op0, local_errors,
            good ? local_errors : op_align,
            good ? op_align     : local_errors));

        /* Check bounds. */
        const intptr_t ialignmask =
            -((intptr_t) sizeof(void *) - 1);
        const operand_t oalignmask =
            operand_t::make_iconstant(ialignmask);
        /* Check for local.
         *
         * If not local:
         *      p1 not initialized.
         *      p2 is false.
         * If valid local:
         *      p1 is true.
         *      p2 is true.
         * If invalid local:
         *      p1 is false.
         *      p2 is true.
         */

        aux->push_back(make_isspacep(local_space, op2, clean_a));

        const type_t uptr_t = upointer_type();
        {
            statement_t c;
            c.has_predicate = true;
            c.is_negated    = false;
            c.predicate     = op2;
            c.op            = op_cvta;
            c.is_to         = true;
            c.space         = local_space;
            c.type          = uptr_t;
            c.operands.push_back(ptr);
            assert(clean_a.op_type != operand_addressable);
            c.operands.push_back(clean_a);
            assert(c.operands[1].op_type == operand_identifier);
            aux->push_back(c);

            /**
             * 0xFFFCA0 was derived by taking the address of
             * variously sized local memory addresses.  Rather
             * consistently, given:
             *    .local .b8 t[N];
             *
             *    t + N == 0xFFFCA0.
             *
             * TODO: Panoptes currently does not consider underflow.
             */
            statement_t s;
            s.has_predicate  = true;
            s.is_negated     = false;
            s.predicate      = op2;
            s.op             = op_setp;
            s.type           = uptr_t;
            s.cmp            = cmp_ge;
            s.has_ppredicate = true;
            s.ppredicate     = op1;
            s.operands.push_back(
                operand_t::make_iconstant(0xFFFCA0));
            s.operands.push_back(ptr);
            aux->push_back(s);
        }

        if (!(good)) {
            aux->push_back(make_not(pred_type, op1, op1));
        }

        if (statement.has_predicate) {
            if (statement.is_negated) {
                aux->push_back(make_or(pred_type, op1, op1, up));
            } else {
                aux->push_back(make_and(pred_type, op1, op1, up));
            }
        }

        inst_t::error_desc_t desc_oobl;
        desc_oobl.type = inst_t::outofbounds_prefetch_local;
        desc_oobl.orig = statement;
        auxillary->inst->errors.push_back(desc_oobl);
        const size_t error_oobl = auxillary->inst->errors.size();
        const operand_t op_oobl =
            operand_t::make_iconstant((int64_t) error_oobl);

        aux->push_back(make_selp(u32_type, op1, local_errors,
            good ? local_errors : op_oobl,
            good ? op_oobl      : local_errors));

        aux->back().has_predicate = true;
        aux->back().predicate     = op2;

        /* Fold validity predicate into p0. */
        if (good) {
            aux->push_back(make_and(pred_type, op0, op0, op1));
        } else {
            aux->push_back(make_or(pred_type, op0, op0, op1));
        }

        aux->back().has_predicate = true;
        aux->back().predicate     = op2;

        /* Check for global bounds.  Panoptes predicates costlier instructions
         * on !(p2), e.g., loads.  While the other instructions are unnecessary
         * as well, shuffling bits around that will discarded later harms
         * nothing.
         */
        {
            const size_t chunk_size = 1u << lg_chunk_bytes;
            const size_t chunk_mask =
                (1u << (lg_max_memory - lg_chunk_bytes)) - 1u;

            /**
             * chunk *** ptr = __master;
             */
            const operand_t master =
                operand_t::make_identifier(__master_symbol);

            /**
             * chunk ** ptr = *ptr;
             */
            aux->push_back(
                make_ld(uptr_t, const_space, ptr, master));

            aux->back().has_predicate = true;
            aux->back().is_negated    = true;
            aux->back().predicate     = op2;

            /**
             * uintptr_t ptr1 = clean_a >> lg_chunk_bytes;
             * ptr1 &= max_chunks;
             */
            const temp_ptr ptr1(*auxillary);
            statement_t c;
            c.op    = op_cvta;
            c.is_to = true;
            c.space = global_space;
            c.type  = uptr_t;
            c.operands.push_back(ptr1);
            c.operands.push_back(clean_a);
            aux->push_back(c);

            aux->push_back(make_shr(ptr_t, ptr1, ptr1, lg_chunk_bytes));
            aux->push_back(make_and(ptr_t, ptr1, ptr1, chunk_mask));

            /**
             * ptr1 *= sizeof(void *);
             */
            if (sizeof(void *) == 8) {
                aux->push_back(make_shl(ptr_t, ptr1, ptr1, 3));
            } else {
                aux->push_back(make_shl(ptr_t, ptr1, ptr1, 2));
            }

            /**
             * ptr += ptr1;
             */
            aux->push_back(make_add(uptr_t, ptr, ptr, ptr1));

            /**
             * chunk * ptr = *ptr;
             */
            aux->push_back(make_ld(uptr_t, global_space, ptr, ptr));

            aux->back().has_predicate = true;
            aux->back().is_negated    = true;
            aux->back().predicate     = op2;

            /*
             * Align:
             * ptr1 = clean_a;
             * ptr1 &= (chunk_size - 1) & ~(1024 - 1);
             * (1024 is the amount of slack permitted for global
             *  addresses.)
             *
             * Ignore lower bits.
             * ptr  >>= 3;
             */
            aux->push_back(c);
            aux->push_back(make_and(ptr_t, ptr1, ptr1,
                (chunk_size - 1) & ~(1024u - 1u)));
            aux->push_back(make_shr(ptr_t, ptr1, ptr1, 3));

            /**
             * uint8_t * ptr += ptr1;
             */
            aux->push_back(make_add(uptr_t, ptr, ptr, ptr1));

            /**
             * uintptr_t ptr = *ptr;
             */
            aux->push_back(make_ld(uptr_t, global_space, ptr, ptr));

            aux->back().has_predicate = true;
            aux->back().is_negated    = true;
            aux->back().predicate     = op2;

            statement_t s;
            s.has_predicate = true;
            s.is_negated    = true;
            s.predicate     = op2;
            s.op            = op_setp;
            s.type          = uptr_t;
            s.cmp           = cmp_eq;
            s.has_ppredicate= true;
            s.ppredicate    = op1;
            s.operands.push_back(ptr);
            s.operands.push_back(operand_t::make_iconstant(-1));
            aux->push_back(s);
        }

        if (!(good)) {
            aux->push_back(make_not(pred_type, op1, op1));
        }

        if (statement.has_predicate) {
            if (statement.is_negated) {
                aux->push_back(make_or(pred_type, op1, op1, up));
            } else {
                aux->push_back(make_and(pred_type, op1, op1, up));
            }
        }

        inst_t::error_desc_t desc_oobg;
        desc_oobg.type = inst_t::outofbounds_prefetch_global;
        desc_oobg.orig = statement;
        auxillary->inst->errors.push_back(desc_oobg);
        const size_t error_oobg = auxillary->inst->errors.size();
        const operand_t op_oobg =
            operand_t::make_iconstant((int64_t) error_oobg);

        aux->push_back(make_selp(u32_type, op1, local_errors,
            good ? local_errors : op_oobg,
            good ? op_oobg      : local_errors));

        aux->back().has_predicate = true;
        aux->back().is_negated    = true;
        aux->back().predicate     = op2;

        /* Fold validity predicate into p0. */
        if (good) {
            aux->push_back(make_and(pred_type, op0, op0, op1));
        } else {
            aux->push_back(make_or(pred_type, op0, op0, op1));
        }

        aux->back().has_predicate = true;
        aux->back().is_negated    = true;
        aux->back().predicate     = op2;

        /* Check validity. */
        aux->push_back(make_setp(ptr_t,
            good ? cmp_eq : cmp_ne, op1, va, zero));
        if (statement.has_predicate) {
            if (statement.is_negated) {
                aux->push_back(make_or(pred_type, op1, op1, up));
            } else {
                aux->push_back(make_and(pred_type, op1, op1, up));
            }
        }

        inst_t::error_desc_t desc_wild;
        desc_wild.type = inst_t::wild_prefetch;
        desc_wild.orig = statement;
        auxillary->inst->errors.push_back(desc_wild);
        const size_t error_wild = auxillary->inst->errors.size();
        const operand_t op_wild =
            operand_t::make_iconstant((int64_t) error_wild);

        aux->push_back(make_selp(u32_type, op1, local_errors,
            good ? local_errors : op_wild,
            good ? op_wild      : local_errors));
        /* Fold validity predicate into p0. */
        if (good) {
            aux->push_back(make_and(pred_type, op0, op0, op1));
        } else {
            aux->push_back(make_or(pred_type, op0, op0, op1));
        }

        /**
         * Prefetch the global read only address if invalid.
         */
        const operand_t global_ro =
            operand_t::make_identifier(__global_ro);
        aux->push_back(make_mov(ptr_t, ptr, global_ro));
        aux->push_back(make_selp(ptr_t, op0, ptr, clean_a, ptr));

        *keep = false;
        statement_t new_prefetch = statement;
        new_prefetch.operands.clear();
        new_prefetch.operands.push_back(ptr);
        aux->push_back(new_prefetch);
    }
}

void global_context_memcheck::instrument_prmt(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 4u);
    assert(statement.type == b32_type);
    *keep = true;

    const operand_t & a = statement.operands[1];
    const operand_t & b = statement.operands[2];
    const operand_t & c = statement.operands[3];
    const operand_t & d = statement.operands[0];

    const operand_t va = make_validity_operand(a, 0);
    const operand_t vb = make_validity_operand(b, 0);
    const operand_t vc = make_validity_operand(c, 0);
    const operand_t vd = make_validity_operand(d, 0);

    assert(!(vd.is_constant()));

    /**
     * If c is constant, then we can just select the validity with c from va
     * and vb.
     */
    if (vc.is_constant()) {
        if (va.is_constant() && vb.is_constant()) {
            aux->push_back(make_mov(b32_type, vd, va));
        } else {
            statement_t vprmt = statement;
            vprmt.operands[0] = vd;
            vprmt.operands[1] = va;
            vprmt.operands[2] = vb;
            aux->push_back(vprmt);
        }

        return;
    }

    /**
     * We always need a temporary 32-bit value.
     */
    const temp_operand tmp(*auxillary, b32_type);
    operand_t immed;
    bool direct;

    if (va.is_constant() && vb.is_constant()) {
        /* We can move the results directly into vd */
        immed = vd;
        direct = true;
    } else {
        immed = tmp;
        direct = false;
    }

    /* An invalid bit anywhere corrupts the output. */
    int64_t mask;
    if (statement.prmt_mode == prmt_default) {
        mask = 0x0000FFFF;
    } else {
        mask = 0x00000003;
    }

    aux->push_back(make_and(b32_type, tmp, vc, mask));
    aux->push_back(make_cnot(b32_type, tmp, tmp));
    aux->push_back(make_sub(s32_type, immed, tmp, 1));

    if (direct) {
        return;
    }

    bool just_convert;
    operand_t immed2;
    operand_t immed3 = immed;
    if (va.is_constant() & vb.is_constant()) {
        /* We're done after we convert. */
        immed2 = vd;
        just_convert = true;
    } else {
        /* We need two more intermediate values, one to hold the result of
         * selecting and one to hold the converted contents of immed. */
        just_convert = false;

        temp_operand t2(*auxillary, b32_type);
        immed2 = t2;
    }

    std::swap(immed2, immed3);

    if (!(just_convert)) {
        /* Mix in validity of va and vb via selection. */
        statement_t vprmt = statement;
        vprmt.operands[0] = immed3;
        vprmt.operands[1] = va;
        vprmt.operands[2] = vb;
        aux->push_back(vprmt);
        aux->push_back(make_or(b32_type, vd, immed2, immed3));
    }
}

void global_context_memcheck::instrument_sad(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 4u);
    *keep = true;

    assert(statement.type == u16_type ||
           statement.type == u32_type ||
           statement.type == u64_type ||
           statement.type == s16_type ||
           statement.type == s32_type ||
           statement.type == s64_type);

    /**
     * We first need to consider the validity bits of the intermediate
     * abs(a-b).  The invalid bits of a and b propagate to the left; abs
     * propagates the invalidity of the sign bit to the right.  Therefore, if
     * any bits of a or b are invalid, the result abs(a-b) is wholly invalid.
     *
     * We place the result of the intermediate validity bits in the operand
     * immed.
     */
    bool immed_constant;
    operand_t immed;

    std::vector<operand_t> source_validity;
    for (unsigned i = 1; i < 3; i++) {
        const operand_t & op  = statement.operands[i];
        const operand_t   vop = make_validity_operand(op, 0);
        if (!(vop.is_constant())) {
            source_validity.push_back(vop);
        }
    }

    const size_t vargs = source_validity.size();
    const type_t btype = bitwise_type(statement.type);
    const type_t stype = signed_type(statement.type);

    const temp_operand tmp0(*auxillary, btype);

    assert(vargs <= 2);
    switch (vargs) {
        case 0:
            /**
             * No variable arguments, result is defined.
             */
            immed = operand_t::make_iconstant(0);
            immed_constant = true;
            break;
        case 1:
            /**
             * Propagate from source.
             */
            aux->push_back(make_cnot(btype, tmp0,
                source_validity[0]));
            aux->push_back(make_sub(stype, tmp0, tmp0, 1));

            immed = tmp0;
            immed_constant = false;
            break;
        case 2:
            /**
             * OR validity bits of two results, then propagate.
             */
            aux->push_back(make_or(btype, tmp0,
                source_validity[0], source_validity[1]));
            aux->push_back(make_cnot(btype, tmp0, tmp0));
            aux->push_back(make_sub(stype, tmp0, tmp0, 1));
            immed = tmp0;
            immed_constant = false;
            break;
    }

    /**
     * Consider the argument to c.
     */
    const operand_t & c  = statement.operands[3];
    const operand_t   vc = make_validity_operand(c, 0);
    const bool cconstant = vc.is_constant();

    const operand_t & d = statement.operands[0];
    const operand_t vd  = make_validity_operand(d, 0);

    if (cconstant && immed_constant) {
        /**
         * Result is constant.  We already have initialized immed to zero.
         */
        aux->push_back(make_mov(btype, vd, immed));
        return;
    }

    if (cconstant) {
        /**
         * We've computed all that we need to validity-bitwise in computing
         * immed.
         */
        aux->push_back(make_mov(btype, vd, immed));
        return;
    }

    if (immed_constant) {
        /**
         * We need to left propagate vc.
         */
        aux->push_back(make_neg(stype, tmp0, vc));
        aux->push_back(make_or(btype, vd, tmp0, vc));
    } else {
        /**
         * OR result and then left propgate.
         *
         * We need the temporary variable for immed and another to compute our
         * left propagation.
         */
        const temp_operand tmp1(*auxillary, btype);

        assert(immed != tmp1);

        aux->push_back(make_or(btype, immed, immed, vc));
        aux->push_back(make_neg(stype, tmp1, immed));
        aux->push_back(make_or(btype, vd, tmp1, immed));
    }
}

void global_context_memcheck::instrument_selp(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 4u);
    *keep = true;

    const operand_t & d = statement.operands[0];
    const operand_t & a = statement.operands[1];
    const operand_t & b = statement.operands[2];
    const operand_t & c = statement.operands[3];
    assert(!(c.is_constant()));

    const operand_t vd = make_validity_operand(d, 0);
    const operand_t va = make_validity_operand(a, 0);
    const operand_t vb = make_validity_operand(b, 0);
    const operand_t vc = make_validity_operand(c, 0);

    /**
     * Sign extend validity bits associated with c to the width of
     * the operation.
     */
    const type_t stype = signed_type(statement.type);
    const temp_operand tmp(*auxillary, stype);
    operand_t immed;
    if (stype == s16_type) {
        immed = vc;
    } else {
        immed = tmp;
        aux->push_back(make_cvt(stype, s16_type, tmp, vc));
    }

    const type_t btype = bitwise_type(statement.type);
    if (!(va.is_constant()) || !(vb.is_constant())) {
        /* Mix validity bits from a and b according to c. */
        const temp_operand btmp(*auxillary, btype);

        statement_t selp;
        selp.op = op_selp;
        selp.type = btype;
        selp.operands.push_back(btmp);
        selp.operands.push_back(va);
        selp.operands.push_back(vb);
        selp.operands.push_back(c);
        aux->push_back(selp);

        aux->push_back(make_or(btype, vd, btmp, immed));
    } else {
        /* Move results of vc into vd. */
        aux->push_back(make_mov(btype, vd, immed));
    }
}

void global_context_memcheck::instrument_set(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() >= 3u &&
           statement.operands.size() <= 4u);
    *keep = true;

    const operand_t & d = statement.operands[0];
    const operand_t & a = statement.operands[1];
    const operand_t & b = statement.operands[2];

    const operand_t vd = make_validity_operand(d, 0);
    const operand_t va = make_validity_operand(a, 0);
    const operand_t vb = make_validity_operand(b, 0);

    const type_t bdtype = bitwise_type(statement.type);
    const type_t sdtype = signed_type(statement.type);
    const size_t dwidth = sizeof_type(statement.type);

    const type_t bstype = bitwise_type(statement.type2);
    const type_t sstype = signed_type(statement.type2);
    const size_t swidth = sizeof_type(statement.type2);

    const temp_operand tmp(*auxillary, bstype);
    const temp_operand tmpd(*auxillary, bdtype);

    operand_t immed;
    if (va.is_constant() && vb.is_constant()) {
        immed = va;
    } else {
        operand_t dest;
        if (swidth == dwidth) {
            dest = tmp;
        } else {
            dest = tmpd;
        }

        if (!(va.is_constant()) && vb.is_constant()) {
            immed = va;
        } else if (va.is_constant() && !(vb.is_constant())) {
            immed = vb;
        } else {
            /* Merge invalid bits. */
            immed = tmp;
            aux->push_back(make_or(bstype, tmp, va, vb));
        }

        /* 0 -> 1, (anything) -> 0 */
        aux->push_back(make_cnot(bstype, tmp, immed));

        if (swidth != dwidth) {
            aux->push_back(make_cvt(sdtype, sstype, dest, tmp));
        }

        /* 0 -> 0, (anything) -> -1 */
        aux->push_back(make_sub(sdtype, dest, dest, 1));

        immed = dest;
    }

    /**
     * Fold in validity bits from c, if provided.
     */
    if (statement.operands.size() == 4u) {
        const operand_t & c = statement.operands[2];
        const operand_t vc = make_validity_operand(c, 0);

        if (immed.is_constant()) {
            immed = vc;
        } else {
            const temp_operand tmp2(*auxillary, bdtype);

            aux->push_back(make_cvt(sdtype, s16_type, tmp2, vc));
            aux->push_back(make_or(bdtype, immed, immed, tmp2));
        }
    }

    aux->push_back(make_mov(bdtype, vd, immed));
}

void global_context_memcheck::instrument_setp(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() >= 2u &&
           statement.operands.size() <= 3u);
    *keep = true;

    const operand_t & a = statement.operands[0];
    const operand_t & b = statement.operands[1];
    const operand_t va = make_validity_operand(a, 0);
    const operand_t vb = make_validity_operand(b, 0);

    const type_t btype = bitwise_type(statement.type);
    const type_t stype = signed_type(statement.type);
    const size_t width = sizeof_type(statement.type);

    const temp_operand tmp(*auxillary, btype);
    const temp_operand tmp16(*auxillary, b16_type);

    operand_t immed;
    if (va.is_constant() && vb.is_constant()) {
        immed = va;
    } else {
        operand_t otmp;
        if (width == 2u) {
            otmp = tmp;
        } else {
            otmp = tmp16;
        }

        if (!(va.is_constant()) && vb.is_constant()) {
            immed = va;
        } else if (va.is_constant() && !(vb.is_constant())) {
            immed = vb;
        } else {
            /* Merge invalid bits. */
            immed = tmp;
            aux->push_back(make_or(btype, tmp, va, vb));
        }

        /* 0 -> 1, (anything) -> 0 */
        aux->push_back(make_cnot(btype, tmp, immed));

        if (width != 2u) {
            /* Convert to 16 bits */
            aux->push_back(make_cvt(s16_type, stype, otmp, tmp));
        }

        /* 0 -> 0, (anything) -> 0xFFFF */
        aux->push_back(make_sub(s16_type, otmp, otmp, 1));

        immed = otmp;
    }

    /**
     * Fold in validity bits from c, if provided.
     */
    if (statement.operands.size() == 3u) {
        const operand_t & c = statement.operands[2];
        const operand_t vc = make_validity_operand(c, 0);

        if (immed.is_constant()) {
            immed = vc;
        } else {
            aux->push_back(make_or(b16_type, immed, immed, vc));
        }
    }

    assert(statement.has_ppredicate);
    const operand_t vp = operand_t::make_identifier(
        make_validity_symbol(statement.ppredicate));
    operand_t vq;
    if (statement.has_qpredicate) {
        vq = operand_t::make_identifier(
            make_validity_symbol(statement.qpredicate));
    }

    aux->push_back(make_mov(b16_type, vp, immed));
    auxillary->inst->unchecked.insert(statement.ppredicate);
    if (statement.has_qpredicate) {
        aux->push_back(make_mov(b16_type, vq, immed));
        auxillary->inst->unchecked.insert(statement.qpredicate);
    }
}

void global_context_memcheck::instrument_shift(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 3u);
    *keep = true;

    const size_t width = sizeof_type(statement.type);

    const operand_t & a  = statement.operands[1];
    const operand_t   va = make_validity_operand(a, 0);
    const bool aconstant = va.is_constant();

    const operand_t & b  = statement.operands[2];
    const operand_t   vb = make_validity_operand(b, 0);
    const bool bconstant = vb.is_constant();

    const operand_t & d = statement.operands[0];
    const operand_t vd = make_validity_operand(d, 0);

    /**
     * We shift the validity bits just as the data bits are shifted
     * with the provision that if the shift amount is in any way
     * invalid, the entire result will be invalid.
     *
     * PTX clamps shifts greater than the register width to the
     * register width, so the bits beyond those needed to represent
     * the width still carry significance (and therefore cannot be
     * masked out).
     */
    const type_t btype = bitwise_type(statement.type);
    const type_t stype = signed_type(statement.type);
    if (aconstant && bconstant) {
        /* Result is completely valid. */
        aux->push_back(make_mov(btype, vd, 0));
    } else if (aconstant) {
        /* Result is dependant on the validity bits of b.  b is
         * always a 32-bit value. */
        const temp_operand tmp(*auxillary, b32_type);

        aux->push_back(make_cnot(b32_type, tmp, vb));
        if (width == 4u) {
            /* We can write the result directly. */
            aux->push_back(make_sub(s32_type, vd, tmp, 1));
        } else {
            /* Compute into temporary, the convert. */
            aux->push_back(make_sub(s32_type, tmp, tmp, 1));
            aux->push_back(make_cvt(stype, s32_type, vd, tmp));
        }
    } else if (bconstant) {
        /* Shift the validity bits of a according to b. */
        statement_t sh;
        sh.op   = statement.op;
        sh.type = statement.type;
        sh.operands.push_back(vd);
        sh.operands.push_back(va);
        sh.operands.push_back(b);
        aux->push_back(sh);
    } else {
        /* Join sources of validity bits. */
        const temp_operand tmp0(*auxillary, btype);

        statement_t sh;
        sh.op   = statement.op;
        sh.type = statement.type;
        sh.operands.push_back(tmp0);
        sh.operands.push_back(va);
        sh.operands.push_back(b);
        aux->push_back(sh);

        operand_t intermediate;
        if (width == 4u) {
            const temp_operand tmp1(*auxillary, b32_type);

            aux->push_back(make_cnot(b32_type, tmp1, vb));
            aux->push_back(make_sub(s32_type, tmp1, tmp1, 1));
            intermediate = tmp1;
        } else {
            assert(btype != b32_type);
            const temp_operand tmp32_0(*auxillary, b32_type);
            assert(tmp0 != tmp32_0);
            const temp_operand tmp1(*auxillary, btype);

            aux->push_back(make_cnot(b32_type, tmp32_0, vb));
            aux->push_back(make_sub(s32_type, tmp32_0, tmp32_0, 1));
            aux->push_back(make_cvt(stype, s32_type, tmp1, tmp32_0));

            intermediate = tmp1;
        }

        /* OR the result from shifting and from spreading b's invalid bits. */
        aux->push_back(make_or(btype, vd, tmp0, intermediate));
    }
}

void global_context_memcheck::instrument_slct(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    *keep = true;

    assert(statement.operands.size() == 4u);
    assert(statement.type2 == f32_type ||
           statement.type2 == s32_type);

    const operand_t & a = statement.operands[1];
    const operand_t & b = statement.operands[2];
    const operand_t & c = statement.operands[3];
    const operand_t & d = statement.operands[0];

    const operand_t va = make_validity_operand(a, 0);
    const operand_t vb = make_validity_operand(b, 0);
    const operand_t vc = make_validity_operand(c, 0);
    const operand_t vd = make_validity_operand(d, 0);

    assert(!(vd.is_constant()));

    /**
     * If c is constant, then we can just select the validity with c from va
     * and vb.
     */
    const type_t btype = bitwise_type(statement.type);
    if (vc.is_constant()) {
        if (va.is_constant() && vb.is_constant()) {
            aux->push_back(make_mov(btype, vd, va));
        } else {
            aux->push_back(make_slct(btype, statement.type2,
                vd, va, vb, c));
        }

        return;
    }

    /**
     * We always need a temporary 32-bit value.
     */
    const temp_operand tmp(*auxillary, b32_type);

    const size_t width = sizeof_type(statement.type);
    operand_t immed;
    bool direct;
    if (va.is_constant() && vb.is_constant() && width == 4u) {
        /* We can move the results directly into vd */
        immed = vd;
        direct = true;
    } else {
        immed = tmp;
        direct = false;
    }

    /**
     * We spread the invalid bits of c according to the type of the operation,
     * we then narrow to the destination type.
     */
    if (statement.type2 == f32_type) {
        /* An invalid bit anywhere corrupts the output. */
        aux->push_back(make_cnot(b32_type, tmp, vc));
        aux->push_back(make_sub(s32_type, immed, tmp, 1));
    } else if (statement.type2 == s32_type) {
        /* The sign bit governs whether the output is invalid */
        aux->push_back(make_shr(s32_type, immed, vd, 31));
    }

    if (direct) {
        return;
    }

    bool just_convert;
    operand_t immed2;
    operand_t immed3 = immed;
    if (va.is_constant() & vb.is_constant()) {
        /* We're done after we convert. */
        immed2 = vd;
        just_convert = true;
    } else {
        /* We need two more intermediate values, one to hold the result of
         * selecting and one to hold the converted contents of immed. */
        just_convert = false;

        const temp_operand t2(*auxillary, btype);
        immed2 = t2;
    }

    if (width != 4u) {
        /* Our width didn't match.  Convert. */
        assert(immed2 != immed);
        aux->push_back(make_cvt(signed_type(statement.type),
            s32_type, immed2, immed));
    } else {
        assert(immed3 == immed);
        std::swap(immed2, immed3);
    }

    if (!(just_convert)) {
        /* Mix in validity of va and vb via selection. */
        aux->push_back(make_slct(btype, statement.type2,
            immed3, va, vb, c));
        aux->push_back(make_or(btype, vd, immed2, immed3));
    }
}

void global_context_memcheck::instrument_st(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);
    const operand_t & dst = statement.operands[0];
    /*
     * This is similar to op_ld
     */

    const temp_ptr original_ptr(*auxillary);
    const temp_ptr inidx(*auxillary);
    const temp_ptr chunk(*auxillary);
    const temp_ptr chunk_ptr(*auxillary);
    const operand_t a_data_ptr = operand_t::make_addressable(chunk_ptr,
        offsetof(metadata_chunk, a_data));
    const temp_ptr data_ptr(*auxillary);
    const temp_ptr validity_ptr_dst(*auxillary);
    const operand_t validity_ptr = operand_t::make_addressable(validity_ptr_dst,
        offsetof(metadata_chunk, v_data));
    const temp_ptr chidx(*auxillary);
    const temp_ptr global_wo_reg(*auxillary);
    const temp_ptr vidx(*auxillary);

    const operand_t master = operand_t::make_identifier(__master_symbol);
    const operand_t global_ro = operand_t::make_identifier(__global_ro);
    const operand_t global_wo = operand_t::make_identifier(__global_wo);

    const temp_operand valid_pred(*auxillary, pred_type);

    const size_t chunk_size = 1u << lg_chunk_bytes;
    const size_t max_chunks = (1u << (lg_max_memory - lg_chunk_bytes)) - 1u;
    const size_t cwidth = sizeof_type(statement.type);
    const size_t width  = cwidth * (unsigned) statement.vector;
    const type_t read_t = unsigned_of(width);
    const temp_operand a_data(*auxillary, read_t);

    const type_t wread_t = width <= 8 ? u16_type : read_t;
    const temp_operand a_wdata(*auxillary, u16_type);
    const operand_t a_cdata = width <= 8 ? a_wdata : a_data;
    const temp_operand a_data32(*auxillary, u32_type);

    operand_t new_dst, new_vdst;

    unsigned log_ptr;
    switch (sizeof(void *)) {
        case 4u: log_ptr = 2; break;
        case 8u: log_ptr = 3; break;
    }

    bool drop_store = false;

    const type_t ptr_t  = pointer_type();
    const type_t uptr_t = upointer_type();

    const operand_t local_errors =
        operand_t::make_identifier(__errors_register);

    typedef internal::instrumentation_t inst_t;
    if (statement.space == param_space) {
        assert(dst.identifier.size() == 1u);
        new_dst  = dst;
        new_vdst = operand_t::make_identifier(
            make_validity_symbol(dst.identifier[0]));
    } else if (statement.space == shared_space) {
        assert(dst.identifier.size() == 1u);
        const std::string & id = dst.identifier[0];

        /* Walk up scopes for identifiers. */
        const function_t * f = NULL;
        const variable_t * v = NULL;
        bool found = find_variable(id, shared_space, auxillary->block, &f, &v);

        if (!(found)) {
            assert(f);
            found = find_variable_global(id, shared_space, f->parent, &v);
        }

        const bool   flexible = found ? v->array_flexible : false;
        const size_t size     = found ? v->size() : 0u;

        if (found && !(flexible)) {
            /* We found a fixed symbol, verify we do not statically overrun
             * it. */
            const size_t end = width + (size_t) dst.offset;
            if (dst.offset < 0) {
                std::stringstream ss;
                ss << statement;

                char msg[256];
                int ret = snprintf(msg, sizeof(msg),
                    "Shared store of %zu bytes at offset "
                    "%ld will underrun buffer:\n"
                    "Disassembly: %s\n", width,
                    dst.offset, ss.str().c_str());

                assert(ret < (int) sizeof(msg) - 1);
                logger::instance().print(msg);

                /* Cast off bits into the ether. */
                drop_store = true;
            } else if (end > size) {
                std::stringstream ss;
                ss << statement;

                char msg[256];
                int ret = snprintf(msg, sizeof(msg),
                    "Shared store of %zu bytes at offset "
                    "%ld will overrun buffer:\n"
                    "Disassembly: %s\n", width,
                    dst.offset, ss.str().c_str());

                assert(ret < (int) sizeof(msg) - 1);
                logger::instance().print(msg);

                /* Cast off bits into the ether. */
                drop_store = true;
            } else {
                /* Map it to the validity symbol. */
                new_dst  = dst;

                new_vdst = dst;
                new_vdst.identifier.clear();
                new_vdst.identifier.push_back(make_validity_symbol(id));
                new_vdst.field.clear();
                new_vdst.field.push_back(field_none);
            }
        } else {
            /* Verify address against the shared size parameter. */

            const operand_t limit =
                operand_t::make_identifier(__shared_reg);

            drop_store = true;

            aux->push_back(make_mov(ptr_t, original_ptr,
                operand_t::make_identifier(dst.identifier[0])));
            if (dst.op_type == operand_addressable &&
                    dst.offset != 0) {
                aux->push_back(make_add(uptr_t, original_ptr,
                    original_ptr, dst.offset));
            }

            {
                const temp_ptr tmp(*auxillary);
                aux->push_back(make_add(uptr_t, tmp, original_ptr, (int) width));
                aux->push_back(make_setp(uptr_t, cmp_ge, valid_pred, limit,
                    tmp));
            }

            statement_t new_store = statement;
            /**
             * TODO:  We need to propagate the predicate flags of predicated
             * stores.
             */
            assert(!(new_store.has_predicate));
            new_store.has_predicate = true;
            new_store.predicate = valid_pred;
            aux->push_back(new_store);

            const temp_ptr vdst(*auxillary);

            aux->push_back(make_add(uptr_t, vdst, original_ptr, limit));

            statement_t new_vstore = statement;
            new_vstore.has_predicate = true;
            new_vstore.predicate = valid_pred;
            new_vstore.type = bitwise_type(new_vstore.type);
            new_vstore.operands[0] = vdst;
            new_vstore.operands[1] =
                make_validity_operand(new_vstore.operands[1], cwidth);
            aux->push_back(new_vstore);

            /* Flag the error condition if one exists. */
            inst_t::error_desc_t desc_oob;
            desc_oob.type = inst_t::outofbounds_st_shared;
            desc_oob.orig = statement;
            auxillary->inst->errors.push_back(desc_oob);
            const size_t error_oob = auxillary->inst->errors.size();
            const operand_t op_oob =
                operand_t::make_iconstant((int64_t) error_oob);

            aux->push_back(make_selp(u32_type, valid_pred, local_errors,
                local_errors, op_oob));
        }
    } else if (statement.space == local_space) {
        const operand_t limit = operand_t::make_identifier(__local_reg);
        drop_store = true;

        /* Verify address against the local size parameter. */
        operand_t origin;

        if (dst.is_constant()) {
            origin = dst;

            const temp_ptr tmp(*auxillary);
            aux->push_back(make_add(uptr_t, tmp, origin, (int) width));
            aux->push_back(make_setp(uptr_t, cmp_ge, valid_pred, limit, tmp));
            /* TODO:  Check for underrun. */
        } else {
            assert(dst.identifier.size() == 1u);
            const std::string & id = dst.identifier[0];

            aux->push_back(make_mov(ptr_t, original_ptr,
                operand_t::make_identifier(id)));
            if (dst.op_type == operand_addressable &&
                    dst.offset != 0) {
                aux->push_back(make_add(uptr_t, original_ptr, original_ptr,
                    dst.offset));
            }

            origin = original_ptr;

            const temp_ptr tmp(*auxillary);
            aux->push_back(make_add(uptr_t, tmp, origin, (int) width));
            aux->push_back(make_setp(uptr_t, cmp_ge, valid_pred,
                operand_t::make_iconstant(0xFFFCA0), tmp));
        }

        statement_t new_store = statement;
        /**
         * TODO:  We need to propagate the predicate flags of predicated
         * stores.
         */
        assert(!(new_store.has_predicate));
        new_store.has_predicate = true;
        new_store.predicate = valid_pred;
        aux->push_back(new_store);

        const temp_ptr vdst(*auxillary);
        aux->push_back(make_add(uptr_t, vdst, origin, limit));

        statement_t new_vstore = statement;
        new_vstore.has_predicate = true;
        new_vstore.predicate = valid_pred;
        new_vstore.type = bitwise_type(new_vstore.type);
        new_vstore.operands[0] = vdst;
        new_vstore.operands[1] =
            make_validity_operand(new_vstore.operands[1], cwidth);
        aux->push_back(new_vstore);

        /* Flag the error condition if one exists. */
        inst_t::error_desc_t desc_oob;
        desc_oob.type = inst_t::outofbounds_st_local;
        desc_oob.orig = statement;
        auxillary->inst->errors.push_back(desc_oob);
        const size_t error_oob = auxillary->inst->errors.size();
        const operand_t op_oob =
            operand_t::make_iconstant((int64_t) error_oob);

        aux->push_back(make_selp(u32_type, valid_pred, local_errors,
            local_errors, op_oob));
    } else {
        aux->push_back(make_mov(uptr_t, global_wo_reg, global_wo));

        aux->push_back(make_mov(ptr_t, chidx,
            operand_t::make_identifier(dst.identifier[0])));
        if (dst.op_type == operand_addressable &&
                dst.offset != 0) {
            aux->push_back(make_add(uptr_t, chidx, chidx, dst.offset));
        }

        aux->push_back(make_mov(ptr_t, original_ptr, chidx));
        aux->push_back(make_mov(ptr_t, inidx, chidx));

        aux->push_back(make_shr(ptr_t, chidx, chidx, lg_chunk_bytes));
        aux->push_back(make_and(ptr_t, chidx, chidx, max_chunks - 1));
        aux->push_back(make_shl(ptr_t, chidx, chidx, log_ptr));

        aux->push_back(make_and(ptr_t, inidx, inidx, chunk_size - 1));
        aux->push_back(make_mov(ptr_t, vidx, inidx));
        aux->push_back(make_shr(ptr_t, inidx, inidx, 3));

        aux->push_back(make_ld(ptr_t, const_space, chunk, master));
        aux->push_back(make_add(uptr_t, chunk, chidx, chunk));
        aux->push_back(make_ld(ptr_t, global_space, chunk_ptr, chunk));

        aux->push_back(make_add(uptr_t, validity_ptr_dst, chunk_ptr, vidx));

        aux->push_back(make_add(uptr_t, chunk_ptr, chunk_ptr, inidx));
        aux->push_back(make_ld(read_t, global_space, a_data, a_data_ptr));
        if (width <= 8) {
            /* We cannot setp on a u8 */
            aux->push_back(make_cvt(wread_t, read_t, a_wdata, a_data));

            /* See explaination under op_ld.
             *
             * Grab lower bits of the address and shift accordingly.  Shifts
             * take u32's as their shift argument.
             */
            const type_t bwread_t = bitwise_type(wread_t);
            switch (sizeof(void *)) {
                case 8u:
                aux->push_back(make_cvt(u32_type, uptr_t, a_data32, vidx));
                break;
                case 4u:
                aux->push_back(make_mov(u32_type, a_data32, vidx));
                break;
            }
            aux->push_back(make_and(b32_type, a_data32, a_data32,
                sizeof(void *) - 1));
            aux->push_back(make_shr(wread_t, a_wdata, a_wdata, a_data32));

            /**
             * Mask high bits.
             */
            aux->push_back(make_and(bwread_t, a_wdata, a_wdata,
                (1 << width) - 1));
        }

        aux->push_back(make_setp(wread_t, cmp_eq, valid_pred, a_cdata,
            operand_t::make_iconstant((1 << width) - 1)));

        aux->push_back(make_selp(ptr_t, valid_pred, data_ptr, original_ptr,
            global_wo_reg));
        if (offsetof(metadata_chunk, v_data)) {
            aux->push_back(make_add(uptr_t, validity_ptr_dst, validity_ptr_dst,
                offsetof(metadata_chunk, v_data)));
        }
        aux->push_back(make_selp(ptr_t, valid_pred, validity_ptr_dst,
            validity_ptr_dst, global_wo_reg));

        /* Flag the error condition if one exists. */
        inst_t::error_desc_t desc_oob;
        desc_oob.type = inst_t::outofbounds_st_global;
        desc_oob.orig = statement;
        auxillary->inst->errors.push_back(desc_oob);
        const size_t error_oob = auxillary->inst->errors.size();
        const operand_t op_oob =
            operand_t::make_iconstant((int64_t) error_oob);

        aux->push_back(make_selp(u32_type, valid_pred, local_errors,
            local_errors, op_oob));

        new_dst  = data_ptr;
        new_vdst = validity_ptr_dst;
    }

    *keep = false;

    statement_t new_st = statement;
    new_st.operands[0] = new_dst;

    statement_t new_vst = statement;
    new_vst.type = bitwise_type(new_vst.type);
    new_vst.operands[0] = new_vdst;
    operand_t & vsrc = new_vst.operands[1];
    assert(vsrc.op_type == operand_identifier);
    assert(vsrc.identifier.size() == vsrc.field.size());
    for (size_t i = 0; i < vsrc.identifier.size(); i++) {
        vsrc.identifier[i] = make_validity_symbol(vsrc.identifier[i]);
    }

    if (!(drop_store)) {
        aux->push_back(new_st);
        aux->push_back(new_vst);
    }
}

void global_context_memcheck::instrument_testp(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);
    const operand_t & d = statement.operands[0];
    const operand_t & a = statement.operands[1];

    const operand_t vd = make_validity_operand(d, 0);
    const operand_t va = make_validity_operand(a, 0);

    const type_t btype = bitwise_type(statement.type);
    const type_t stype = signed_type(statement.type);

    if (va.is_constant()) {
        /* Answer is well defined. */
        aux->push_back(make_mov(b16_type, vd, 0));
    } else {
        /* Any invalid bits mean the result is invalid. */
        const temp_operand tmp(*auxillary, btype);

        aux->push_back(make_cnot(btype, tmp, va));
        aux->push_back(make_sub(stype, tmp, tmp, 1));
        aux->push_back(make_cvt(s16_type, stype, vd, tmp));
    }

    *keep = true;
}

void global_context_memcheck::instrument_vote(const statement_t & statement,
        statement_vt * aux, bool * keep, internal::auxillary_t * auxillary) {
    assert(statement.operands.size() == 2u);
    const operand_t & d = statement.operands[0];
    const operand_t & a = statement.operands[1];

    const operand_t vd = make_validity_operand(d, 0);
    const operand_t va = make_validity_operand(a, 0);

    const operand_t zero   = operand_t::make_iconstant(0);
    const operand_t negone = operand_t::make_iconstant(-1);

    switch (statement.vote_mode) {
        case invalid_vote_mode:
            assert(0 && "Invalid vote mode.");
            break;
        case vote_all:
        case vote_any:
        case vote_uniform:
            /**
             * Panoptes adopts a slightly less precise approach: If any inputs
             * are invalid, the result is invalid.  (In principle, it is
             * possible for say, vote.any, to be valid if at least one true
             * input is valid.)
             */
            if (va.is_constant()) {
                aux->push_back(make_mov(b16_type, vd, 0));
            } else {
                const temp_operand tmpp(*auxillary, pred_type);
                aux->push_back(make_setp(b16_type, cmp_ne, tmpp, va, zero));

                statement_t v;
                v.op = op_vote;
                v.type = pred_type;
                v.vote_mode = vote_any;
                v.operands.push_back(tmpp);
                v.operands.push_back(tmpp);
                aux->push_back(v);

                aux->push_back(make_selp(b16_type, tmpp, vd, negone, zero));
            }

            break;
        case vote_ballot:
            if (va.is_constant()) {
                aux->push_back(make_mov(b32_type, vd, 0));
            } else {
                const temp_operand tmpp(*auxillary, pred_type);
                aux->push_back(make_setp(b16_type, cmp_ne, tmpp, va, zero));

                statement_t v;
                v.op = op_vote;
                v.type = b32_type;
                v.vote_mode = vote_ballot;
                v.operands.push_back(vd);
                v.operands.push_back(tmpp);
                aux->push_back(v);
            }

            break;
    }

    *keep = true;
}

void global_context_memcheck::instrument_block(block_t * block,
        internal::instrumentation_t * inst,
        const entry_info_t & e) {
    if (block->block_type == block_invalid) {
        assert(0 && "Invalid block type.");
        return;
    } else if (block->block_type == block_label) {
        /* No operations necessary. */
        return;
    } else if (block->block_type == block_statement) {
        /* We do not instrument lone statements. */
        assert(0 && "Lone statements cannot be instrumented.");
    }

    assert(block->block_type == block_scope);
    scope_t * scope = block->scope;
    const size_t vcount = scope->variables.size();
    for (size_t i = 0; i < vcount; i++) {
        /* Note which predicates haven't been checked. */
        const variable_t & v = scope->variables[i];
        if (v.type == pred_type) {
            if (v.has_suffix) {
                const char * name = v.name.c_str();
                const size_t len = v.name.size();
                char buf[256];

                assert(len < sizeof(buf) - 1u);
                const size_t mlen = std::min(len, sizeof(buf) - 1u);
                const size_t rem = sizeof(buf) - mlen;
                memcpy(buf, name, mlen);

                for (int j = 0; j < v.suffix; j++) {
                    int ret = snprintf(buf + mlen, rem, "%d", j);
                    assert(ret < (int) rem);
                    inst->unchecked.insert(buf);
                }
            } else {
                inst->unchecked.insert(v.name);
            }
        }

        variable_t vv = scope->variables[i];
        vv.name = make_validity_symbol(vv.name);
        if (vv.type == f32_type) {
            vv.type = b32_type;
        } else if (vv.type == f64_type) {
            vv.type = b64_type;
        } else if (vv.type == pred_type) {
            vv.type = b16_type;
        }
        scope->variables.push_back(vv);
    }

    const operand_t local_errors =
        operand_t::make_identifier(__errors_register);

    internal::auxillary_t temps;
    temps.inst = inst;
    temps.block = block;

    for (scope_t::block_vt::iterator it = scope->blocks.begin();
            it != scope->blocks.end(); ) {
        if ((*it)->block_type != block_statement) {
            instrument_block(*it, inst, e);
            ++it;
            continue;
        }

        const statement_t & statement = *(*it)->statement;
        std::vector<statement_t> aux;
        bool keep = true;

        /* Check for predicate. */
        if (statement.has_predicate) {
            typedef internal::instrumentation_t inst_t;
            inst_t::sset_t::iterator jit =
                inst->unchecked.find(statement.predicate);
            if (jit != inst->unchecked.end()) {
                /* We haven't checked this predicate lately, insert a check. */
                const operand_t vp = operand_t::make_identifier(
                    make_validity_symbol(statement.predicate));
                const temp_operand tmpp(temps, pred_type);

                inst_t::error_desc_t desc;
                desc.type = inst_t::wild_branch;
                desc.orig = statement;
                inst->errors.push_back(desc);
                const size_t error_number = inst->errors.size();

                aux.push_back(make_setp(b16_type, cmp_ne, tmpp, vp,
                    operand_t::make_iconstant(0)));
                aux.push_back(make_selp(u32_type, tmpp, local_errors,
                    operand_t::make_iconstant((int64_t) error_number),
                    local_errors));

                inst->unchecked.erase(jit);
            }
        }

        switch (statement.op) {
            case op_abs:
                instrument_abs(statement, &aux, &keep, &temps);
                break;
            case op_add:
            case op_addc:
            case op_mul:
            case op_mul24:
            case op_sub:
            case op_subc:
                instrument_add(statement, &aux, &keep, &temps);
                break;
            case op_and:
                instrument_and(statement, &aux, &keep, &temps);
                break;
            case op_atom:
            case op_red:
                instrument_atom(statement, &aux, &keep, &temps);
                break;
            case op_bar:
                instrument_bar(statement, &aux, &keep, &temps);
                break;
            case op_bfe:
                instrument_bfe(statement, &aux, &keep, &temps);
                break;
            case op_bfi:
                instrument_bfi(statement, &aux, &keep, &temps);
                break;
            case op_brev:
                instrument_brev(statement, &aux, &keep, &temps);
                break;
            case op_brkpt: /* No-op */ break;
            case op_copysign:
                instrument_copysign(statement, &aux, &keep, &temps);
                break;
            case op_testp:
                instrument_testp(statement, &aux, &keep, &temps);
                break;
            case op_cos:
            case op_ex2:
            case op_lg2:
            case op_rcp:
            case op_rsqrt:
            case op_sin:
            case op_sqrt: /* Floating point, 1-argument operations. */
                instrument_fp1(statement, &aux, &keep, &temps);
                break;
            case op_cvt:
                instrument_cvt(statement, &aux, &keep, &temps);
                break;
            case op_cvta:
                /**
                 * Just pass through invalid bits when loading from registers,
                 * but use mov's machinery for recognizing identifiers when
                 * initializing to a good value.
                 */
            case op_mov:
                instrument_mov(statement, &aux, &keep, &temps);
                break;
            case op_min:
            case op_max:
                /**
                 * This is an imprecise calculation of the validity bits for
                 * min and max.  In principle, if the min/max value would
                 * remain the min/max value (respectively) regardless of the
                 * validity bits, we could safely propagate the validity bits
                 * of the min/max value *alone*.  Since doing so would require
                 * substantially more logic for a (hopefully) fleetingly rare
                 * condition, this imprecise method is preferable.
                 *
                 * Fallthrough.
                 */
            case op_div:
            case op_rem:
                instrument_math2(statement, &aux, &keep, &temps);
                break;
            case op_exit:
            case op_ret:
                instrument_ret(statement, &aux, &keep, &temps);
                break;
            case op_fma: /* Floating point, 3-argument operations. */
                instrument_fp3(statement, &aux, &keep, &temps);
                break;
            case op_invalid: /* No-op */ break;
            case op_isspacep:
                instrument_isspacep(statement, &aux, &keep, &temps);
                break;
            case op_ld:
            case op_ldu:
                instrument_ld(statement, &aux, &keep, &temps);
                break;
            case op_mad:
            case op_mad24:
            case op_madc:
                instrument_mad(statement, &aux, &keep, &temps);
                break;
            case op_membar: /* No-op */ break;
            case op_neg:
                instrument_neg(statement, &aux, &keep, &temps);
                break;
            case op_not:
                instrument_not(statement, &aux, &keep, &temps);
                break;
            case op_bra: /* No-op */ break;
            case op_call: /* No-op */ break;
            case op_cnot:
                instrument_cnot(statement, &aux, &keep, &temps);
                break;
            case op_bfind:
            case op_clz:
                /**
                 * This approximation is less precise than it could be, but it
                 * is likely good enough for our purposes.
                 *
                 * Fall through.
                 */
            case op_popc:
                instrument_bit1(statement, &aux, &keep, &temps);
                break;
            case op_pmevent: /* No-op */ break;
            case op_prefetch:
            case op_prefetchu:
                instrument_prefetch(statement, &aux, &keep, &temps);
                break;
            case op_prmt:
                instrument_prmt(statement, &aux, &keep, &temps);
                break;
            case op_sad:
                instrument_sad(statement, &aux, &keep, &temps);
                break;
            case op_shl:
            case op_shr:
                instrument_shift(statement, &aux, &keep, &temps);
                break;
            case op_selp:
                instrument_selp(statement, &aux, &keep, &temps);
                break;
            case op_set:
                instrument_set(statement, &aux, &keep, &temps);
                break;
            case op_setp:
                instrument_setp(statement, &aux, &keep, &temps);
                break;
            case op_slct:
                instrument_slct(statement, &aux, &keep, &temps);
                break;
            case op_st:
                instrument_st(statement, &aux, &keep, &temps);
                break;
            case op_trap: /* noop */ break;
            case op_vote:
                instrument_vote(statement, &aux, &keep, &temps);
                break;
            case op_or:
            case op_xor:
                instrument_or(statement, &aux, &keep, &temps);
                break;
        }

        scope_t::block_vt::iterator tmp_it = it;
        scope_t::block_vt::iterator old_it = it;

        ++tmp_it;
        const size_t acount = aux.size();
        for (size_t i = 0; i < acount; i++) {
            block_t * b = new block_t();
            b->block_type   = block_statement;
            b->parent       = (*it);
            b->statement    = new statement_t(aux[i]);

            it = scope->blocks.insert(it, b);
            ++it;
        }

        if (!(keep)) {
            delete *old_it;
            scope->blocks.erase(old_it);
        }
        it = tmp_it;
    }

    for (unsigned i = 0; i < 4; i++) {
        const unsigned c = temps.b[i];
        if (c == 0) {
            continue;
        }

        variable_t v;

        v.space = reg_space;
        v.type = bitwise_of(1u << i);

        char sym[16];
        int ret = snprintf(sym, sizeof(sym), "__panoptes_b%d_", 8u << i);
        assert(ret < (int) sizeof(sym));

        v.name = sym;
        v.has_suffix = true;
        v.suffix = (int) c;

        scope->variables.push_back(v);
    }

    for (unsigned i = 0; i < 4; i++) {
        const unsigned c = temps.s[i];
        if (c == 0) {
            continue;
        }

        variable_t v;

        v.space = reg_space;
        v.type = signed_of(1u << i);

        char sym[16];
        int ret = snprintf(sym, sizeof(sym), "__panoptes_s%d_", 8u << i);
        assert(ret < (int) sizeof(sym));

        v.name = sym;
        v.has_suffix = true;
        v.suffix = (int) c;

        scope->variables.push_back(v);
    }

    for (unsigned i = 0; i < 4; i++) {
        const unsigned c = temps.u[i];
        if (c == 0) {
            continue;
        }

        variable_t v;

        v.space = reg_space;
        v.type = unsigned_of(8u << i);

        char sym[16];
        int ret = snprintf(sym, sizeof(sym), "__panoptes_u%d_", 8u << i);
        assert(ret < (int) sizeof(sym));

        v.name = sym;
        v.has_suffix = true;
        v.suffix = (int) c;

        scope->variables.push_back(v);
    }

    if (temps.pred > 0) {
        variable_t v;
        v.space = reg_space;
        v.type = pred_type;
        v.name = "__panoptes_pred_";
        v.has_suffix = true;
        v.suffix = (int) temps.pred;
        scope->variables.push_back(v);
    }

    if (temps.ptr > 0) {
        variable_t v;
        v.space = reg_space;
        v.type  = upointer_type();
        v.name = "__panoptes_ptr";
        v.has_suffix = true;
        v.suffix = (int) temps.ptr;
        scope->variables.push_back(v);
    }
}

cudaError_t global_context_memcheck::cudaDeviceDisablePeerAccess(
        int peerDevice) {
    if (peerDevice < 0 || (unsigned) peerDevice >= devices_) {
        return cudaErrorInvalidDevice;
    }

    scoped_lock lock(mx_);
    const cudaError_t ret =
        global_context::cudaDeviceDisablePeerAccess(peerDevice);
    if (ret == cudaSuccess) {
        state_->disable_peers(static_cast<int>(current_device()), peerDevice);
    }

    return ret;
}

cudaError_t global_context_memcheck::cudaDeviceEnablePeerAccess(int peerDevice_,
        unsigned int flags) {
    const unsigned peerDevice = static_cast<unsigned>(peerDevice_);
    if (peerDevice_ < 0 || peerDevice >= devices_) {
        return cudaErrorInvalidDevice;
    }

    scoped_lock lock(mx_);
    const cudaError_t ret =
        global_context::cudaDeviceEnablePeerAccess(peerDevice_, flags);
    if (ret == cudaSuccess) {
        /* Initialize the contexts for the current device and peerDevice. */
        (void) context_impl(current_device());
        (void) context_impl(peerDevice);
        state_->enable_peers(static_cast<int>(current_device()), peerDevice_);
    }

    return ret;
}

void global_context_memcheck::cudaRegisterVar(void **fatCubinHandle,
        char *hostVar, char *deviceAddress, const char *deviceName,
        int ext, int size, int constant, int global) {
    global_context::cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress,
        deviceName, ext, size, constant, global);

    /* Examine the variable definition and check whether it was initialized. */
    scoped_lock lock(mx_);
    variable_handle_t h(fatCubinHandle, deviceName);
    variable_definition_map_t::iterator it =
        variable_definitions_.find(h);
    if (it == variable_definitions_.end()) {
        /* Attempting to register a variable that does not exist.  This should
         * have been caught at the cuda_context level, so return quietly. */
        return;
    }

    it->second.hostVar = hostVar;
}

state_ptr_t global_context_memcheck::state() {
    return state_;
}
