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

#include <cassert>
#include <iomanip>
#include "ptx_formatter.h"

using namespace std;
using namespace panoptes;

ostream & operator<<(ostream & o, const approximation_t & a) {
    switch (a) {
        case default_approximation: return o;
        case approximate:           return o << ".approx";
        case full_approximation:    return o << ".full";
    }

    return o;
}

ostream & operator<<(ostream & o, const atom_op_t & a) {
    switch (a) {
        case atom_and:  return o << ".and";
        case atom_or:   return o << ".or";
        case atom_xor:  return o << ".xor";
        case atom_inc:  return o << ".inc";
        case atom_dec:  return o << ".dec";
        case atom_add:  return o << ".add";
        case atom_min:  return o << ".min";
        case atom_max:  return o << ".max";
        case atom_cas:  return o << ".cas";
        case atom_exch: return o << ".exch";
        case atom_invalid:
            assert(0 && "Unknown atomic operation type.");
            return o;
    }

    return o;
}

ostream & operator<<(ostream & o, const barrier_op_t & t) {
    switch (t) {
        case barrier_sync:      return o << ".sync";
        case barrier_arrive:    return o << ".arrive";
        case barrier_reduce:    return o << ".red";
        case barrier_invalid:
            assert(0 && "Unknown barrier type.");
            return o;
    }

    __builtin_unreachable();
}

ostream & operator<<(ostream & o, const barrier_scope_t & t) {
    switch (t) {
        case barrier_cta:   return o << ".cta";
        case barrier_gl:    return o << ".gl";
        case barrier_sys:   return o << ".sys";
        case invalid_barrier_scope:
            assert(0 && "Unknown barrier type.");
            return o;
    }

    __builtin_unreachable();
}

ostream & operator<<(ostream & o, const geom_t & g) {
    switch (g) {
        case geom_1d:       return o << ".1d";
        case geom_2d:       return o << ".2d";
        case geom_3d:       return o << ".3d";
        case geom_a1d:      return o << ".a1d";
        case geom_a2d:      return o << ".a2d";
        case geom_cube:     return o << ".cube";
        case geom_acube:    return o << ".acube";
        case geom_invalid:
            assert(0 && "Unknown geometry type.");
            return o;
    }

    __builtin_unreachable();
}

ostream & operator<<(ostream & o, const type_t & t) {
    switch (t) {
        case b8_type:
            return o << ".b8";
        case b16_type:
            return o << ".b16";
        case b32_type:
            return o << ".b32";
        case b64_type:
            return o << ".b64";
        case u8_type:
            return o << ".u8";
        case u16_type:
            return o << ".u16";
        case u32_type:
            return o << ".u32";
        case u64_type:
            return o << ".u64";
        case s8_type:
            return o << ".s8";
        case s16_type:
            return o << ".s16";
        case s32_type:
            return o << ".s32";
        case s64_type:
            return o << ".s64";
        case f16_type:
            return o << ".f16";
        case f32_type:
            return o << ".f32";
        case f64_type:
            return o << ".f64";
        case pred_type:
            return o << ".pred";
        case invalid_type:
        default:
            assert(0 && "Invalid type.");
            return o;
    }
}

ostream & operator<<(ostream & o, const vector_t & v) {
    switch (v) {
        case v1:
            return o /* << ".v1" */;
        case v2:
            return o << ".v2";
        case v3:
            return o << ".v3";
        case v4:
            return o << ".v4";
        default:
            assert(0 && "Invalid vector type.");
            return o;
    }
}

ostream & operator<<(ostream & o, const space_t & s) {
    switch (s) {
        case const_space:
            return o << ".const";
        case global_space:
            return o << ".global";
        case local_space:
            return o << ".local";
        case shared_space:
            return o << ".shared";
        case generic_space:
            return o << ".generic";
        case param_space:
            return o << ".param";
        case reg_space:
            return o << ".reg";
        case invalid_space:
            assert(0 && "Invalid space.");
            return o;
    }

    __builtin_unreachable();
}

ostream & operator<<(ostream & o, const cache_t & c) {
    switch (c) {
        case cache_ca: return o << ".ca";
        case cache_cg: return o << ".cg";
        case cache_cs: return o << ".cs";
        case cache_cv: return o << ".cv";
        case cache_lu: return o << ".lu";
        case cache_wb: return o << ".wb";
        case cache_wt: return o << ".wt";
        case cache_default: return o;
        case cache_invalid:
        default:
            assert(0 && "Unknown cache type.");
            return o;
    }
}

ostream & operator<<(ostream & o, const panoptes::op_set_cmp_t & cmp) {
    switch (cmp) {
        case cmp_eq:  return o << ".eq";
        case cmp_ne:  return o << ".ne";
        case cmp_lt:  return o << ".lt";
        case cmp_le:  return o << ".le";
        case cmp_gt:  return o << ".gt";
        case cmp_ge:  return o << ".ge";
        case cmp_lo:  return o << ".lo";
        case cmp_ls:  return o << ".ls";
        case cmp_hi:  return o << ".hi";
        case cmp_hs:  return o << ".hs";
        case cmp_equ: return o << ".equ";
        case cmp_neu: return o << ".neu";
        case cmp_ltu: return o << ".ltu";
        case cmp_leu: return o << ".leu";
        case cmp_gtu: return o << ".gtu";
        case cmp_geu: return o << ".geu";
        case cmp_num: return o << ".num";
        case cmp_nan: return o << ".nan";
        case cmp_invalid:
        default:
            assert(0 && "Unknown comparison operator.");
            return o;
    }
}

ostream & operator<<(ostream & o, const panoptes::op_set_bool_t & bool_op) {
    switch (bool_op) {
        case bool_none: return o;
        case bool_and:  return o << ".and";
        case bool_or:   return o << ".or";
        case bool_popc: return o << ".popc";
        case bool_xor:  return o << ".xor";
    }

    return o;
}

ostream & operator<<(ostream & o, const statement_t & s) {
    if (s.has_predicate) {
        o << "@";

        if (s.is_negated) {
            o << "!";
        }

        assert(s.predicate.size() > 0);
        o << s.predicate << " ";
    }

    o << s.op;
    switch (s.op) {
        case op_abs:
            assert(s.operands.size() == 2u);
            if (s.ftz) {
                assert(s.type == f32_type);
                o << ".ftz";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_add:
            assert(s.operands.size() == 3u);
            assert(!(s.saturating && s.carry_out));
            if (s.saturating) {
                o << ".sat";
            } else if (s.carry_out) {
                assert(s.type == u32_type || s.type == s32_type);
                o << ".cc";
            }
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_addc:
            assert(s.operands.size() == 3u);
            assert(s.type == u32_type || s.type == s32_type);
            assert(!(s.saturating));

            if (s.carry_out) {
                o << ".cc";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_and:
            assert(s.operands.size() == 3u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_atom:
            assert(s.operands.size() >= 3u);
            assert(s.operands.size() <= 4u);
            if (s.space != generic_space) {
                o << s.space;
            }

            o << s.atomic_op << s.type << " " << s.operands[0] << ", [" <<
                s.operands[1] << "], " << s.operands[2];

            if (s.operands.size() == 4u) {
                o << ", " << s.operands[3];
            }
            break;
        case op_bar:
            o << s.barrier;
            switch (s.barrier) {
                case barrier_sync:
                    assert(s.operands.size() >= 1u);
                    assert(s.operands.size() <= 2u);
                    o << " " << s.operands[0];
                    if (s.operands.size() > 1u) {
                        o << ", " << s.operands[1];
                    }

                    break;
                case barrier_arrive:
                    assert(0 && "Not implemented.");
                    break;
                case barrier_reduce:
                    assert(s.operands.size() >= 3u);
                    assert(s.operands.size() <= 4u);
                    o << s.bool_op << s.type << " " << s.operands[0] << ", " <<
                        s.operands[1] << ", ";

                    if (s.operands.size() > 3u) {
                        o << s.operands[2] << ", ";
                        if (s.is_negated) {
                            o << "!";
                        }
                        o << s.operands[3];
                    } else {
                        if (s.is_negated) {
                            o << "!";
                        }
                        o << s.operands[2];
                    }
                    break;
                case barrier_invalid:
                    assert(0 && "Unknown barrier type.");
                    break;
            }
            break;
        case op_bfe:
            assert(s.operands.size() == 4u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2] << ", " << s.operands[3];
            break;
        case op_bfi:
            assert(s.operands.size() == 5u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2] << ", " << s.operands[3] <<
                ", " << s.operands[4];
            break;
        case op_bfind:
            assert(s.operands.size() == 2u);
            if (s.shiftamt) {
                o << ".shiftamt";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_bra:
            assert(s.operands.size() == 1u);
            if (s.uniform) {
                o << ".uni";
            }
            o << " " << s.operands[0];
            break;
        case op_brev:
            assert(s.operands.size() == 2u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_brkpt: break;
        case op_call:
            assert(s.operands.size() >= 1u);
            if (s.uniform) {
                o << ".uni";
            }

            o << " ";

            if (s.has_return_value) {
                o << "(" << s.operands[0] << "), ";
            }

            o << s.operands[s.has_return_value ? 1u : 0];

            if (s.operands.size() > (s.has_return_value ? 2u : 1u)) {
                o << ", (";

                bool first = true;
                for (size_t i = s.has_return_value ? 2u : 1u;
                        i < s.operands.size(); i++) {
                    if (first) {
                        first = false;
                    } else {
                        o << ", ";
                    }

                    o << s.operands[i];
                }

                o << ")";
            }

            break;
        case op_clz:
            assert(s.operands.size() == 2u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_cnot:
            assert(s.operands.size() == 2u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_copysign:
            assert(s.operands.size() == 3u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_cos:
            assert(s.operands.size() == 2u);
            assert(s.approximation == approximate);
            o << ".approx";
            if (s.ftz) {
                o << ".ftz";
            }
            assert(s.type == f32_type);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_cvt:
            assert(s.operands.size() == 2u);
            o << s.rounding;
            if (s.ftz) {
                o << ".ftz";
            }
            if (s.saturating) {
                o << ".sat";
            }
            o << s.type << s.type2 << " " << s.operands[0] << ", " <<
                s.operands[1];
            break;
        case op_cvta:
            assert(s.operands.size() == 2u);
            if (s.is_to) {
                o << ".to";
            }

            o << s.space << s.type << " " << s.operands[0] << ", " <<
                s.operands[1];
            break;
        case op_div:
            assert(s.operands.size() == 3u);
            switch (s.type) {
                case f32_type:
                case f64_type:
                    switch (s.approximation) {
                        case approximate:
                            o << ".approx";
                            break;
                        case default_approximation:
                            o << s.rounding;
                            break;
                        case full_approximation:
                            o << ".full";
                            break;
                    }

                    if (s.ftz) {
                        assert(s.type != f64_type);
                        o << ".ftz";
                    }

                    // Fall through
                case s16_type:
                case s32_type:
                case s64_type:
                case u16_type:
                case u32_type:
                case u64_type:
                    o << s.type << " " << s.operands[0] << ", " <<
                        s.operands[1] << ", " << s.operands[2];
                    break;
                default:
                    assert(0 && "Unknown type.");
                    break;
            }
            break;
        case op_ex2:
            assert(s.operands.size() == 2u);
            o << s.approximation;
            if (s.ftz) {
                o << ".ftz";
            }
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_exit: break;
        case op_fma:
            assert(s.operands.size() == 4u);
            assert(s.type == f32_type || s.type == f64_type);

            o << s.rounding;
            if (s.type == f32_type) {
                if (s.ftz) {
                    o << ".ftz";
                }

                if (s.saturating) {
                    o << ".sat";
                }
            }
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2] << ", " << s.operands[3];
            break;
        case op_isspacep:
            assert(s.operands.size() == 2u);
            o << s.space << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_ld:
            assert(s.operands.size() == 2u);
            if (s.is_volatile) {
                o << ".volatile";
            }
            o << s.space << s.cache << s.vector << s.type << " " <<
                s.operands[0] << ", [" << s.operands[1] << "]";
            break;
        case op_ldu:
            assert(s.operands.size() == 2u);
            o << s.space << s.vector << s.type << " " << s.operands[0] <<
                ", [" << s.operands[1] << "]";
            break;
        case op_lg2:
            assert(s.operands.size() == 2u);
            o << s.approximation;
            if (s.ftz) {
                o << ".ftz";
            }
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_mad:
            assert(s.operands.size() == 4u);
            switch (s.type) {
                /* Integer type. */
                case s16_type:
                case s32_type:
                case s64_type:
                case u16_type:
                case u32_type:
                case u64_type:
                    o << s.width;
                    assert(!(s.saturating && s.carry_out));
                    if (s.saturating) {
                        assert(s.type == s32_type);
                        o << ".sat";
                    } else if (s.carry_out) {
                        assert(s.type == u32_type || s.type == s32_type);
                        o << ".cc";
                    }

                    o << s.type << " " << s.operands[0] << ", " <<
                        s.operands[1] << ", " << s.operands[2] << ", " <<
                        s.operands[3];
                    break;
                /* Floating point type. */
                case f32_type:
                case f64_type:
                    o << s.rounding;
                    assert(!(s.carry_out));
                    if (s.type == f32_type) {
                        if (s.ftz) {
                            o << ".ftz";
                        }

                        if (s.saturating) {
                            o << ".sat";
                        }
                    }
                    o << s.type << " " << s.operands[0] << ", " <<
                        s.operands[1] << ", " << s.operands[2] << ", " <<
                        s.operands[3];
                    break;
                case b8_type:
                case b16_type:
                case b32_type:
                case b64_type:
                case f16_type:
                case invalid_type:
                case pred_type:
                case texref_type:
                case s8_type:
                case u8_type:
                    assert(0 && "Invalid type.");
                    break;
            }
            break;
        case op_mad24:
            assert(s.operands.size() == 4u);
            switch (s.type) {
                /* Integer type. */
                case s32_type:
                case u32_type:
                    o << s.width;
                    if (s.saturating) {
                        assert(s.type == s32_type);
                        o << ".sat";
                    }

                    o << s.type << " " << s.operands[0] << ", " <<
                        s.operands[1] << ", " << s.operands[2] << ", " <<
                        s.operands[3];
                    break;
                case b8_type:
                case b16_type:
                case b32_type:
                case b64_type:
                case s16_type:
                case s64_type:
                case u16_type:
                case u64_type:
                case f16_type:
                case f32_type:
                case f64_type:
                case invalid_type:
                case pred_type:
                case texref_type:
                case s8_type:
                case u8_type:
                    assert(0 && "Invalid type.");
                    break;
            }
            break;
        case op_madc:
            assert(s.operands.size() == 4u);
            assert(s.type == u32_type || s.type == s32_type);
            assert(!(s.saturating));

            o << s.width;
            if (s.carry_out) {
                o << ".cc";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2] << ", " << s.operands[3];
            break;
        case op_max:
            assert(s.operands.size() == 3u);
            if (s.ftz) {
                assert(s.type == f32_type || s.type == f64_type);
                o << ".ftz";
            }
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_membar:
            assert(s.operands.size() == 0u);
            o << s.barrier_scope;
            break;
        case op_min:
            assert(s.operands.size() == 3u);
            if (s.ftz) {
                assert(s.type == f32_type || s.type == f64_type);
                o << ".ftz";
            }
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_mov:
            assert(s.operands.size() == 2u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_mul:
            assert(s.operands.size() == 3u);
            if (s.type == f32_type || s.type == f64_type) {
                o << s.rounding;
                if (s.ftz) {
                    o << ".ftz";
                }
                if (s.saturating) {
                    o << ".sat";
                }
                o << s.type << " " << s.operands[0] << ", " << s.operands[1]
                    << ", " << s.operands[2];
            } else {
                o << s.width << s.type << " " << s.operands[0] << ", " <<
                    s.operands[1] << ", " << s.operands[2];
            }
            break;
        case op_mul24:
            assert(s.operands.size() == 3u);
            assert(s.type == u32_type || s.type == s32_type);
            o << s.width << s.type << " " << s.operands[0] << ", " <<
                s.operands[1] << ", " << s.operands[2];
            break;
        case op_neg:
            assert(s.operands.size() == 2u);
            if (s.type == f32_type && s.ftz) {
                o << ".ftz";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_not:
            assert(s.operands.size() == 2u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_or:
            assert(s.operands.size() == 3u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_pmevent:
            assert(s.operands.size() == 1u);
            if (s.mask) {
                o << ".mask";
            }
            o << " " << s.operands[0];
            break;
        case op_prefetch:
        case op_prefetchu:
            assert(s.operands.size() == 1u);
            if (s.space != generic_space) {
                o << s.space;
            }
            o << s.prefetch_cache << " [" << s.operands[0] << "]";
            break;
        case op_prmt:
            assert(s.operands.size() == 4u);
            o << s.type << s.prmt_mode << " " << s.operands[0] << ", " <<
                s.operands[1] << ", " << s.operands[2] << ", " <<
                s.operands[3];
            break;
        case op_popc:
            assert(s.operands.size() == 2u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_rcp:
            assert(s.operands.size() == 2u);
            if (s.approximation == approximate) {
                o << ".approx";
            } else {
                o << s.rounding;
            }

            if (s.ftz) {
                o << ".ftz";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_red:
            assert(s.operands.size() == 2u);
            if (s.space != generic_space) {
                o << s.space;
            }

            o << s.atomic_op << s.type << " [" << s.operands[0] << "], " <<
                s.operands[1];
            break;
        case op_rem:
            assert(s.operands.size() == 3u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_ret:
            assert(s.operands.size() == 0);
            if (s.uniform) {
                o << ".uni";
            }

            break;
        case op_rsqrt:
            assert(s.operands.size() == 2u);
            assert(s.type == f32_type || s.type == f64_type);
            o << s.approximation;

            if (s.ftz) {
                assert(s.type == f32_type);
                o << ".ftz";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_sad:
            assert(s.operands.size() == 4u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2] << ", " << s.operands[3];
            break;
        case op_selp:
            assert(s.operands.size() == 4u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2] << ", " << s.operands[3];
            break;
        case op_set:
            assert(s.operands.size() >= 3u);

            o << s.cmp << s.bool_op;
            if (s.ftz) {
                o << ".ftz";
            }
            o << s.type << s.type2;

            o << " " << s.operands[0] << ", " << s.operands[1] << ", " <<
                s.operands[2];
            if (s.operands.size() > 3u) {
                o << ", " << s.operands[3];
            }
            break;
        case op_setp:
            assert(s.operands.size() >= 2u);

            o << s.cmp << s.bool_op;
            if (s.ftz) {
                o << ".ftz";
            }
            o << s.type;

            assert(s.has_ppredicate);
            o << " " << s.ppredicate;

            if (s.has_qpredicate) {
                o << "|" << s.qpredicate;
            }

            o << ", " << s.operands[0] << ", " << s.operands[1];
            if (s.operands.size() > 2u) {
                o << ", " << s.operands[2];
            }
            break;
        case op_shl:
            assert(s.operands.size() == 3u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_shr:
            assert(s.operands.size() == 3u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_sin:
            assert(s.operands.size() == 2u);
            assert(s.approximation == approximate);
            o << ".approx";
            if (s.ftz) {
                o << ".ftz";
            }
            assert(s.type == f32_type);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
      case op_slct:
            assert(s.operands.size() == 4u);

            /**
             * ftz implies f32_type
             */
            assert(!(s.ftz) || s.type2 == f32_type);
            if (s.ftz) {
                o << ".ftz";
            }

            o << s.type << s.type2 << " " << s.operands[0] << ", " <<
                s.operands[1] << ", " << s.operands[2] << ", " <<
                s.operands[3];
            break;
        case op_sqrt:
            assert(s.operands.size() == 2u);
            assert(s.type == f32_type || s.type == f64_type);
            if (s.type == f32_type && s.approximation == approximate) {
                o << ".approx";
            } else {
                o << s.rounding;
            }

            if (s.ftz) {
                assert(s.type == f32_type);
                o << ".ftz";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1];
            break;
        case op_st:
            assert(s.operands.size() == 2u);
            if (s.is_volatile) {
                o << ".volatile";
            }
            o << s.space << s.cache << s.vector << s.type << " [" <<
                s.operands[0] << "], " << s.operands[1];
            break;
        case op_sub:
            assert(s.operands.size() == 3u);
            assert(!(s.saturating && s.carry_out));
            if (s.saturating) {
                assert(s.type == s32_type);
                o << ".sat";
            } else if (s.carry_out) {
                assert(s.type == u32_type || s.type == s32_type);
                o << ".cc";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_subc:
            assert(s.operands.size() == 3u);
            assert(s.type == u32_type || s.type == s32_type);
            assert(!(s.saturating));

            if (s.carry_out) {
                o << ".cc";
            }

            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_testp:
            assert(s.operands.size() == 2u);
            o << s.testp_op << s.type << " " << s.operands[0] << ", " <<
                s.operands[1];
            break;
        case op_tex:
            assert(s.operands.size() >= 3u);
            assert(s.operands.size() <= 4u);

            o << s.geometry << s.vector << s.type << s.type2 << " " <<
                s.operands[0] << ", [" << s.operands[1] << ", " <<
                s.operands[2];

            if (s.operands.size() == 4u) {
                o << ", " << s.operands[3];
            }

            o << "]";
            break;
        case op_txq:
            assert(s.operands.size() == 2u);
            o << s.query << s.type << " " << s.operands[0] << ", [" <<
                s.operands[1] << "]";
            break;
        case op_trap: break;
        case op_vote:
            assert(s.operands.size() == 2u);
            o << s.vote_mode << s.type << " " << s.operands[0] << ", ";
            if (s.is_negated) {
                o << "!";
            }
            o << s.operands[1];
            break;
        case op_xor:
            assert(s.operands.size() == 3u);
            o << s.type << " " << s.operands[0] << ", " << s.operands[1] <<
                ", " << s.operands[2];
            break;
        case op_invalid:
            assert(0 && "Unknown opcode.");
            break;
    }

    return o << ";" << endl;
}

ostream & operator<<(ostream & o, const variable_t & v) {
    o << v.linkage << v.space << " ";
    if (v.has_align) {
        o << ".align " << v.alignment << " ";
    }

    if (v.vector_size != v1) {
        o << v.vector_size << " ";
    }
    o << v.type << " " << v.name;

    if (v.suffix > 0) {
        o << "<" << v.suffix << ">";
    }

    if (v.is_array) {
        assert(v.array_dimensions <= 3u);
        for (unsigned i = 0; i < v.array_dimensions; i++) {
            o << "[";
            if (i == v.array_dimensions - 1u && v.array_flexible) {
                /* Do nothing. */
            } else {
                o << v.array_size[i];
            }
            o << "]";
        }
    }

    if (v.has_initializer) {
        o << " = ";
        if (v.initializer_vector) {
            o << "{";
            bool first = true;
            for (size_t i = 0; i < v.initializer.size(); i++) {
                if (first) {
                    first = false;
                } else {
                    o << ", ";
                }

                o << v.initializer[i];
            }
            o << "}";
        } else {
            assert(v.initializer.size() == 1u);
            o << v.initializer[0];
        }
    }

    return o << ";" << endl;
}

ostream & operator<<(ostream & o, const block_t & b) {
    switch (b.block_type) {
        case block_scope:
            return o << *b.scope;
        case block_statement:
            return o << *b.statement;
        case block_label:
            return o << *b.label;
        case block_invalid:
        default:
            assert(0 && "Invalid block type.");
            return o;
    }
}

ostream & operator<<(ostream & o, const label_t & b) {
    return o << b.label << ":" << endl;
}

ostream & operator<<(ostream & o, const linkage_t & l) {
    switch (l) {
        case linkage_default: return o;
        case linkage_extern:  return o << ".extern ";
        case linkage_visible: return o << ".visible ";
    }

    assert(0 && "Unknown linkage type.");
    return o;
}

ostream & operator<<(ostream & o, const scope_t & s) {
    o << "{" << endl;

    for (scope_t::variable_vt::const_iterator it = s.variables.begin();
            it != s.variables.end(); ++it) {
        o << *it;
    }

    for (scope_t::block_vt::const_iterator it = s.blocks.begin();
            it != s.blocks.end(); ++it) {
        o << **it;
    }


    return o << "}" << endl;
}

ostream & operator<<(ostream & o, const param_t & p) {
    o << p.space << " ";
    if (p.has_align) {
        o << ".align " << p.alignment << " ";
    }

    o << p.type;
    if (p.is_ptr) {
        /* TODO:  We need to know the PTX version we are outputing. */
        /* o << ".ptr"; */
    }

    o << " " << p.name;

    if (p.is_array) {
        assert(p.array_dimensions == 1u);
        o << "[" << p.array_size[0] << "]";
    }

    return o;
}

ostream & operator<<(ostream & o, const function_t & f) {
    if (f.entry) {
        o << ".entry ";
    } else {
        o << f.linkage << " .func ";

        if (f.has_return_value) {
            o << "(" << f.return_value << ") ";
        }
    }
    o << f.entry_name << " (";

    for (size_t i = 0; i < f.params.size(); i++) {
        if (i > 0) {
            o << ",";
        }
        o << endl << f.params[i];
    }

    o << ")";

    if (f.no_body) {
        o << ";";
    } else {
        o << endl << f.scope;
    }

    return o << endl;
}

ostream & operator<<(ostream & o, const sm_t & sm) {
    switch (sm) {
        case SM10:
            return o << "sm_10";
        case SM11:
            return o << "sm_11";
        case SM12:
            return o << "sm_12";
        case SM13:
            return o << "sm_13";
        case SM20:
            return o << "sm_20";
        case SM21:
            return o << "sm_21";
        case DEBUG:
            return o << "debug";
        default:
            assert(0 && "Invalid shader model.");
            return o;
    }
}

ostream & operator<<(ostream & o, const op_t & op) {
    switch (op) {
        case op_abs:        return o << "abs";
        case op_add:        return o << "add";
        case op_addc:       return o << "addc";
        case op_and:        return o << "and";
        case op_atom:       return o << "atom";
        case op_bar:        return o << "bar";
        case op_bfe:        return o << "bfe";
        case op_bfi:        return o << "bfi";
        case op_bfind:      return o << "bfind";
        case op_bra:        return o << "bra";
        case op_brev:       return o << "brev";
        case op_brkpt:      return o << "brkpt";
        case op_call:       return o << "call";
        case op_clz:        return o << "clz";
        case op_cnot:       return o << "cnot";
        case op_copysign:   return o << "copysign";
        case op_cos:        return o << "cos";
        case op_cvt:        return o << "cvt";
        case op_cvta:       return o << "cvta";
        case op_div:        return o << "div";
        case op_ex2:        return o << "ex2";
        case op_exit:       return o << "exit";
        case op_fma:        return o << "fma";
        case op_isspacep:   return o << "isspacep";
        case op_ld:         return o << "ld";
        case op_ldu:        return o << "ldu";
        case op_lg2:        return o << "lg2";
        case op_mad:        return o << "mad";
        case op_mad24:      return o << "mad24";
        case op_madc:       return o << "madc";
        case op_max:        return o << "max";
        case op_membar:     return o << "membar";
        case op_min:        return o << "min";
        case op_mov:        return o << "mov";
        case op_mul:        return o << "mul";
        case op_mul24:      return o << "mul24";
        case op_neg:        return o << "neg";
        case op_not:        return o << "not";
        case op_or:         return o << "or";
        case op_pmevent:    return o << "pmevent";
        case op_popc:       return o << "popc";
        case op_prefetch:   return o << "prefetch";
        case op_prefetchu:  return o << "prefetchu";
        case op_prmt:       return o << "prmt";
        case op_rcp:        return o << "rcp";
        case op_red:        return o << "red";
        case op_rem:        return o << "rem";
        case op_ret:        return o << "ret";
        case op_rsqrt:      return o << "rsqrt";
        case op_sad:        return o << "sad";
        case op_selp:       return o << "selp";
        case op_set:        return o << "set";
        case op_setp:       return o << "setp";
        case op_shl:        return o << "shl";
        case op_shr:        return o << "shr";
        case op_sin:        return o << "sin";
        case op_slct:       return o << "slct";
        case op_sqrt:       return o << "sqrt";
        case op_st:         return o << "st";
        case op_sub:        return o << "sub";
        case op_subc:       return o << "subc";
        case op_suld:       return o << "suld";
        case op_suq:        return o << "suq";
        case op_sured:      return o << "sured";
        case op_sust:       return o << "sust";
        case op_testp:      return o << "testp";
        case op_tex:        return o << "tex";
        case op_tld4:       return o << "tld4";
        case op_trap:       return o << "trap";
        case op_txq:        return o << "txq";
        case op_vabsdiff:   return o << "vabsdiff";
        case op_vadd:       return o << "vadd";
        case op_vmad:       return o << "vmad";
        case op_vmax:       return o << "vmax";
        case op_vmin:       return o << "vmin";
        case op_vote:       return o << "vote";
        case op_vset:       return o << "vset";
        case op_vshl:       return o << "vshl";
        case op_vshr:       return o << "vshr";
        case op_vsub:       return o << "vsub";
        case op_xor:        return o << "xor";
        case op_invalid:
            assert(0 && "Invalid opcode.");
            return o;
    }

    assert(0 && "Invalid opcode.");
    return o;
}

ostream & operator<<(ostream & o, const rounding_t & r) {
    switch (r) {
        case rounding_default:  return o;
        case rounding_rni:      return o << ".rni";
        case rounding_rzi:      return o << ".rzi";
        case rounding_rmi:      return o << ".rmi";
        case rounding_rpi:      return o << ".rpi";
        case rounding_rn:       return o << ".rn";
        case rounding_rz:       return o << ".rz";
        case rounding_rm:       return o << ".rm";
        case rounding_rp:       return o << ".rp";
        case rounding_invalid:
            assert(0 && "Invalid rounding mode.");
            return o;
    }

    return o;
}

ostream & operator<<(ostream & o, const texture_t & t) {
    if (t.type == texref_type) {
        o << ".global .texref ";

        assert(t.names.size() > 0);
        for (size_t i = 0; i < t.names.size(); i++) {
            if (i > 1u) {
                o << ", ";
            }

            o << t.names[i];
        }

        return o << ";" << endl;
    } else {
        o << ".tex " << t.type << " ";

        assert(t.names.size() > 0);
        for (size_t i = 0; i < t.names.size(); i++) {
            if (i > 1u) {
                o << ", ";
            }

            o << t.names[i];
        }

        return o << ";" << endl;
    }
}

ostream & operator<<(ostream & o, const ptx_t & p) {
    o << ".version " << p.version_major << "." << p.version_minor << endl;
    o << ".target " << p.sm;
    if (p.map_f64_to_f32) {
        o << ", map_f64_to_f32";
    }
    o << endl;
    if (p.version_major > 2 ||
            (p.version_major == 2 && p.version_minor >= 3)) {
        o << ".address_size " << p.address_size << endl;
    }

    for (ptx_t::texture_vt::const_iterator it = p.textures.begin();
            it != p.textures.end(); ++it) {
        o << *it;
    }

    for (ptx_t::variable_vt::const_iterator it = p.variables.begin();
            it != p.variables.end(); ++it) {
        o << *it;
    }

    for (ptx_t::entry_vt::const_iterator it = p.entries.begin();
            it != p.entries.end(); ++it) {
        o << **it;
    }

    return o;
}

ostream & operator<<(ostream & o, const field_t & f) {
    switch (f) {
        case field_x: return o << ".x";
        case field_y: return o << ".y";
        case field_z: return o << ".z";
        case field_w: return o << ".w";
        case field_none: return o;
        default:
            assert(0 && "Unknown field type.");
            return o;
    }
}

namespace {
    static void write_double(ostream & o, double d) {
        union {
            uint64_t u;
            double d;
        } u;

        u.d = d;
        o << "0d" << std::setw(16) << std::setfill('0') << std::hex <<
            u.u << std::dec;
    }

    static void write_float(ostream & o, float f) {
        union {
            uint32_t u;
            float f;
        } u;

        u.f = f;
        o << "0f" << std::setw(8) << std::setfill('0') << std::hex <<
            u.u << std::dec;
    }
}

ostream & operator<<(ostream & o, const operand_t & op) {
    if (op.negated) {
        o << "!";
    }

    switch (op.op_type) {
        case operand_identifier:
            assert(op.identifier.size() > 0);
            assert(op.field.size() == op.identifier.size());
            if (op.identifier.size() > 1u) {
                o << "{";
            }

            for (size_t i = 0; i < op.identifier.size(); i++) {
                if (i > 0) {
                    o << ",";
                }
                o << op.identifier[i] << op.field[i];
            }

            if (op.identifier.size() > 1u) {
                o << "}";
            }
            break;
        case operand_addressable:
            assert(op.identifier.size() == 1u);
            o << op.identifier[0];
            if (op.offset != 0) {
                if (op.offset > 0) {
                    o << "+";
                }

                o << op.offset;
            } else {
                o << "+0";
            }
            break;
        case operand_indexed:
            assert(op.identifier.size() == 1u);
            o << op.identifier[0] << "[" << op.offset << "]";
            break;
        case operand_constant:
            o << op.offset;
            break;
        case operand_double:
            write_double(o, op.dvalue);
            break;
        case operand_float:
            write_float(o, op.fvalue);
            break;
        case invalid_operand:
            assert(0 && "Unknown operand type.");
            break;
    }

    return o;
}

ostream & operator<<(ostream & o, const panoptes::op_mul_width_t & w) {
    switch (w) {
        case width_default: return o;
        case width_hi:      return o << ".hi";
        case width_lo:      return o << ".lo";
        case width_wide:    return o << ".wide";
    }

    __builtin_unreachable();
}

ostream & operator<<(ostream & o, const panoptes::variant_t & v) {
    switch (v.type) {
        case variant_double:
            write_double(o, v.data.d);
            return o;
        case variant_integer:
            return o << v.data.u;
        case variant_single:
            write_float(o, v.data.f);
            return o;
    }

    assert(0 && "Unknown variant type.");
    return o;
}

ostream & operator<<(ostream & o, const panoptes::vote_mode_t & v) {
    switch (v) {
        case invalid_vote_mode:
            assert(0 && "Invalid vote mode type.");
            return o;
        case vote_all:      return o << ".all";
        case vote_any:      return o << ".any";
        case vote_ballot:   return o << ".ballot";
        case vote_uniform:  return o << ".uni";
    }

    return o;
}

ostream & operator<<(ostream & o, const panoptes::testp_op_t & t) {
    switch (t) {
        case invalid_testp_op:
            assert(0 && "Invalid testp op.");
            return o;
        case testp_finite:    return o << ".finite";
        case testp_infinite:  return o << ".infinite";
        case testp_number:    return o << ".number";
        case testp_nan:       return o << ".notanumber";
        case testp_normal:    return o << ".normal";
        case testp_subnormal: return o << ".subnormal";
    }

    return o;
}

ostream & operator<<(ostream & o, const panoptes::prmt_mode_t & m) {
    switch (m) {
        case invalid_prmt_mode:
            assert(0 && "Invalid permute mode.");
            return o;
        case prmt_default:  return o;
        case prmt_f4e:      return o << ".f4e";
        case prmt_b4e:      return o << ".b4e";
        case prmt_rc8:      return o << ".rc8";
        case prmt_ecl:      return o << ".ecl";
        case prmt_ecr:      return o << ".ecr";
        case prmt_rc16:     return o << ".rc16";
    }

    return o;
}

ostream & operator<<(ostream & o, const panoptes::prefetch_cache_t & c) {
    switch (c) {
        case invalid_cache:
            assert(0 && "Invalid cache level.");
            return o;
        case cache_L1: return o << ".L1";
        case cache_L2: return o << ".L2";
    }

    __builtin_unreachable();
}

ostream & operator<<(ostream & o, const panoptes::query_t & q) {
    switch (q) {
        case invalid_query:
            assert(0 && "Invalid query type.");
            return o;
        case query_width:
            return o << ".width";
        case query_height:
            return o << ".height";
        case query_depth:
            return o << ".depth";
        case query_channel_data_type:
            return o << ".channel_data_type";
        case query_channel_order:
            return o << ".channel_order";
        case query_normalized_coords:
            return o << ".normalized_coords";
        case query_force_unnormalized_coords:
            return o << ".force_unnormalized_coords";
        case query_filter_mode:
            return o << ".filter_mode";
        case query_addr_mode0:
            return o << ".addr_mode_0";
        case query_addr_mode1:
            return o << ".addr_mode_1";
        case query_addr_mode2:
            return o << ".addr_mode_2";
    }

    __builtin_unreachable();
}
