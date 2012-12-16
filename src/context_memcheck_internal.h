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

#ifndef __PANOPTES__CONTEXT_MEMCHECK_INTERNAL_H__
#define __PANOPTES__CONTEXT_MEMCHECK_INTERNAL_H__

#include "ptx_ir.h"

namespace panoptes {
namespace internal {

static const size_t max_errors        = 1u << 8;
extern const char * __master_symbol;
extern const char * __texture_prefix;

struct instrumentation_t {
    instrumentation_t();

    enum error_type_t {
        no_error,
        wild_branch,
        wild_prefetch,
        misaligned_prefetch,
        outofbounds_prefetch_global,
        outofbounds_prefetch_local,
        outofbounds_atomic_shared,
        outofbounds_atomic_global,
        outofbounds_ld_global,
        outofbounds_ld_local,
        outofbounds_ld_shared,
        outofbounds_st_global,
        outofbounds_st_local,
        outofbounds_st_shared,
        wild_texture
    };

    struct error_desc_t {
        error_type_t type;
        statement_t orig;
    };

    /**
     * Mapping of each error code (offset by 1) to some metadata to give
     * a better description.
     */
    std::vector<error_desc_t> errors;

    /**
     * Set of unchecked predicate operations.
     */
    typedef std::set<std::string> sset_t;
    sset_t unchecked;
};

/* Forward declaration. */
class auxillary_t;

class temp_operand : boost::noncopyable {
public:
    temp_operand(auxillary_t * parent, type_t type);
    ~temp_operand();

    operator const std::string &() const;
    operator const operand_t &() const;

    bool operator!=(const operand_t & rhs) const;
private:
    auxillary_t & parent_;
    type_t        type_;
    unsigned      id_;

    const std::string identifier_;
    const operand_t   operand_;
};

class temp_ptr : boost::noncopyable {
public:
    temp_ptr(auxillary_t * parent);
    ~temp_ptr();

    operator const std::string &() const;
    operator const operand_t &() const;

    bool operator!=(const operand_t & rhs) const;
private:
    auxillary_t & parent_;
    unsigned      id_;

    const std::string identifier_;
    const operand_t   operand_;
};

class auxillary_t {
public:
    auxillary_t();

    unsigned allocate(type_t type);
    unsigned allocate_ptr();
    void deallocate(type_t type, unsigned id);
    void deallocate_ptr(unsigned id);

    /* These are high water marks. */
    unsigned b[4];
    unsigned s[4];
    unsigned u[4];
    unsigned pred;
    unsigned ptr;

    instrumentation_t * inst;
    block_t * block;
private:
    /* These are the number in actual use. */
    unsigned b_[4];
    unsigned s_[4];
    unsigned u_[4];
    unsigned pred_;
    unsigned ptr_;
};

}
}

#endif // __PANOPTES__CONTEXT_MEMCHECK_INTERNAL_H__
