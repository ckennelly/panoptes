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

struct instrumentation_t {
    instrumentation_t();

    enum error_type_t {
        no_error,
        wild_branch,
        wild_prefetch,
        misaligned_prefetch,
        outofbounds_prefetch_global,
        outofbounds_prefetch_local
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

}
}

#endif // __PANOPTES__CONTEXT_MEMCHECK_INTERNAL_H__
