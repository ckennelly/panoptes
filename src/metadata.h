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

#ifndef __PANOPTES__METADATA_H_
#define __PANOPTES__METADATA_H_

#include <stdint.h>

namespace panoptes {

enum {
    lg_chunk_bytes  = 16,
    lg_max_memory   = 36
};

/**
 * Contains metadata pertaining to memory initialization.
 */
struct metadata_chunk {
    /* a bit -> accessibility */
    uint8_t  a_data[(1 << lg_chunk_bytes) / (sizeof(uint8_t ) * CHAR_BIT)];

    /* v bits -> validity
     *
     * Stored as: vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
     */
    uint64_t v_data [(1 << lg_chunk_bytes) /  sizeof(uint64_t)];
};

} // end namespace panoptes

#endif // __PANOPTES__METADATA_H_
