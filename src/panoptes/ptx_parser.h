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

#ifndef __PANOPTES__PTX_PARSER_H__
#define __PANOPTES__PTX_PARSER_H__

#include <string>
#include <vector>
#include "ptx_ir.h"

namespace panoptes {

/* Forward declaration */
class ptx_parser_state;

class ptx_parser {
public:
    ptx_parser();
    ~ptx_parser();

    void parse(const std::string & ptx, ptx_t * program) const;
};

} // end namespace panoptes

#endif // __PANOPTES__PTX_PARSER_H__
