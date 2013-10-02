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

#ifndef __PANOPTES__PTX_GRAMMAR_H__
#define __PANOPTES__PTX_GRAMMAR_H__

#include <stdint.h>

namespace panoptes {
    /* Forward declarations for ptx_grammar.tab.hh. */
    class ptx_lexer;
    class ptx_parser_state;

    #include <ptx_io/ptx_grammar.tab.hh>
}

#endif // __PANOPTES__PTX_GRAMMAR_H__
