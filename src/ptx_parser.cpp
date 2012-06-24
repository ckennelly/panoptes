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

#include <boost/lexical_cast.hpp>
#include <cassert>
#include <cstdio>

#undef yyFlexLexer
#define yyFlexLexer ptxFlexLexer
#include <FlexLexer.h>
#include "ptx_lexer.h"
#include "ptx_parser.h"
#include "ptx_parser_state.h"

using namespace std;
using namespace panoptes;

namespace panoptes {
extern int yyparse(ptx_lexer * lexer, ptx_parser_state * state);
}

ptx_parser::ptx_parser() { }

ptx_parser::~ptx_parser() { }

void ptx_parser::parse(const std::string & ptx, ptx_t * program) const {
    stringstream in(ptx), out;

    ptx_lexer lexer(&in, &out);
    ptx_parser_state state;

    int ret = yyparse(&lexer, &state);
    assert(ret == 0);

    state.to_ir(program);
}
