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

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>
#include <cassert>
#include <cstdio>

#undef yyFlexLexer
#define yyFlexLexer ptxFlexLexer
#include <FlexLexer.h>
#include <ptx_io/ptx_lexer.h>
#include <ptx_io/ptx_parser.h>
#include <ptx_io/ptx_parser_state.h>

using namespace std;
using namespace panoptes;

ptx_parser_exception::ptx_parser_exception(const std::string & ptx,
    int line, int col, const ptx_token & token) : ptx_(ptx), line_(line),
    col_(col), token_(token) { }

ptx_parser_exception::~ptx_parser_exception() throw() { }

const char *ptx_parser_exception::what() const throw() {
    return "Parse error";
}

std::string ptx_parser_exception::detail() const {
    std::vector<std::string> lines;
    boost::split(lines, ptx_, boost::is_any_of("\r\n"));
    const int n_lines = static_cast<int>(lines.size());

    stringstream ss;
    ss << "Parse error.  Unexpected " << token_ << " token" << std::endl;

    /* The lines reported by the lexer are 1-indexed. */
    int actual_line = line_ - 1;

    static const int context = 5;
    int first_line = std::max(0, actual_line - context);
    int last_line  = std::min(n_lines - 1, actual_line + context);

    for (int line_number = first_line; line_number <= last_line;
            line_number++) {
        const std::string & line = lines[size_t(line_number)];
        const int n_columns = static_cast<int>(line.size());

        ss << line << std::endl;
        if (line_number == actual_line) {
            /* Add context. */
            if (col_ > n_columns) {
                /* The column number doesn't line up. */
                ss << std::string(line.size(), '~') << std::endl;
            } else {
                ss << std::string(size_t(col_), ' ') + "^~~~~~" << std::endl;
            }
        }
    }

    return ss.str();
}

ptx_parser::ptx_parser() { }

ptx_parser::~ptx_parser() { }

void ptx_parser::parse(const std::string & ptx, ptx_t * program) const {
    stringstream in(ptx), out;

    ptx_lexer lexer(&in, &out);
    ptx_parser_state state;

    int ret = yyparse(&lexer, &state);
    if (ret != 0) {
        assert(state.error_encountered);

        throw ptx_parser_exception(ptx, state.error_location_line,
            state.error_location_column, state.last_token);
    }

    state.to_ir(program);
}
