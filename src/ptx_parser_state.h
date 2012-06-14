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

#ifndef __PANOPTES__PTX_PARSER_STATE_H__
#define __PANOPTES__PTX_PARSER_STATE_H__

#include <boost/utility.hpp>
#include <deque>
#include <map>
#include "ptx_ir.h"
#include <stack>
#include <string>
#include <vector>

namespace panoptes {

class ptx_parser_statement : boost::noncopyable {
public:
    ptx_parser_statement();
    virtual ~ptx_parser_statement();

    virtual void to_ir(block_t * block) const = 0;
};

class ptx_parser_instruction : public ptx_parser_statement {
public:
    ptx_parser_instruction();
    ~ptx_parser_instruction();

    statement_t ir;
    virtual void to_ir(block_t * block) const;
};

class ptx_parser_label : public ptx_parser_statement {
public:
    ptx_parser_label();
    ~ptx_parser_label();

    std::string name;
    virtual void to_ir(block_t * block) const;
};

class ptx_parser_block : public ptx_parser_statement {
public:
    ptx_parser_block();
    ~ptx_parser_block();

    void finish_label();
    ptx_parser_label * label;

    statement_t instruction;
    void declare_instruction();

    variable_t variable;
    typedef std::vector<variable_t> variable_vt;
    variable_vt variables;
    void finish_variable();

    typedef std::deque<ptx_parser_statement *> statements_t;
    statements_t statements;

    virtual void to_ir(block_t * block) const;
};

class ptx_parser_function : boost::noncopyable {
public:
    ptx_parser_function();
    ~ptx_parser_function();

    void block_close();
    void block_open();

    linkage_t linkage;
    bool entry;
    std::string name;

    bool has_return_value;
    param_t return_value;

    param_t param;
    typedef std::vector<param_t> param_vt;
    param_vt params;
    void finish_param();

    bool no_body;
    ptx_parser_block * root;
    ptx_parser_block * top;
    std::stack<ptx_parser_block *> block_stack;

    void to_ir(function_t * entry) const;
};

class ptx_parser_state : boost::noncopyable {
public:
    ptx_parser_state();
    ~ptx_parser_state();

    unsigned version_major;
    unsigned version_minor;

    void set_target(int token);
    sm_t sm;
    bool map_f64_to_f32;

    operand_t operand;
    typedef std::vector<operand_t> operand_vt;
    operand_vt operands;

    void add_file(int file, const std::string & path);

    int file_number;
    int line_number;
    int address_size;

    void set_linkage(int token);
    linkage_t get_linkage();

    void declare_function(const std::string & name);
    ptx_parser_function * function;

    void declare_variable();
    variable_t variable;

    void set_type(int token);
    /* Returns the type set by set_type.  If subsequent calls to get_type occur
     * without another call to set_type, invalid_type will be returned. */
    type_t get_type();

    void to_ir(ptx_t * program) const;

    void declare_texture();
    texture_t texture;
protected:
    linkage_t linkage;
    type_t type;

    typedef std::map<int, std::string> file_map_t;
    file_map_t files;

    typedef std::vector<ptx_parser_function *> function_vector_t;
    function_vector_t functions;

    typedef std::vector<variable_t> variable_vt;
    variable_vt variables;

    typedef std::vector<texture_t> texture_vt;
    texture_vt textures;
};

}

#endif // __PANOPTES__PTX_PARSER_STATE_H__
