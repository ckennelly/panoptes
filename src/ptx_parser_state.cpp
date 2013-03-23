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
#include <climits>
#include "ptx_grammar.h"
#include "ptx_parser_state.h"
#include <stdint.h>
#include <string>

using namespace panoptes;

ptx_parser_statement::ptx_parser_statement() { }

ptx_parser_statement::~ptx_parser_statement() { }

ptx_parser_instruction::ptx_parser_instruction() { }

ptx_parser_instruction::~ptx_parser_instruction() { }

void ptx_parser_instruction::to_ir(block_t * block) const {
    assert(block->block_type == block_invalid);
    block->block_type = block_statement;

    block->statement = new statement_t(ir);
}

ptx_parser_label::ptx_parser_label() { }

ptx_parser_label::~ptx_parser_label() { }

void ptx_parser_label::to_ir(block_t * block) const {
    assert(block->block_type == block_invalid);
    block->block_type = block_label;

    block->label = new label_t();
    block->label->label = name;
}

ptx_parser_block::ptx_parser_block() :
        label(new ptx_parser_label()) { }

ptx_parser_block::~ptx_parser_block() {
    for (statements_t::iterator it = statements.begin();
            it != statements.end(); ++it) {
        delete *it;
    }
    statements.clear();

    delete label;
}

void ptx_parser_block::declare_instruction() {
    ptx_parser_instruction * ins = new ptx_parser_instruction();
    ins->ir = instruction;
    instruction.reset();

    statements.push_back(ins);
}

void ptx_parser_block::finish_variable() {
    variables.push_back(variable);
    variable = variable_t();
}

void ptx_parser_block::finish_label() {
    statements.push_back(label);
    label = new ptx_parser_label();
}

void ptx_parser_block::to_ir(block_t * block) const {
    assert(block->block_type == block_invalid);
    block->block_type = block_scope;

    scope_t * ret = new scope_t();
    block->scope = ret;

    ret->variables = variables;

    for (statements_t::const_iterator it = statements.begin();
            it != statements.end(); ++it) {
        block_t * child = new block_t();
        child->parent = block;
        (*it)->to_ir(child);

        ret->blocks.push_back(child);
    }
}

ptx_parser_function::ptx_parser_function() : linkage(linkage_default),
        has_return_value(false), no_body(false), root(NULL), top(root) {
    block_stack.push(root);
}

ptx_parser_function::~ptx_parser_function() {
    delete root;
}

void ptx_parser_function::block_close() {
    block_stack.pop();
    if (block_stack.size() == 0) {
        top = NULL;
    } else {
        top = block_stack.top();
    }
}

void ptx_parser_function::block_open() {
    ptx_parser_block * child_block = new ptx_parser_block();
    if (root) {
        assert(block_stack.size() > 0);
        top->statements.push_back(child_block);
        top = child_block;
        block_stack.push(child_block);
    } else {
        top = root = child_block;
        block_stack.push(child_block);
    }
}

void ptx_parser_function::finish_param() {
    params.push_back(param);
    param = param_t();
}

void ptx_parser_function::to_ir(function_t * f) const {
    f->linkage    = linkage;
    f->entry      = entry;
    f->entry_name = name;

    f->has_return_value = has_return_value;
    f->return_value     = return_value;

    f->params = params;
    if (no_body) {
        f->no_body = true;
    } else {
        f->no_body = false;
        root->to_ir(&f->scope);
        f->scope.fparent = f;
    }
}

ptx_parser_state::ptx_parser_state() : map_f64_to_f32(false),
        address_size(CHAR_BIT * sizeof(void *)),
        linkage(linkage_default) {
    function = new ptx_parser_function();
}

linkage_t ptx_parser_state::get_linkage() {
    linkage_t ret = linkage;
    linkage = linkage_default;
    return ret;
}

void ptx_parser_state::set_linkage(yytokentype token) {
    switch (token) {
        case TOKEN_EXTERN:
            linkage = linkage_extern;
            return;
        case TOKEN_VISIBLE:
            linkage = linkage_visible;
            return;
        default:
            assert(0 && "Unknown linkage token.");
            return;
    }
}

ptx_parser_state::~ptx_parser_state() {
    delete function;
    for (function_vector_t::iterator it = functions.begin();
            it != functions.end(); ++it) {
        delete *it;
    }
}

void ptx_parser_state::add_file(int file, const std::string & path) {
    files.insert(file_map_t::value_type(file, path));
}

void ptx_parser_state::set_target(yytokentype token) {
    switch (token) {
        case TOKEN_SM10:  sm = SM10; break;
        case TOKEN_SM11:  sm = SM11; break;
        case TOKEN_SM12:  sm = SM12; break;
        case TOKEN_SM13:  sm = SM13; break;
        case TOKEN_SM20:  sm = SM20; break;
        case TOKEN_SM21:  sm = SM21; break;
        case TOKEN_DEBUG: sm = DEBUG; break;
        case TOKEN_MAP_F64_TO_F32:
            map_f64_to_f32 = true;
            break;
        default:
            assert(0 && "Unknown SM token.");
            break;
    }
}

void ptx_parser_state::declare_function(const std::string & name) {
    function->name = name;

    functions.push_back(function);
    function = new ptx_parser_function();
}

void ptx_parser_state::declare_variable() {
    variables.push_back(variable);
    variable = variable_t();
}

void ptx_parser_state::declare_texture() {
    textures.push_back(texture);
    texture = texture_t();
}

void ptx_parser_state::to_ir(ptx_t * program) const {
    assert(program);

    /* Copy version, sm, and map flags. */
    program->version_major  = version_major;
    program->version_minor  = version_minor;
    program->sm             = sm;
    program->map_f64_to_f32 = map_f64_to_f32;
    program->address_size   = address_size;
    program->textures       = textures;
    program->variables      = variables;

    for (function_vector_t::const_iterator it = functions.begin();
            it != functions.end(); ++it) {
        function_t * f = new function_t();
        (*it)->to_ir(f);
        f->parent = program;
        program->entries.push_back(f);
    }
}

void ptx_parser_state::set_type(int64_t token) {
    switch (token) {
        case TOKEN_U64:
            type  = u64_type;
            return;
        case TOKEN_U32:
            type  = u32_type;
            return;
        case TOKEN_U16:
            type  = u16_type;
            return;
        case TOKEN_U8:
            type  = u8_type;
            return;
        case TOKEN_S64:
            type  = s64_type;
            return;
        case TOKEN_S32:
            type  = s32_type;
            return;
        case TOKEN_S16:
            type  = s16_type;
            return;
        case TOKEN_S8:
            type  = s8_type;
            return;
        case TOKEN_B64:
            type  = b64_type;
            return;
        case TOKEN_B32:
            type  = b32_type;
            return;
        case TOKEN_B16:
            type  = b16_type;
            return;
        case TOKEN_B8:
            type  = b8_type;
            return;
        case TOKEN_F64:
            type  = f64_type;
            return;
        case TOKEN_F32:
            type  = f32_type;
            return;
        case TOKEN_F16:
            type  = f16_type;
            return;
        case TOKEN_PRED:
            type  = pred_type;
            return;
        case TOKEN_TEXREF:
            type  = texref_type;
            return;
        default:
            assert(0 && "Unknown type token.");
            break;
    }
}

type_t ptx_parser_state::get_type() {
    type_t ret = type;
    type = invalid_type;
    return ret;
}
