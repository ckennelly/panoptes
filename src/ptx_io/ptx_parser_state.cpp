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
#include <cstring>
#include <iomanip>
#include <ptx_io/ptx_grammar.h>
#include <ptx_io/ptx_parser_state.h>
#include <sstream>
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
        error_encountered(false), linkage(linkage_default) {
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
    files.insert(ptx_t::file_map_t::value_type(file, path));
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
    program->files          = files;

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

ptx_token::ptx_token() { }

ptx_token::ptx_token(yytokentype t, const YYSTYPE *v) : token_(t) {
    memcpy(&value_, v, sizeof(value_));
}

yytokentype ptx_token::token() const {
    return token_;
}

const YYSTYPE & ptx_token::value() const {
    return value_;
}

ptx_token::~ptx_token() { }

namespace {

std::string integer_to_hex(uint64_t u, int width) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(width) << std::hex << u;

    std::string tmp;
    ss >> tmp;
    return tmp;
}

}

std::ostream & operator<<(std::ostream & o, const ptx_token & token) {
    switch (token.token()) {
        case OPCODE_ABS:
            return o << "abs";
        case OPCODE_ADD:
            return o << "add";
        case OPCODE_ADDC:
            return o << "addc";
        case OPCODE_AND:
            return o << "and";
        case OPCODE_ATOM:
            return o << "atom";
        case OPCODE_BAR:
            return o << "bar";
        case OPCODE_BFE:
            return o << "bfe";
        case OPCODE_BFI:
            return o << "bfi";
        case OPCODE_BFIND:
            return o << "bfind";
        case OPCODE_BRA:
            return o << "bra";
        case OPCODE_BREV:
            return o << "brev";
        case OPCODE_BRKPT:
            return o << "brkpt";
        case OPCODE_CALL:
            return o << "call";
        case OPCODE_CLZ:
            return o << "clz";
        case OPCODE_CNOT:
            return o << "cnot";
        case OPCODE_COPYSIGN:
            return o << "copysign";
        case OPCODE_COS:
            return o << "cos";
        case OPCODE_CVT:
            return o << "cvt";
        case OPCODE_CVTA:
            return o << "cvta";
        case OPCODE_DIV:
            return o << "div";
        case OPCODE_EX2:
            return o << "ex2";
        case OPCODE_EXIT:
            return o << "exit";
        case OPCODE_FMA:
            return o << "fma";
        case OPCODE_ISSPACEP:
            return o << "isspacep";
        case OPCODE_LD:
            return o << "ld";
        case OPCODE_LDU:
            return o << "ldu";
        case OPCODE_LG2:
            return o << "lg2";
        case OPCODE_MAD24:
            return o << "mad24";
        case OPCODE_MAD:
            return o << "mad";
        case OPCODE_MADC:
            return o << "madc";
        case OPCODE_MAX:
            return o << "max";
        case OPCODE_MEMBAR:
            return o << "membar";
        case OPCODE_MIN:
            return o << "min";
        case OPCODE_MOV:
            return o << "mov";
        case OPCODE_MUL24:
            return o << "mul24";
        case OPCODE_MUL:
            return o << "mul";
        case OPCODE_NEG:
            return o << "neg";
        case OPCODE_NOT:
            return o << "not";
        case OPCODE_OR:
            return o << "or";
        case OPCODE_PMEVENT:
            return o << "pmevent";
        case OPCODE_POPC:
            return o << "popc";
        case OPCODE_PREFETCH:
            return o << "prefetch";
        case OPCODE_PREFETCHU:
            return o << "prefetchu";
        case OPCODE_PRMT:
            return o << "prmt";
        case OPCODE_RCP:
            return o << "rcp";
        case OPCODE_RED:
            return o << "red";
        case OPCODE_REM:
            return o << "rem";
        case OPCODE_RET:
            return o << "ret";
        case OPCODE_RSQRT:
            return o << "rsqrt";
        case OPCODE_SAD:
            return o << "sad";
        case OPCODE_SELP:
            return o << "selp";
        case OPCODE_SET:
            return o << "set";
        case OPCODE_SETP:
            return o << "setp";
        case OPCODE_SHL:
            return o << "shl";
        case OPCODE_SHR:
            return o << "shr";
        case OPCODE_SIN:
            return o << "sin";
        case OPCODE_SLCT:
            return o << "slct";
        case OPCODE_SQRT:
            return o << "sqrt";
        case OPCODE_ST:
            return o << "st";
        case OPCODE_SUB:
            return o << "sub";
        case OPCODE_SUBC:
            return o << "subc";
        case OPCODE_SULD:
            return o << "suld";
        case OPCODE_SUQ:
            return o << "suq";
        case OPCODE_SURED:
            return o << "sured";
        case OPCODE_SUST:
            return o << "sust";
        case OPCODE_TESTP:
            return o << "testp";
        case OPCODE_TEX:
            return o << "tex";
        case OPCODE_TLD4:
            return o << "tld4";
        case OPCODE_TRAP:
            return o << "trap";
        case OPCODE_TXQ:
            return o << "txq";
        case OPCODE_VABSDIFF:
            return o << "vabsdiff";
        case OPCODE_VADD:
            return o << "vadd";
        case OPCODE_VMAD:
            return o << "vmad";
        case OPCODE_VMAX:
            return o << "vmax";
        case OPCODE_VMIN:
            return o << "vmin";
        case OPCODE_VOTE:
            return o << "vote";
        case OPCODE_VSET:
            return o << "vset";
        case OPCODE_VSHL:
            return o << "vshl";
        case OPCODE_VSHR:
            return o << "vshr";
        case OPCODE_VSUB:
            return o << "vsub";
        case OPCODE_XOR:
            return o << "xor";
        case TOKEN_1D:
            return o << ".1d";
        case TOKEN_2D:
            return o << ".2d";
        case TOKEN_3D:
            return o << ".3d";
        case TOKEN_A1D:
            return o << ".a1d";
        case TOKEN_A2D:
            return o << ".a2d";
        case TOKEN_ACUBE:
            return o << ".acube";
        case TOKEN_ADD:
            return o << ".add";
        case TOKEN_ADDRESS_SIZE:
            return o << ".address_size";
        case TOKEN_ADDRMODE0:
            return o << ".addr_mode_0";
        case TOKEN_ADDRMODE1:
            return o << ".addr_mode_1";
        case TOKEN_ADDRMODE2:
            return o << ".addr_mode_2";
        case TOKEN_ALIGN:
            return o << ".align";
        case TOKEN_ALL:
            return o << ".all";
        case TOKEN_AND:
            return o << ".and";
        case TOKEN_ANY:
            return o << ".any";
        case TOKEN_APPROX:
            return o << ".approx";
        case TOKEN_ARRIVE:
            return o << ".arrive";
        case TOKEN_B16:
            return o << ".b16";
        case TOKEN_B32:
            return o << ".b32";
        case TOKEN_B4E:
            return o << ".b4e";
        case TOKEN_B64:
            return o << ".b64";
        case TOKEN_B8:
            return o << ".b8";
        case TOKEN_BALLOT:
            return o << ".ballot";
        case TOKEN_BYTE:
            return o << ".byte";
        case TOKEN_CA:
            return o << ".ca";
        case TOKEN_CARRY:
            return o << ".carry";
        case TOKEN_CAS:
            return o << ".cas";
        case TOKEN_CDATATYPE:
            return o << ".channel_data_type";
        case TOKEN_CG:
            return o << ".cg";
        case TOKEN_COLON:
            return o << "':'";
        case TOKEN_COMMA:
            return o << "','";
        case TOKEN_CONST:
            return o << ".const";
        case TOKEN_CONSTANT_DECIMAL:
            return o << "constant (" << token.value().vsigned << ")";
        case TOKEN_CONSTANT_DOUBLE:
            return o << "double (0d" <<
                integer_to_hex(token.value().vsigned, 16) << ")";
        case TOKEN_CONSTANT_FLOAT:
            return o << "float (0f" <<
                integer_to_hex(token.value().vsigned, 8) << ")";
        case TOKEN_CORDER:
            return o << ".channel_order";
        case TOKEN_CS:
            return o << ".cs";
        case TOKEN_CUBE:
            return o << ".cube";
        case TOKEN_CV:
            return o << ".cv";
        case TOKEN_DEBUG:
            return o << ".debug";
        case TOKEN_DEC:
            return o << ".dec";
        case TOKEN_DEPTH:
            return o << ".depth";
        case TOKEN_ECL:
            return o << ".ecl";
        case TOKEN_ECR:
            return o << ".ecr";
        case TOKEN_ENTRY:
            return o << ".entry";
        case TOKEN_EQ:
            return o << ".eq";
        case TOKEN_EQU:
            return o << ".equ";
        case TOKEN_EQUAL:
            return o << "'='";
        case TOKEN_EXCH:
            return o << ".exch";
        case TOKEN_EXTERN:
            return o << ".extern";
        case TOKEN_F16:
            return o << ".f16";
        case TOKEN_F32:
            return o << ".f32";
        case TOKEN_F4E:
            return o << ".f4e";
        case TOKEN_F64:
            return o << ".f64";
        case TOKEN_FILE:
            return o << ".file";
        case TOKEN_FILTERMODE:
            return o << ".filter_mode";
        case TOKEN_FINITE:
            return o << ".finite";
        case TOKEN_FTZ:
            return o << ".ftz";
        case TOKEN_FULL:
            return o << ".full";
        case TOKEN_FUNCTION:
            return o << ".func";
        case TOKEN_FUNNORM:
            return o << ".force_unnormalized_coords";
        case TOKEN_GE:
            return o << ".ge";
        case TOKEN_GENERIC:
            return o << ".generic";
        case TOKEN_GEU:
            return o << ".geu";
        case TOKEN_GLOBAL:
            return o << ".global";
        case TOKEN_GT:
            return o << ".gt";
        case TOKEN_GTU:
            return o << ".gtu";
        case TOKEN_HEIGHT:
            return o << ".height";
        case TOKEN_HI:
            return o << ".hi";
        case TOKEN_HS:
            return o << ".hs";
        case TOKEN_IDENTIFIER:
            return o << "identifier (" << token.value().text << ")";
        case TOKEN_INC:
            return o << ".inc";
        case TOKEN_INDEPENDENT:
            return o << "texmode_independent";
        case TOKEN_INFINITE:
            return o << ".infinite";
        case TOKEN_L1:
            return o << ".L1";
        case TOKEN_L2:
            return o << ".L2";
        case TOKEN_LANGLE:
            return o << "'<'";
        case TOKEN_LBRACE:
            return o << "'{'";
        case TOKEN_LBRACKET:
            return o << "'['";
        case TOKEN_LE:
            return o << ".le";
        case TOKEN_LEU:
            return o << ".leu";
        case TOKEN_LO:
            return o << ".lo";
        case TOKEN_LOC:
            return o << ".loc";
        case TOKEN_LOCAL:
            return o << ".local";
        case TOKEN_LPAREN:
            return o << "'('";
        case TOKEN_LS:
            return o << ".ls";
        case TOKEN_LT:
            return o << ".lt";
        case TOKEN_LTU:
            return o << ".ltu";
        case TOKEN_LU:
            return o << ".lu";
        case TOKEN_MAP_F64_TO_F32:
            return o << "map_f64_to_f32";
        case TOKEN_MASK:
            return o << ".mask";
        case TOKEN_MAX:
            return o << ".max";
        case TOKEN_MCTA:
            return o << ".cta";
        case TOKEN_MGL:
            return o << ".gl";
        case TOKEN_MIN:
            return o << ".min";
        case TOKEN_MINUS:
            return o << "'-'";
        case TOKEN_MSYS:
            return o << ".sys";
        case TOKEN_NAN:
            return o << ".nan";
        case TOKEN_NE:
            return o << ".ne";
        case TOKEN_NEG_PREDICATE:
            return o << "predicate (@!" << token.value().text << ")";
        case TOKEN_NEU:
            return o << ".neu";
        case TOKEN_NORMAL:
            return o << ".normal";
        case TOKEN_NORMCOORD:
            return o << ".normalized_coords";
        case TOKEN_NOT:
            return o << "'!'";
        case TOKEN_NOTANUMBER:
            return o << ".notanumber";
        case TOKEN_NUM:
            return o << ".num";
        case TOKEN_NUMBER:
            return o << ".number";
        case TOKEN_OR:
            return o << ".or";
        case TOKEN_PARAM:
            return o << ".param";
        case TOKEN_PERIOD:
            return o << "'.'";
        case TOKEN_PIPE:
            return o << "'|'";
        case TOKEN_PLUS:
            return o << "'+'";
        case TOKEN_POPC:
            return o << ".popc";
        case TOKEN_PRED:
            return o << ".pred";
        case TOKEN_PREDICATE:
            return o << "predicate (@" << token.value().text << ")";
        case TOKEN_RANGLE:
            return o << "'>'";
        case TOKEN_RBRACE:
            return o << "'}'";
        case TOKEN_RBRACKET:
            return o << "']'";
        case TOKEN_RC16:
            return o << ".rc16";
        case TOKEN_RC8:
            return o << ".rc8";
        case TOKEN_RED:
            return o << ".red";
        case TOKEN_REG:
            return o << ".red";
        case TOKEN_RM:
            return o << ".rm";
        case TOKEN_RMI:
            return o << ".rmi";
        case TOKEN_RN:
            return o << ".rn";
        case TOKEN_RNI:
            return o << ".rni";
        case TOKEN_RP:
            return o << ".rp";
        case TOKEN_RPAREN:
            return o << "')'";
        case TOKEN_RPI:
            return o << ".rpi";
        case TOKEN_RZ:
            return o << ".rz";
        case TOKEN_RZI:
            return o << ".rzi";
        case TOKEN_S16:
            return o << ".s16";
        case TOKEN_S32:
            return o << ".s32";
        case TOKEN_S64:
            return o << ".s64";
        case TOKEN_S8:
            return o << ".s8";
        case TOKEN_SAT:
            return o << ".sat";
        case TOKEN_SECTION:
            return o << ".section";
        case TOKEN_SEMICOLON:
            return o << "';'";
        case TOKEN_SHARED:
            return o << ".shared";
        case TOKEN_SHIFTAMT:
            return o << ".shiftamt";
        case TOKEN_SM10:
            return o << "sm_10";
        case TOKEN_SM11:
            return o << "sm_11";
        case TOKEN_SM12:
            return o << "sm_12";
        case TOKEN_SM13:
            return o << "sm_13";
        case TOKEN_SM20:
            return o << "sm_20";
        case TOKEN_SM21:
            return o << "sm_21";
        case TOKEN_STRING:
            return o << "string ('" << token.value().text << "')";
        case TOKEN_SUBNORMAL:
            return o << ".subnormal";
        case TOKEN_SYNC:
            return o << ".sync";
        case TOKEN_TARGET:
            return o << ".target";
        case TOKEN_TEX:
            return o << ".tex";
        case TOKEN_TEXREF:
            return o << ".texref";
        case TOKEN_TO:
            return o << ".to";
        case TOKEN_U16:
            return o << ".u16";
        case TOKEN_U32:
            return o << ".u32";
        case TOKEN_U64:
            return o << ".u64";
        case TOKEN_U8:
            return o << ".u8";
        case TOKEN_UNDERSCORE:
            return o << "'_'";
        case TOKEN_UNI:
            return o << ".uni";
        case TOKEN_UNIFIED:
            return o << "texmode_unified";
        case TOKEN_V2:
            return o << ".v2";
        case TOKEN_V4:
            return o << ".v4";
        case TOKEN_VERSION:
            return o << ".version";
        case TOKEN_VISIBLE:
            return o << ".visible";
        case TOKEN_VOLATILE:
            return o << ".volatile";
        case TOKEN_W:
            return o << ".w";
        case TOKEN_WB:
            return o << ".wb";
        case TOKEN_WIDE:
            return o << ".wide";
        case TOKEN_WIDTH:
            return o << ".width";
        case TOKEN_WT:
            return o << ".wt";
        case TOKEN_X:
            return o << ".x";
        case TOKEN_XOR:
            return o << ".xor";
        case TOKEN_Y:
            return o << ".y";
        case TOKEN_Z:
            return o << ".z";
    }

    return o << token.token();
}
