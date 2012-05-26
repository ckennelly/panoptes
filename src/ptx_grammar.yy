%{

#undef yyFlexLexer
#define yyFlexLexer ptxFlexLexer
#include <FlexLexer.h>
#include "ptx_lexer.h"
#include "ptx_parser.h"
#include "ptx_parser_state.h"
#include "ptx_grammar.tab.hh"
#include <stdint.h>
#include <stdio.h>

namespace panoptes {

int yylex(YYSTYPE * token, YYLTYPE * location, panoptes::ptx_lexer * lexer, panoptes::ptx_parser_state * parser);
void yyerror(YYLTYPE * location, panoptes::ptx_lexer * lexer, panoptes::ptx_parser_state * parser,
    const char * message);

%}

%locations

%union {
    char text[1024];
    int64_t vsigned;
    uint64_t vunsigned;
    float vsingle;
    double vdouble;
}

%parse-param {panoptes::ptx_lexer * lexer}
%parse-param {panoptes::ptx_parser_state * parser}
%lex-param {panoptes::ptx_lexer * lexer}
%lex-param {panoptes::ptx_parser_state * parser}

%token<text> OPCODE_ADD OPCODE_SUB OPCODE_MUL OPCODE_MAD OPCODE_MUL24
%token<text> OPCODE_MAD24 OPCODE_SAD OPCODE_DIV OPCODE_REM OPCODE_ABS
%token<text> OPCODE_NEG OPCODE_MIN OPCODE_MAX OPCODE_SET OPCODE_SETP
%token<text> OPCODE_SLCT OPCODE_AND OPCODE_OR OPCODE_XOR OPCODE_NOT
%token<text> OPCODE_CNOT OPCODE_MOV OPCODE_LD OPCODE_ST OPCODE_CVT OPCODE_TEX
%token<text> OPCODE_CALL OPCODE_RET OPCODE_EXIT OPCODE_BAR OPCODE_ATOM
%token<text> OPCODE_SIN OPCODE_COS OPCODE_LG2 OPCODE_EX2 OPCODE_RCP
%token<text> OPCODE_SQRT OPCODE_RSQRT OPCODE_TRAP OPCODE_BRKPT TOKEN_STRING
%token<text> OPCODE_BRA OPCODE_ADDC OPCODE_BFE OPCODE_BFI OPCODE_BFIND
%token<text> OPCODE_CLZ OPCODE_COPYSIGN OPCODE_CVTA OPCODE_LDU OPCODE_SHL
%token<text> OPCODE_MADC OPCODE_MEMBAR OPCODE_PMEVENT OPCODE_POPC OPCODE_SHR
%token<text> OPCODE_PREFETCH OPCODE_PREFETCHU OPCODE_PRMT OPCODE_RED
%token<text> OPCODE_SELP OPCODE_SUBC OPCODE_SULD OPCODE_SUQ OPCODE_SURED
%token<text> OPCODE_SUST OPCODE_TESTP OPCODE_TLD4 OPCODE_TXQ OPCODE_VABSDIFF
%token<text> OPCODE_VADD OPCODE_VMAD OPCODE_VMAX OPCODE_VMIN OPCODE_VOTE
%token<text> OPCODE_VSET OPCODE_VSHL OPCODE_VSHR OPCODE_VSUB OPCODE_BREV
%token<text> OPCODE_FMA OPCODE_ISSPACEP

/* The precise type of these tokens is unimportant. */
%token<vsigned> TOKEN_ALIGN TOKEN_CONST TOKEN_ENTRY TOKEN_EXTERN TOKEN_FILE
%token<vsigned> TOKEN_FUNCTION TOKEN_GLOBAL TOKEN_LOCAL TOKEN_LOC TOKEN_PARAM
%token<vsigned> TOKEN_REG TOKEN_SECTION TOKEN_SHARED TOKEN_TARGET TOKEN_VERSION
%token<vsigned> TOKEN_VISIBLE TOKEN_SM10 TOKEN_SM11 TOKEN_SM12 TOKEN_SM13
%token<vsigned> TOKEN_SM20 TOKEN_SM21 TOKEN_MAP_F64_TO_F32 TOKEN_U32 TOKEN_S32
%token<vsigned> TOKEN_S8 TOKEN_S16 TOKEN_S64 TOKEN_U8 TOKEN_U16 TOKEN_U64
%token<vsigned> TOKEN_B8 TOKEN_B16 TOKEN_B32 TOKEN_B64 TOKEN_F16 TOKEN_F64
%token<vsigned> TOKEN_F32 TOKEN_PRED TOKEN_EQ TOKEN_NE TOKEN_LT TOKEN_LE
%token<vsigned> TOKEN_GT TOKEN_GE TOKEN_LS TOKEN_HS TOKEN_EQU TOKEN_NEU
%token<vsigned> TOKEN_LTU TOKEN_LEU TOKEN_GTU TOKEN_GEU TOKEN_NUM TOKEN_NAN
%token<vsigned> TOKEN_AND TOKEN_OR TOKEN_XOR TOKEN_HI TOKEN_LO TOKEN_RN
%token<vsigned> TOKEN_RM TOKEN_RZ TOKEN_RP TOKEN_RNI TOKEN_RMI TOKEN_RZI
%token<vsigned> TOKEN_RPI TOKEN_SAT TOKEN_UNI TOKEN_BYTE TOKEN_WIDE TOKEN_V2
%token<vsigned> TOKEN_V4 TOKEN_X TOKEN_Y TOKEN_Z TOKEN_W TOKEN_MIN TOKEN_MAX
%token<vsigned> TOKEN_DEC TOKEN_INC TOKEN_ADD TOKEN_CAS TOKEN_EXCH TOKEN_1D
%token<vsigned> TOKEN_2D TOKEN_3D TOKEN_IDENTIFIER TOKEN_PREDICATE
%token<vsigned> TOKEN_NEG_PREDICATE TOKEN_COMMA TOKEN_PERIOD TOKEN_SEMICOLON
%token<vsigned> TOKEN_LBRACE TOKEN_RBRACE TOKEN_LBRACKET TOKEN_RBRACKET
%token<vsigned> TOKEN_LPAREN TOKEN_RPAREN TOKEN_LANGLE TOKEN_RANGLE TOKEN_PLUS
%token<vsigned> TOKEN_MINUS TOKEN_NOT TOKEN_EQUAL TOKEN_UNDERSCORE TOKEN_DEBUG
%token<vsigned> TOKEN_UNIFIED TOKEN_INDEPENDENT TOKEN_TEX TOKEN_COLON
%token<vsigned> TOKEN_CA TOKEN_CG TOKEN_CS TOKEN_LU TOKEN_CV TOKEN_WB TOKEN_WT
%token<vsigned> TOKEN_PIPE TOKEN_FTZ TOKEN_SYNC TOKEN_ARRIVE TOKEN_RED
%token<vsigned> TOKEN_VOLATILE TOKEN_ADDRESS_SIZE TOKEN_TO TOKEN_GENERIC
%token<vsigned> TOKEN_APPROX TOKEN_FULL TOKEN_MCTA TOKEN_MGL TOKEN_MSYS
%token<vsigned> TOKEN_MASK TOKEN_ALL TOKEN_ANY TOKEN_BALLOT TOKEN_POPC
%token<vsigned> TOKEN_A1D TOKEN_A2D TOKEN_CUBE TOKEN_ACUBE TOKEN_TEXREF
%token<vsigned> TOKEN_CARRY TOKEN_SHIFTAMT TOKEN_FINITE TOKEN_INFINITE
%token<vsigned> TOKEN_NUMBER TOKEN_NOTANUMBER TOKEN_NORMAL TOKEN_SUBNORMAL
%token<vsigned> TOKEN_F4E TOKEN_B4E TOKEN_RC8 TOKEN_ECL TOKEN_ECR TOKEN_RC16

/* Important pairings. */
%token<vsigned> TOKEN_CONSTANT_DECIMAL
%token<vsingle> TOKEN_CONSTANT_FLOAT
%token<vdouble> TOKEN_CONSTANT_DOUBLE

/* Reentrancy is nice to have */
%define api.pure

%start statements

%% /* The grammar follows. */

statements : directiveStatement | statements directiveStatement;

label : TOKEN_IDENTIFIER TOKEN_COLON {
    parser->function->top->label->name = $<text>1;
    parser->function->top->finish_label();
};

/* Per PTX ISA 3.0 pg 25 */
directiveStatement : addressSize | /* branchTargets | callPrototype | */
    entry | file | linkable | loc | /* maxnctapersm |
    maxnreg | maxntid | minnctapersm | pragma | reqntid |
    section | */ target | texDeclaration | version ;

addressSize : TOKEN_ADDRESS_SIZE TOKEN_CONSTANT_DECIMAL {
    parser->address_size = $<vsigned>2;
};

branchTargets : ;

callPrototype : ;

rootBlock : TOKEN_SEMICOLON {
    parser->function->no_body = true;
}

rootBlock : basicBlock {
    parser->function->no_body = false;
};

linkable : linkingDirective func ;
linkable : linkingDirective globalVariableDeclaration ;

returnValue : TOKEN_LPAREN entryArgument TOKEN_RPAREN {
    parser->function->has_return_value = true;
    parser->function->return_value = parser->function->params.back();
    parser->function->params.pop_back();
};

optionalReturnValue : /* */ | returnValue ;

func : TOKEN_FUNCTION optionalReturnValue TOKEN_IDENTIFIER entryArguments
        rootBlock {
    parser->function->linkage = parser->get_linkage();
    parser->function->entry = false;
    parser->declare_function($<text>3);
};

entry : TOKEN_ENTRY TOKEN_IDENTIFIER entryArguments rootBlock {
    parser->function->entry = true;
    parser->declare_function($<text>2);
};

entryArguments : /* */;
entryArguments : TOKEN_LPAREN TOKEN_RPAREN ;
entryArguments : TOKEN_LPAREN entryArgumentList TOKEN_RPAREN ;

paramAlignment : TOKEN_ALIGN TOKEN_CONSTANT_DECIMAL {
    parser->function->param.has_align = true;
    parser->function->param.alignment = $<vsigned>2;
};

optionalParamAlignment : /* */ | paramAlignment ;

/**
 * Parameters cannot be passed as multidimensional arrays.  The PTX ISA 3.0
 * document does not discuss their existence.  Attempting to compile a
 * visually valid multidimensional array as a kernel parameter causes ptxas
 * version 4.1 to emit an error.
 */
paramArrayDimension : TOKEN_LBRACKET TOKEN_CONSTANT_DECIMAL TOKEN_RBRACKET {
    assert(parser->function->param.array_dimensions < 3u);
    parser->function->param.array_size[
        parser->function->param.array_dimensions] = $<vsigned>2;
    parser->function->param.array_dimensions++;
};

optionalParamArray : /* */ | paramArrayDimension ;

argumentVariableProperties : optionalParamAlignment addressableVariableType
        {
    parser->set_type($<vsigned>2);
};

entryArgument : TOKEN_PARAM argumentVariableProperties TOKEN_IDENTIFIER
        optionalParamArray {
    parser->function->param.space = param_space;

    parser->function->param.type  = parser->get_type();

    parser->function->param.name = $<text>3;
    parser->function->finish_param();
};

entryArgumentList : entryArgument ;
entryArgumentList : entryArgumentList TOKEN_COMMA entryArgument;

dataTypeToken : TOKEN_U64 | TOKEN_U32 | TOKEN_U16 | TOKEN_U8 |
        TOKEN_S64 | TOKEN_S32 | TOKEN_S16 | TOKEN_S8 | TOKEN_B64 | TOKEN_B32 |
        TOKEN_B16 | TOKEN_B8 | TOKEN_F64 | TOKEN_F32 | TOKEN_F16 | TOKEN_PRED ;
dataType : dataTypeToken {
    parser->set_type($<vsigned>1);
};

declarationSpace : TOKEN_REG | TOKEN_PARAM | TOKEN_LOCAL | TOKEN_SHARED ;
declarationSuffix : TOKEN_LANGLE TOKEN_CONSTANT_DECIMAL TOKEN_RANGLE {
    parser->function->top->variable.suffix = $<vsigned>2;
};
declarationSuffix : /* */ ;

flexibleArray : TOKEN_LBRACKET TOKEN_RBRACKET {
    assert(parser->function->top->variable.array_dimensions < 3u);
    parser->function->top->variable.array_dimensions++;
    parser->function->top->variable.array_flexible = true;
};

arrayDimension : TOKEN_LBRACKET TOKEN_CONSTANT_DECIMAL TOKEN_RBRACKET {
    assert(parser->function->top->variable.array_dimensions < 3u);
    parser->function->top->variable.array_size[
        parser->function->top->variable.array_dimensions] = $<vsigned>2;
    parser->function->top->variable.array_dimensions++;
};

arrayLastDimension : arrayDimension | flexibleArray ;
arrayDiemnsionNT : arrayDimension ;
arrayDiemnsionNT : arrayDiemnsionNT arrayDimension ;
arrayDimensions : arrayLastDimension ;
arrayDimensions : arrayDiemnsionNT arrayLastDimension ;
array : arrayDimensions {
    parser->function->top->variable.is_array = true;
};
optionalArray: /* */ | array ;

alignment : TOKEN_ALIGN TOKEN_CONSTANT_DECIMAL {
    parser->function->top->variable.has_align = true;
    parser->function->top->variable.alignment = $<vsigned>2;
};

optionalAlignment : /* */ | alignment ;

declaration: declarationSpace optionalAlignment dataType TOKEN_IDENTIFIER declarationSuffix optionalArray TOKEN_SEMICOLON {
    parser->function->top->variable.set_token($<vsigned>1);

    parser->function->top->variable.type = parser->get_type();
    parser->function->top->variable.name = $<text>4;

    parser->function->top->finish_variable();
};

basicBlockStatement : declaration | instruction | label | loc | basicBlock;

basicBlockStatements : basicBlockStatement | basicBlockStatements basicBlockStatement;
zeroOrMoreBasicBlockStatements : /* */ | basicBlockStatements ;

blockOpen : TOKEN_LBRACE {
    parser->function->block_open();
};

blockClose : TOKEN_RBRACE {
    parser->function->block_close();
}

basicBlock : blockOpen zeroOrMoreBasicBlockStatements blockClose ;

file : TOKEN_FILE TOKEN_CONSTANT_DECIMAL TOKEN_STRING ;

loc : TOKEN_LOC TOKEN_CONSTANT_DECIMAL TOKEN_CONSTANT_DECIMAL TOKEN_CONSTANT_DECIMAL ;

maxnctapersm : ;

maxnreg : ;

maxntid : ;

minnctapersm : ;

pragma : ;

reqntid : ;

section : ;

targetSM : TOKEN_SM10 | TOKEN_SM11 | TOKEN_SM12 | TOKEN_SM13 | TOKEN_SM20 |
        TOKEN_SM21 | TOKEN_DEBUG;
targetTM : TOKEN_UNIFIED | TOKEN_INDEPENDENT ;
targetFlagStub : TOKEN_MAP_F64_TO_F32 ;
targetFlagStub : targetSM | targetTM ;
targetFlag : targetFlagStub {
    parser->set_target($<vsigned>1);
};

targetFlags : targetFlag ;
targetFlags : targetFlag TOKEN_COMMA targetFlags ;
target : TOKEN_TARGET targetFlags ;


texIdentifierList : TOKEN_IDENTIFIER {
    parser->texture.names.push_back($<text>1);
}
texIdentifierList : texIdentifierList TOKEN_COMMA TOKEN_IDENTIFIER {
    parser->texture.names.push_back($<text>3);
};
texDeclaration : TOKEN_TEX dataType texIdentifierList TOKEN_SEMICOLON {
    parser->texture.type = parser->get_type();
    parser->declare_texture();
};
texrefDeclaration : TOKEN_TEXREF TOKEN_IDENTIFIER TOKEN_SEMICOLON {
    parser->texture.names.push_back($<text>2);

    parser->set_type($<vsigned>1);
    parser->texture.type = parser->get_type();
    parser->declare_texture();
};

version : TOKEN_VERSION TOKEN_CONSTANT_DECIMAL TOKEN_PERIOD TOKEN_CONSTANT_DECIMAL {
    parser->version_major = $<vsigned>2;
    parser->version_minor = $<vsigned>4;
};

/* These are global in the sense of "declared outside of a function", they are
   not necessarily .global's. */
globalAddressSpace : TOKEN_CONST | TOKEN_GLOBAL ;

/* PTX ISA 3.0 pg. 190 */
linkingDirectiveToken : TOKEN_EXTERN | TOKEN_VISIBLE ;
linkingDirective : linkingDirectiveToken {
    parser->set_linkage($<vsigned>1);
}
linkingDirective : /* */ ;

/* PTX ISA 3.0 pg. 36 "Predicate variables may only be declared in the register
                       state space." */
addressableVariableTypeToken : TOKEN_U64 | TOKEN_U32 | TOKEN_U16 | TOKEN_U8 |
        TOKEN_S64 | TOKEN_S32 | TOKEN_S16 | TOKEN_S8 | TOKEN_B64 | TOKEN_B32 |
        TOKEN_B16 | TOKEN_B8 | TOKEN_F64 | TOKEN_F32 | TOKEN_F16 ;
addressableVariableType : addressableVariableTypeToken {
    parser->set_type($<vsigned>1);
};

globalAlign : TOKEN_ALIGN TOKEN_CONSTANT_DECIMAL {
    parser->variable.has_align = true;
    parser->variable.alignment = $<vsigned>2;
};

addressableVariableProperty : addressableVariableType | globalAlign ;
addressableVariableProperties : addressableVariableProperty |
    addressableVariableProperties addressableVariableProperty ;

globalFlexibleArray : TOKEN_LBRACKET TOKEN_RBRACKET {
    assert(parser->variable.array_dimensions < 3u);
    parser->variable.array_dimensions++;
    parser->variable.array_flexible = true;
};
globalArrayDimension : TOKEN_LBRACKET TOKEN_CONSTANT_DECIMAL TOKEN_RBRACKET {
    assert(parser->variable.array_dimensions < 3u);
    parser->variable.array_size[parser->variable.array_dimensions] =
        $<vsigned>2;
    parser->variable.array_dimensions++;
};

globalArrayLastDimension : globalArrayDimension | globalFlexibleArray ;
globalArrayDiemnsionNT : globalArrayDimension ;
globalArrayDiemnsionNT : globalArrayDiemnsionNT globalArrayDimension ;
globalArrayDimensions : globalArrayLastDimension ;
globalArrayDimensions : globalArrayDiemnsionNT globalArrayLastDimension ;
globalArray : globalArrayDimensions {
    parser->variable.is_array = true;
};
optionalGlobalArray: /* */ | globalArray ;

initializerValue : TOKEN_CONSTANT_DECIMAL {
    variant_t v;
    v.type = variant_integer;
    v.data.u = $<vsigned>1;
    parser->variable.initializer.push_back(v);
};
initializerValue : TOKEN_CONSTANT_DOUBLE {
    variant_t v;
    v.type = variant_double;
    v.data.d = $<vdouble>1;
    parser->variable.initializer.push_back(v);
};
initializerValue : TOKEN_CONSTANT_FLOAT {
    variant_t v;
    v.type = variant_single;
    v.data.d = $<vsingle>1;
    parser->variable.initializer.push_back(v);
};

initializerValues : initializerValue ;
initializerValues : initializerValues TOKEN_COMMA initializerValue ;
initializer : TOKEN_EQUAL TOKEN_LBRACE initializerValues TOKEN_RBRACE {
    parser->variable.has_initializer = true;
    parser->variable.initializer_vector = true;
};
initializer : TOKEN_EQUAL initializerValue {
    parser->variable.has_initializer = true;
    parser->variable.initializer_vector = false;
};

initializer : /* */ ;


/* Only .const and .global can be initialized. */
globalInitializableDeclaration : globalAddressSpace
        addressableVariableProperties TOKEN_IDENTIFIER optionalGlobalArray initializer TOKEN_SEMICOLON {
    parser->variable.linkage = parser->get_linkage();
    parser->variable.type = parser->get_type();
    parser->variable.set_token($<vsigned>1);
    parser->variable.name = $<text>3;
    parser->declare_variable();
};

globalInitializableDeclaration : TOKEN_GLOBAL texrefDeclaration ;

globalSharedDeclaration : TOKEN_SHARED
        addressableVariableProperties TOKEN_IDENTIFIER optionalGlobalArray TOKEN_SEMICOLON {
    parser->variable.linkage = parser->get_linkage();
    parser->variable.name = $<text>3;
    parser->variable.set_token($<vsigned>1);
    parser->variable.type = parser->get_type();
    parser->declare_variable();
}

globalVariableDeclaration : globalInitializableDeclaration | globalSharedDeclaration ;

predicate : TOKEN_PREDICATE {
    parser->function->top->instruction.has_predicate = true;
    parser->function->top->instruction.predicate = $<text>1;

};
predicate : TOKEN_NEG_PREDICATE {
    parser->function->top->instruction.is_negated = true;
    parser->function->top->instruction.has_predicate = true;
    parser->function->top->instruction.predicate = $<text>1;
};
optionalPredicate : /* */ | predicate ;

instruction : optionalPredicate instructionOp TOKEN_SEMICOLON {
    parser->function->top->declare_instruction();
};

instructionOp : abs | add | addc | and | atom | bar | bfe | bfi | bfind |
    bra | brev | brkpt | call | clz | cnot | copysign | cos | cvt | cvta |
    div | ex2 | exit | fma | isspacep | ld | ldu | lg2 | mad | mad24 | madc |
    max | membar | min | mov | mul | mul24 | neg | not | or | pmevent | popc |
    prefetch | prefetchu | prmt | rcp | red | rem | ret | rsqrt | sad | selp |
    set | setp | shl | shr | sin | slct | sqrt | st | sub | subc | suld |
    suq | sured | sust | testp | tex | tld4 | trap | txq | vabsdiff | vadd |
    vmad | vmax | vmin | vote | vset | vshl | vshr | vsub | xor {
}

abs : OPCODE_ABS optionalFTZ TOKEN_F32 identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->set_type(TOKEN_F32);
    parser->function->top->instruction.type = parser->get_type();

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();

};

abs : OPCODE_ABS TOKEN_F64 identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->set_type(TOKEN_F64);
    parser->function->top->instruction.type = parser->get_type();

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

abs : OPCODE_ABS integerDataType identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

saturating : TOKEN_SAT {
    parser->function->top->instruction.saturating = true;
};
optionalSaturating : /* */ | saturating ;

immediateFloat : TOKEN_CONSTANT_FLOAT {
    operand_t op;
    op.op_type = operand_float;
    op.fvalue = $<vsingle>1;
    parser->operands.push_back(op);
};

immediateFloat : TOKEN_CONSTANT_DOUBLE {
    operand_t op;
    op.op_type = operand_double;
    op.dvalue = $<vdouble>1;
    parser->operands.push_back(op);
};

immediateValue : immediateDecimal | immediateFloat ;
immedOrVarOperand : immediateValue | identifierOperand ;

optionalCarryOut : TOKEN_CARRY {
    parser->function->top->instruction.carry_out = true;
};
optionalCarryOut : ;

add : OPCODE_ADD optionalCarryOut optionalSaturating dataType immedOrVarOperand
        TOKEN_COMMA immedOrVarOperand TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

addc : OPCODE_ADDC optionalCarryOut dataType immedOrVarOperand TOKEN_COMMA
        immedOrVarOperand TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

and : OPCODE_AND dataType identifierOperand TOKEN_COMMA identifierOperand TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

atomicSpaceToken : TOKEN_SHARED | TOKEN_GLOBAL ;
atomicSpace : atomicSpaceToken {
    parser->function->top->instruction.set_token($<vsigned>1);
};
atomicSpace : /* */ {
    parser->function->top->instruction.space = generic_space;
}
atomicOpToken : TOKEN_AND | TOKEN_OR | TOKEN_XOR | TOKEN_CAS | TOKEN_EXCH |
    TOKEN_ADD | TOKEN_INC | TOKEN_DEC | TOKEN_MIN | TOKEN_MAX ;
atomicOp : atomicOpToken {
    parser->function->top->instruction.set_atomic_op($<vsigned>1);
};
optionalImmediateOrVariableOperand : /* */ | TOKEN_COMMA immedOrVarOperand ;
atom : OPCODE_ATOM atomicSpace atomicOp dataType identifierOperand TOKEN_COMMA
        stDestination TOKEN_COMMA immedOrVarOperand
        optionalImmediateOrVariableOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

decimalOrVarOperand : immediateDecimal | identifierOperand ;

bar : OPCODE_BAR TOKEN_ARRIVE {
    assert(0);
};

optionalDecOrVarOperand : /* */ | TOKEN_COMMA decimalOrVarOperand ;
bar : OPCODE_BAR TOKEN_SYNC decimalOrVarOperand optionalDecOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.set_token($<vsigned>2);
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

barRedOpToken : TOKEN_POPC | TOKEN_AND | TOKEN_OR ;
optionalIdentifier : /* */ | identifierOperand TOKEN_COMMA ;
bar : OPCODE_BAR TOKEN_RED barRedOpToken dataType identifierOperand
        TOKEN_COMMA immedOrVarOperand optionalIdentifier TOKEN_COMMA
        optionalNegatedOperand identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.set_token($<vsigned>2);
    parser->function->top->instruction.set_token($<vsigned>3);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

bfe : OPCODE_BFE dataType identifierOperand TOKEN_COMMA immedOrVarOperand
        TOKEN_COMMA immedOrVarOperand TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

bfi : OPCODE_BFI dataType identifierOperand TOKEN_COMMA immedOrVarOperand
        TOKEN_COMMA immedOrVarOperand TOKEN_COMMA immedOrVarOperand
        TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

shiftAmount : TOKEN_SHIFTAMT {
    parser->function->top->instruction.shiftamt = true;
};

optionalShiftAmount : /* */ | shiftAmount ;

bfind : OPCODE_BFIND optionalShiftAmount dataType identifierOperand TOKEN_COMMA
        immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

uni : TOKEN_UNI {
    parser->function->top->instruction.uniform = true;
};
optionalUni : /* */ | uni ;

bra : OPCODE_BRA optionalUni identifierOperand {
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();

    parser->function->top->instruction.set_token($<vsigned>1);
}

brev : OPCODE_BREV rawDataType identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

brkpt : OPCODE_BRKPT {
    parser->function->top->instruction.set_token($<vsigned>1);
};

returnParam : TOKEN_LPAREN identifierOperand TOKEN_RPAREN TOKEN_COMMA {
    parser->function->top->instruction.has_return_value = true;
};
optionalReturnParam : /* */ | returnParam ;
callParams: identifierOperand | callParams TOKEN_COMMA identifierOperand ;
callParamList: TOKEN_COMMA TOKEN_LPAREN callParams TOKEN_RPAREN ;
optionalCallParamList: /* */ | callParamList ;
/* TODO: This does not support indirect calls. */
call : OPCODE_CALL optionalUni optionalReturnParam identifierOperand
        optionalCallParamList {
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();

    parser->function->top->instruction.set_token($<vsigned>1);
};

rawDataTypeToken : TOKEN_B16 | TOKEN_B32 | TOKEN_B64 ;
rawDataType : rawDataTypeToken {
    parser->set_type($<vsigned>1);
};

clz : OPCODE_CLZ rawDataType identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

cnot : OPCODE_CNOT rawDataType identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

copysign : OPCODE_COPYSIGN floatingDataType identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

cos : OPCODE_COS optionalApprox optionalFTZ TOKEN_F32 identifierOperand
        TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.approximation = approximate;

    parser->set_type(TOKEN_F32);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

iRoundingToken : TOKEN_RNI | TOKEN_RZI | TOKEN_RMI | TOKEN_RPI ;
iRounding : iRoundingToken {
    parser->function->top->instruction.set_token($<vsigned>1);
};
fRoundingToken : TOKEN_RN  | TOKEN_RZ  | TOKEN_RM  | TOKEN_RP ;
fRounding: fRoundingToken {
    parser->function->top->instruction.set_token($<vsigned>1);
}
rounding : iRounding | fRounding ;
optionalRounding : /* */ | rounding ;
optionalFRounding : /* */ | fRounding ;

cvt : OPCODE_CVT optionalRounding optionalFTZ optionalSaturating dataType
        dataType identifierOperand TOKEN_COMMA singleOperand {
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();

    parser->function->top->instruction.set_token($<vsigned>1);
    parser->set_type($<vsigned>5);
    parser->function->top->instruction.type = parser->get_type();
    parser->set_type($<vsigned>6);
    parser->function->top->instruction.type2 = parser->get_type();
};

spaceTypeToken : TOKEN_GENERIC | TOKEN_LOCAL | TOKEN_SHARED | TOKEN_GLOBAL ;
spaceType : spaceTypeToken {
    parser->function->top->instruction.set_token($<vsigned>1);
};

integerWidthType : TOKEN_U32 | TOKEN_U64 ;

cvta : OPCODE_CVTA TOKEN_TO spaceType integerWidthType identifierOperand
        TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.is_to = true;
    parser->set_type($<vsigned>4);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

cvta : OPCODE_CVTA spaceType integerWidthType identifierOperand
        TOKEN_COMMA addressableOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->set_type($<vsigned>3);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

div : OPCODE_DIV integerDataType identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand {
    /* Integer division */
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

approximationToken : TOKEN_APPROX | TOKEN_FULL ;
approximation : approximationToken {
    parser->function->top->instruction.set_token($<vsigned>1);
}

div : OPCODE_DIV approximation optionalFTZ TOKEN_F32 identifierOperand
        TOKEN_COMMA identifierOperand TOKEN_COMMA identifierOperand {
    /* Approximate floating point division */
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->set_type(TOKEN_F32);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

floatingDataTypeToken : TOKEN_F32 | TOKEN_F64 ;
floatingDataType : floatingDataTypeToken {
    parser->set_type($<vsigned>1);
};

div : OPCODE_DIV fRounding optionalFTZ floatingDataType identifierOperand
        TOKEN_COMMA identifierOperand TOKEN_COMMA identifierOperand {
    /* IEEE754 floating point division */
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

ex2 : OPCODE_EX2 optionalApprox optionalFTZ floatingDataType
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.approximation = approximate;
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

exit : OPCODE_EXIT {
    parser->function->top->instruction.set_token($<vsigned>1);
};

fma : OPCODE_FMA fRounding optionalFTZ optionalSaturating TOKEN_F32
        identifierOperand TOKEN_COMMA identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->set_type(TOKEN_F32);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

fma : OPCODE_FMA fRounding TOKEN_F64 identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->set_type(TOKEN_F64);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

isspacep : OPCODE_ISSPACEP stateSpace identifierOperand TOKEN_COMMA
        immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.set_token($<vsigned>2);

    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

immediateDecimal: TOKEN_CONSTANT_DECIMAL {
    parser->operand.op_type = operand_constant;
    parser->operand.offset = $<vsigned>1;
    parser->operands.push_back(parser->operand);
    parser->operand.reset();
}

stateSpace          : TOKEN_CONST | TOKEN_GLOBAL | TOKEN_LOCAL | TOKEN_PARAM | TOKEN_SHARED ;
optionalSpace       : /* */
optionalSpace       : stateSpace {
    parser->function->top->instruction.set_token($<vsigned>1);
}

ldCacheOp           : TOKEN_CA | TOKEN_CG | TOKEN_CS | TOKEN_LU | TOKEN_CV ;
optionalLDCacheOp   : /* */ | ldCacheOp {
    parser->function->top->instruction.set_token($<vsigned>1);
}

vectorType          : TOKEN_V2 | TOKEN_V4 ;
optionalVectorType  : /* */ | vectorType {
    parser->function->top->instruction.set_token($<vsigned>1);
}

ldSource : TOKEN_LBRACKET addressableOperand TOKEN_RBRACKET ;
ldSource : TOKEN_LBRACKET immediateDecimal TOKEN_RBRACKET {
    parser->operands.push_back(parser->operand);
    parser->operand.reset();
}

volatileFlag : TOKEN_VOLATILE {
    parser->function->top->instruction.is_volatile = true;
};
optionalVolatile : /* */ | volatileFlag ;

ld : OPCODE_LD optionalVolatile optionalSpace optionalLDCacheOp
        optionalVectorType dataType lValue TOKEN_COMMA ldSource {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

ldu : OPCODE_LDU optionalSpace optionalVectorType dataType lValue TOKEN_COMMA
        ldSource {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

lg2 : OPCODE_LG2 optionalApprox optionalFTZ floatingDataType
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.approximation = approximate;
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

/* Integers. */
integerDataTypeToken : TOKEN_U16 | TOKEN_U32 | TOKEN_U64 | TOKEN_S16 |
    TOKEN_S32 | TOKEN_S64 ;
integerDataType : integerDataTypeToken {
    parser->set_type($<vsigned>1);
};

mad : OPCODE_MAD optionalWidth optionalCarryOut optionalSaturating
        integerDataType identifierOperand TOKEN_COMMA identifierOperand
        TOKEN_COMMA identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

/* Floating point. */
mad : OPCODE_MAD optionalFRounding optionalSaturating TOKEN_F32
        identifierOperand TOKEN_COMMA identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->set_type(TOKEN_F32);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

mad : OPCODE_MAD fRounding TOKEN_F64 identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->set_type(TOKEN_F64);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

mad24 : OPCODE_MAD24 optionalWidth optionalSaturating integerDataType
        identifierOperand TOKEN_COMMA identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

madc : OPCODE_MADC optionalWidth optionalCarryOut
        integerDataType identifierOperand TOKEN_COMMA immedOrVarOperand
        TOKEN_COMMA immedOrVarOperand TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

max : OPCODE_MAX optionalFTZ dataType identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

membarLevelToken : TOKEN_MCTA | TOKEN_MGL | TOKEN_MSYS ;
membar : OPCODE_MEMBAR membarLevelToken {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.set_token($<vsigned>2);
}

min : OPCODE_MIN optionalFTZ dataType identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

optionalPackedField : /* */ {
    parser->operand.field.push_back(field_none);
};

packedFieldToken : TOKEN_X | TOKEN_Y | TOKEN_Z | TOKEN_W ;
optionalPackedField : packedFieldToken {
    parser->operand.push_field($<vsigned>1);
};

operand : TOKEN_IDENTIFIER optionalPackedField {
    parser->operand.identifier.push_back($<text>1);
};

singleOperand : operand {
    parser->operand.op_type = operand_identifier;

    parser->operands.push_back(parser->operand);
    parser->operand.reset();
}

lValue : operand {
    parser->operand.op_type = operand_identifier;

    parser->operands.push_back(parser->operand);
    parser->operand.reset();
};

operandList : operand | operandList TOKEN_COMMA operand ;
lValue : TOKEN_LBRACE operandList TOKEN_RBRACE {
    parser->operand.op_type = operand_identifier;

    parser->operands.push_back(parser->operand);
    parser->operand.reset();
};

addressableOperandBase : TOKEN_IDENTIFIER {
    parser->operand.op_type = operand_addressable;
    parser->operand.identifier.push_back($<text>1);
};

addressableOperandBase : TOKEN_IDENTIFIER TOKEN_PLUS TOKEN_CONSTANT_DECIMAL {
    parser->operand.op_type = operand_addressable;
    parser->operand.identifier.push_back($<text>1);
    parser->operand.offset = $<vsigned>3;
};

addressableOperandBase : TOKEN_IDENTIFIER TOKEN_MINUS TOKEN_CONSTANT_DECIMAL {
    assert(0);
    parser->operand.op_type = operand_addressable;
    parser->operand.identifier.push_back($<text>1);
    parser->operand.offset = -1 * $<vsigned>3;
};

addressableOperand : addressableOperandBase {
    parser->operands.push_back(parser->operand);
    parser->operand.reset();
};

indexedOperand : TOKEN_IDENTIFIER TOKEN_LBRACKET TOKEN_CONSTANT_DECIMAL TOKEN_RBRACKET {
    assert(0);
    parser->operand.op_type = operand_indexed;
    parser->operand.identifier.push_back($<text>1);
    parser->operand.offset = $<vsigned>3;
};

movSource : lValue | immediateValue ;

movSource : addressableOperandBase | indexedOperand {
    assert(0);

    parser->operands.push_back(parser->operand);
    parser->operand.reset();
};

mov : OPCODE_MOV dataType lValue TOKEN_COMMA movSource {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

widthToken: TOKEN_HI | TOKEN_LO | TOKEN_WIDE ;
width : widthToken {
    parser->function->top->instruction.set_width($<vsigned>1);
};
optionalWidth : /* */ | width ;

mul : OPCODE_MUL optionalWidth dataType identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

mul : OPCODE_MUL optionalFRounding optionalFTZ optionalSaturating dataType
        identifierOperand TOKEN_COMMA identifierOperand TOKEN_COMMA
        immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

narrowHalfToken : TOKEN_HI | TOKEN_LO ;
narrowHalf : narrowHalfToken {
    parser->function->top->instruction.set_width($<vsigned>1);
};
optionalNarrowHalf : /* */ | narrowHalf ;

mul24 : OPCODE_MUL24 optionalNarrowHalf integerDataType identifierOperand
        TOKEN_COMMA identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

neg : OPCODE_NEG integerDataType identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

neg : OPCODE_NEG optionalFTZ floatingDataType identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

not : OPCODE_NOT dataType identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

or : OPCODE_OR dataType identifierOperand TOKEN_COMMA identifierOperand
        TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

mask : TOKEN_MASK {
    parser->function->top->instruction.mask = true;
};
optionalMask : /* */ | mask ;

pmevent : OPCODE_PMEVENT optionalMask immediateDecimal {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

popc : OPCODE_POPC dataType identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

prefetch : ;
prefetchu : ;

prmtModeToken: TOKEN_F4E | TOKEN_B4E | TOKEN_RC8 | TOKEN_ECL | TOKEN_ECR |
    TOKEN_RC16 ;
prmtMode : prmtModeToken {
    parser->function->top->instruction.set_token($<vsigned>1);
};
prmtMode : /* */ {
    parser->function->top->instruction.prmt_mode = prmt_default;
};

prmt : OPCODE_PRMT dataType prmtMode identifierOperand TOKEN_COMMA
        immedOrVarOperand TOKEN_COMMA immedOrVarOperand TOKEN_COMMA
        immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

rcp : OPCODE_RCP TOKEN_APPROX optionalFTZ floatingDataType identifierOperand
        TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.approximation = approximate;
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

rcp : OPCODE_RCP fRounding optionalFTZ floatingDataType identifierOperand
        TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

red : OPCODE_RED atomicSpace atomicOp dataType stDestination TOKEN_COMMA
        immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

rem : OPCODE_REM integerDataType identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

ret : OPCODE_RET optionalUni {
    parser->function->top->instruction.set_token($<vsigned>1);
};

rsqrt : OPCODE_RSQRT optionalApprox optionalFTZ floatingDataType
        identifierOperand TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.approximation = approximate;
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

sad : OPCODE_SAD integerDataType identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

selp : OPCODE_SELP dataType identifierOperand TOKEN_COMMA immedOrVarOperand
        TOKEN_COMMA immedOrVarOperand TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

set : OPCODE_SET cmpOp optionalBoolOp optionalFTZ dataType dataType
        identifierOperand TOKEN_COMMA identifierOperand TOKEN_COMMA
        immedOrVarOperand optionalCpred {
    parser->set_type($<vsigned>5);
    parser->function->top->instruction.type = parser->get_type();
    parser->set_type($<vsigned>6);
    parser->function->top->instruction.type2 = parser->get_type();

    parser->function->top->instruction.set_token($<vsigned>1);

    parser->function->top->instruction.has_ppredicate = true;
    parser->function->top->instruction.ppredicate     = $<text>6;

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};



cmpOpToken : TOKEN_EQ | TOKEN_NE | TOKEN_LT | TOKEN_LE | TOKEN_GT | TOKEN_GE |
    TOKEN_LO | TOKEN_LS | TOKEN_HI | TOKEN_HS | TOKEN_EQU | TOKEN_NEU |
    TOKEN_LTU | TOKEN_GTU | TOKEN_GEU | TOKEN_NUM | TOKEN_NAN ;
cmpOp : cmpOpToken {
    parser->function->top->instruction.set_token($<vsigned>1);
};

boolOpToken : TOKEN_AND | TOKEN_OR | TOKEN_XOR ;
boolOp : boolOpToken {
    parser->function->top->instruction.set_token($<vsigned>1);
}
optionalBoolOp: /* */ | boolOp ;

ftz : TOKEN_FTZ {
    parser->function->top->instruction.ftz = true;
};
optionalFTZ : /* */ | ftz ;

Qpred : TOKEN_IDENTIFIER {
    parser->function->top->instruction.has_qpredicate = true;
    parser->function->top->instruction.qpredicate = $<text>1;
};
optionalQpred : /* */ | TOKEN_PIPE Qpred ;

negatedOperand : TOKEN_NOT {
    parser->operand.negated = true;
};
optionalNegatedOperand : /* */ | negatedOperand ;
optionalCpred : /* */ | TOKEN_COMMA optionalNegatedOperand operand ;

setp : OPCODE_SETP cmpOp optionalBoolOp optionalFTZ dataType TOKEN_IDENTIFIER
        optionalQpred TOKEN_COMMA identifierOperand TOKEN_COMMA
        immedOrVarOperand optionalCpred {
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->function->top->instruction.has_ppredicate = true;
    parser->function->top->instruction.ppredicate     = $<text>6;

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

shl : OPCODE_SHL dataType identifierOperand TOKEN_COMMA identifierOperand
        TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

shr : OPCODE_SHR dataType identifierOperand TOKEN_COMMA identifierOperand
        TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

optionalApprox : /* */ | TOKEN_APPROX ;

sin : OPCODE_SIN optionalApprox optionalFTZ TOKEN_F32 identifierOperand
        TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.approximation = approximate;

    parser->set_type(TOKEN_F32);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

slct : OPCODE_SLCT optionalFTZ dataType dataType identifierOperand TOKEN_COMMA
        immedOrVarOperand TOKEN_COMMA immedOrVarOperand TOKEN_COMMA
        immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);

    parser->set_type($<vsigned>3);
    parser->function->top->instruction.type = parser->get_type();
    parser->set_type($<vsigned>4);
    parser->function->top->instruction.type2 = parser->get_type();

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

sqrt : OPCODE_SQRT TOKEN_APPROX optionalFTZ TOKEN_F32 identifierOperand
        TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.approximation = approximate;

    parser->set_type(TOKEN_F32);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

sqrt : OPCODE_SQRT fRounding optionalFTZ TOKEN_F32 identifierOperand
        TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->set_type(TOKEN_F32);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

sqrt : OPCODE_SQRT fRounding TOKEN_F64 identifierOperand TOKEN_COMMA
        identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->set_type(TOKEN_F64);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

stCacheOp           : TOKEN_WB | TOKEN_CG | TOKEN_CS | TOKEN_WT ;
optionalSTCacheOp   : /* */ | stCacheOp ;

stDestination : TOKEN_LBRACKET addressableOperand TOKEN_RBRACKET ;

stDestination : TOKEN_LBRACKET immediateDecimal TOKEN_RBRACKET {
    parser->operands.push_back(parser->operand);
    parser->operand.reset();
};

identifierOperand : TOKEN_IDENTIFIER {
    operand_t op;
    op.op_type = operand_identifier;
    op.identifier.push_back($<text>1);
    op.field.push_back(field_none);

    parser->operands.push_back(op);
};

st : OPCODE_ST optionalVolatile optionalSpace optionalSTCacheOp
        optionalVectorType dataType stDestination TOKEN_COMMA lValue {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

subType : dataType ;
subType : saturating dataType ;

sub : OPCODE_SUB optionalCarryOut subType identifierOperand TOKEN_COMMA
        identifierOperand TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

subc : OPCODE_SUBC optionalCarryOut subType identifierOperand TOKEN_COMMA
        immedOrVarOperand TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

suld : ;
suq : ;
sured : ;
sust : ;

testpOpToken : TOKEN_FINITE | TOKEN_INFINITE | TOKEN_NUMBER |
    TOKEN_NOTANUMBER | TOKEN_NORMAL | TOKEN_SUBNORMAL ;

testp : OPCODE_TESTP testpOpToken floatingDataType identifierOperand
        TOKEN_COMMA identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.set_token($<vsigned>2);

    parser->function->top->instruction.type = parser->get_type();

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

texGeomToken : TOKEN_1D | TOKEN_2D | TOKEN_3D | TOKEN_A1D | TOKEN_A2D |
    TOKEN_CUBE | TOKEN_ACUBE ;
tex : OPCODE_TEX texGeomToken TOKEN_V4 dataType dataType lValue TOKEN_COMMA
        TOKEN_LBRACKET identifierOperand TOKEN_COMMA optionalIdentifier
        lValue TOKEN_RBRACKET {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.set_geometry($<vsigned>2);
    parser->function->top->instruction.set_token($<vsigned>3);

    parser->set_type($<vsigned>4);
    parser->function->top->instruction.type = parser->get_type();
    parser->set_type($<vsigned>5);
    parser->function->top->instruction.type2 = parser->get_type();

    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

tld4 : ;

trap : OPCODE_TRAP {
    parser->function->top->instruction.set_token($<vsigned>1);
};

txq : ;
vabsdiff : ;
vadd : ;
vmad : ;
vmax : ;
vmin : ;

voteModeToken : TOKEN_ALL | TOKEN_ANY | TOKEN_UNI ;
vote : OPCODE_VOTE voteModeToken TOKEN_PRED identifierOperand TOKEN_COMMA
        optionalNegatedOperand identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.set_token($<vsigned>2);
    parser->set_type($<vsigned>3);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

vote : OPCODE_VOTE TOKEN_BALLOT TOKEN_B32 identifierOperand TOKEN_COMMA
        optionalNegatedOperand identifierOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->function->top->instruction.set_token($<vsigned>2);
    parser->set_type($<vsigned>3);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
}

vset : ;
vshl : ;
vshr : ;
vsub : ;

xorType : TOKEN_PRED | TOKEN_B16 | TOKEN_B32 | TOKEN_B64 ;
xor : OPCODE_XOR xorType identifierOperand TOKEN_COMMA identifierOperand
        TOKEN_COMMA immedOrVarOperand {
    parser->function->top->instruction.set_token($<vsigned>1);
    parser->set_type($<vsigned>2);
    parser->function->top->instruction.type = parser->get_type();
    parser->function->top->instruction.set_operands(parser->operands);
    parser->operands.clear();
};

/* End of grammar. */
%%

int yylex(YYSTYPE * token, YYLTYPE * location, panoptes::ptx_lexer * lexer, panoptes::ptx_parser_state * parser) {
    lexer->yylval = token;
    return lexer->yylex();
}

void yyerror(YYLTYPE * location, panoptes::ptx_lexer * lexer, panoptes::ptx_parser_state * parser, const char * message) {
    /* TODO */
}

}
