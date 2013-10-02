%{
/**
 * Panoptes - A Binary Translation Framework for CUDA
 * (c) 2011-2013 Chris Kennelly <chris@ckennelly.com>
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

#ifndef __PANOPTES__PTX_LEXER_LL__
#define __PANOPTES__PTX_LEXER_LL__

#include <boost/lexical_cast.hpp>
#include <sstream>
#include <stdint.h>
#include <ptx_io/ptx_grammar.h>
#include <ptx_io/ptx_lexer.h>
#include <ptx_io/ptx_parser.h>

namespace {
    size_t strlcpy(char * dst, const char * src, size_t size);
}

%}

%option c++
%option yyclass="panoptes::ptx_lexer"
%option yylineno
%option prefix="ptx"

DIGIT            [0-9]
DIGIT_OCT        [0-7]
DIGIT_NONZERO    [1-9]
CONSTANT_DECIMAL ("-"?{DIGIT_NONZERO}[0-9]*)
CONSTANT_HEX     (0[xX][[:xdigit::]]+)
/* TODO:  This does not support decimal representations of floating point numbers */
CONSTANT_FLOAT   (0[dDfF][[:xdigit:]]{8})
CONSTANT_DOUBLE  (0[dDfF][[:xdigit:]]{16})
CONSTANT_OCT     (0{DIGIT_OCT}*)

IDENTIFIER_TAIL  [[:alnum:]_$]
IDENTIFIER       ([[:alpha:]]{IDENTIFIER_TAIL}*|[_$%]{IDENTIFIER_TAIL}*)

STRING           \"(\\.|[^\\"])*\"

NEW_LINE         ([\n]*)
TAB              [\t]*
SPACE            [ ]*
WHITESPACE       [ \t]*
COMMENT ("/*"([^*]|"*"[^/])*"*/")|("/"(\\\n)*"/"[^\n]*)

%%

"abs"               { yylval->vsigned = OPCODE_ABS;         return OPCODE_ABS;      }
"add"               { yylval->vsigned = OPCODE_ADD;         return OPCODE_ADD;      }
"addc"              { yylval->vsigned = OPCODE_ADDC;        return OPCODE_ADDC;     }
"and"               { yylval->vsigned = OPCODE_AND;         return OPCODE_AND;      }
"atom"              { yylval->vsigned = OPCODE_ATOM;        return OPCODE_ATOM;     }
"bar"               { yylval->vsigned = OPCODE_BAR;         return OPCODE_BAR;      }
"bfe"               { yylval->vsigned = OPCODE_BFE;         return OPCODE_BFE;      }
"bfi"               { yylval->vsigned = OPCODE_BFI;         return OPCODE_BFI;      }
"bfind"             { yylval->vsigned = OPCODE_BFIND;       return OPCODE_BFIND;    }
"bra"               { yylval->vsigned = OPCODE_BRA;         return OPCODE_BRA;      }
"brev"              { yylval->vsigned = OPCODE_BREV;        return OPCODE_BREV;     }
"brkpt"             { yylval->vsigned = OPCODE_BRKPT;       return OPCODE_BRKPT;    }
"call"              { yylval->vsigned = OPCODE_CALL;        return OPCODE_CALL;     }
"clz"               { yylval->vsigned = OPCODE_CLZ;         return OPCODE_CLZ;      }
"cnot"              { yylval->vsigned = OPCODE_CNOT;        return OPCODE_CNOT;     }
"copysign"          { yylval->vsigned = OPCODE_COPYSIGN;    return OPCODE_COPYSIGN; }
"cos"               { yylval->vsigned = OPCODE_COS;         return OPCODE_COS;      }
"cvt"               { yylval->vsigned = OPCODE_CVT;         return OPCODE_CVT;      }
"cvta"              { yylval->vsigned = OPCODE_CVTA;        return OPCODE_CVTA;     }
"div"               { yylval->vsigned = OPCODE_DIV;         return OPCODE_DIV;      }
"ex2"               { yylval->vsigned = OPCODE_EX2;         return OPCODE_EX2;      }
"exit"              { yylval->vsigned = OPCODE_EXIT;        return OPCODE_EXIT;     }
"fma"               { yylval->vsigned = OPCODE_FMA;         return OPCODE_FMA;      }
"isspacep"          { yylval->vsigned = OPCODE_ISSPACEP;    return OPCODE_ISSPACEP; }
"ld"                { yylval->vsigned = OPCODE_LD;          return OPCODE_LD;       }
"ldu"               { yylval->vsigned = OPCODE_LDU;         return OPCODE_LDU;      }
"lg2"               { yylval->vsigned = OPCODE_LG2;         return OPCODE_LG2;      }
"mad"               { yylval->vsigned = OPCODE_MAD;         return OPCODE_MAD;      }
"mad24"             { yylval->vsigned = OPCODE_MAD24;       return OPCODE_MAD24;    }
"madc"              { yylval->vsigned = OPCODE_MADC;        return OPCODE_MADC;     }
"max"               { yylval->vsigned = OPCODE_MAX;         return OPCODE_MAX;      }
"membar"            { yylval->vsigned = OPCODE_MEMBAR;      return OPCODE_MEMBAR;   }
"min"               { yylval->vsigned = OPCODE_MIN;         return OPCODE_MIN;      }
"mov"               { yylval->vsigned = OPCODE_MOV;         return OPCODE_MOV;      }
"mul"               { yylval->vsigned = OPCODE_MUL;         return OPCODE_MUL;      }
"mul24"             { yylval->vsigned = OPCODE_MUL24;       return OPCODE_MUL24;    }
"neg"               { yylval->vsigned = OPCODE_NEG;         return OPCODE_NEG;      }
"not"               { yylval->vsigned = OPCODE_NOT;         return OPCODE_NOT;      }
"or"                { yylval->vsigned = OPCODE_OR;          return OPCODE_OR;       }
"pmevent"           { yylval->vsigned = OPCODE_PMEVENT;     return OPCODE_PMEVENT;  }
"popc"              { yylval->vsigned = OPCODE_POPC;        return OPCODE_POPC;     }
"prefetch"          { yylval->vsigned = OPCODE_PREFETCH;    return OPCODE_PREFETCH; }
"prefetchu"         { yylval->vsigned = OPCODE_PREFETCHU;   return OPCODE_PREFETCHU;}
"prmt"              { yylval->vsigned = OPCODE_PRMT;        return OPCODE_PRMT;     }
"rcp"               { yylval->vsigned = OPCODE_RCP;         return OPCODE_RCP;      }
"red"               { yylval->vsigned = OPCODE_RED;         return OPCODE_RED;      }
"rem"               { yylval->vsigned = OPCODE_REM;         return OPCODE_REM;      }
"ret"               { yylval->vsigned = OPCODE_RET;         return OPCODE_RET;      }
"rsqrt"             { yylval->vsigned = OPCODE_RSQRT;       return OPCODE_RSQRT;    }
"sad"               { yylval->vsigned = OPCODE_SAD;         return OPCODE_SAD;      }
"selp"              { yylval->vsigned = OPCODE_SELP;        return OPCODE_SELP;     }
"set"               { yylval->vsigned = OPCODE_SET;         return OPCODE_SET;      }
"setp"              { yylval->vsigned = OPCODE_SETP;        return OPCODE_SETP;     }
"shl"               { yylval->vsigned = OPCODE_SHL;         return OPCODE_SHL;      }
"shr"               { yylval->vsigned = OPCODE_SHR;         return OPCODE_SHR;      }
"sin"               { yylval->vsigned = OPCODE_SIN;         return OPCODE_SIN;      }
"slct"              { yylval->vsigned = OPCODE_SLCT;        return OPCODE_SLCT;     }
"sqrt"              { yylval->vsigned = OPCODE_SQRT;        return OPCODE_SQRT;     }
"st"                { yylval->vsigned = OPCODE_ST;          return OPCODE_ST;       }
"sub"               { yylval->vsigned = OPCODE_SUB;         return OPCODE_SUB;      }
"subc"              { yylval->vsigned = OPCODE_SUBC;        return OPCODE_SUBC;     }
"suld"              { yylval->vsigned = OPCODE_SULD;        return OPCODE_SULD;     }
"suq"               { yylval->vsigned = OPCODE_SURED;       return OPCODE_SURED;    }
"sust"              { yylval->vsigned = OPCODE_SUST;        return OPCODE_SUST;     }
"testp"             { yylval->vsigned = OPCODE_TESTP;       return OPCODE_TESTP;    }
"tex"               { yylval->vsigned = OPCODE_TEX;         return OPCODE_TEX;      }
"tld4"              { yylval->vsigned = OPCODE_TLD4;        return OPCODE_TLD4;     }
"trap"              { yylval->vsigned = OPCODE_TRAP;        return OPCODE_TRAP;     }
"txq"               { yylval->vsigned = OPCODE_TXQ;         return OPCODE_TXQ;      }
"vabsdiff"          { yylval->vsigned = OPCODE_VABSDIFF;    return OPCODE_VABSDIFF; }
"vadd"              { yylval->vsigned = OPCODE_VADD;        return OPCODE_VADD;     }
"vmad"              { yylval->vsigned = OPCODE_VMAD;        return OPCODE_VMAD;     }
"vmax"              { yylval->vsigned = OPCODE_VMAX;        return OPCODE_VMAX;     }
"vmin"              { yylval->vsigned = OPCODE_VMIN;        return OPCODE_VMIN;     }
"vote"              { yylval->vsigned = OPCODE_VOTE;        return OPCODE_VOTE;     }
"vset"              { yylval->vsigned = OPCODE_VSET;        return OPCODE_VSET;     }
"vshl"              { yylval->vsigned = OPCODE_VSHL;        return OPCODE_VSHL;     }
"vshr"              { yylval->vsigned = OPCODE_VSHR;        return OPCODE_VSHR;     }
"vsub"              { yylval->vsigned = OPCODE_VSUB;        return OPCODE_VSUB;     }
"xor"               { yylval->vsigned = OPCODE_XOR;         return OPCODE_XOR;      }

".align"             { yylval->vsigned = TOKEN_ALIGN;    return TOKEN_ALIGN;    }
".const"             { yylval->vsigned = TOKEN_CONST;    return TOKEN_CONST;    }
".entry"             { yylval->vsigned = TOKEN_ENTRY;    return TOKEN_ENTRY;    }
".extern"            { yylval->vsigned = TOKEN_EXTERN;   return TOKEN_EXTERN;   }
".file"              { yylval->vsigned = TOKEN_FILE;     return TOKEN_FILE;     }
".func"              { yylval->vsigned = TOKEN_FUNCTION; return TOKEN_FUNCTION; }
".global"            { yylval->vsigned = TOKEN_GLOBAL;   return TOKEN_GLOBAL;   }
".local"             { yylval->vsigned = TOKEN_LOCAL;    return TOKEN_LOCAL;    }
".loc"               { yylval->vsigned = TOKEN_LOC;      return TOKEN_LOC;      }
".param"             { yylval->vsigned = TOKEN_PARAM;    return TOKEN_PARAM;    }
".reg"               { yylval->vsigned = TOKEN_REG;      return TOKEN_REG;      }
".section"           { yylval->vsigned = TOKEN_SECTION;  return TOKEN_SECTION;  }
".shared"            { yylval->vsigned = TOKEN_SHARED;   return TOKEN_SHARED;   }
".texref"            { yylval->vsigned = TOKEN_TEXREF;   return TOKEN_TEXREF;   }
".tex"               { yylval->vsigned = TOKEN_TEX;      return TOKEN_TEX;      }
".target"            { yylval->vsigned = TOKEN_TARGET;   return TOKEN_TARGET;   }
".version"           { yylval->vsigned = TOKEN_VERSION;  return TOKEN_VERSION;  }
".visible"           { yylval->vsigned = TOKEN_VISIBLE;  return TOKEN_VISIBLE;  }
".volatile"          { yylval->vsigned = TOKEN_VOLATILE; return TOKEN_VOLATILE; }

"sm_10"              { yylval->vsigned = TOKEN_SM10; return TOKEN_SM10; }
"sm_11"              { yylval->vsigned = TOKEN_SM11; return TOKEN_SM11; }
"sm_12"              { yylval->vsigned = TOKEN_SM12; return TOKEN_SM12; }
"sm_13"              { yylval->vsigned = TOKEN_SM13; return TOKEN_SM13; }
"sm_20"              { yylval->vsigned = TOKEN_SM20; return TOKEN_SM20; }
"sm_21"              { yylval->vsigned = TOKEN_SM21; return TOKEN_SM21; }
"debug"              { yylval->vsigned = TOKEN_DEBUG; return TOKEN_DEBUG; }
"map_f64_to_f32"     { yylval->vsigned = TOKEN_MAP_F64_TO_F32; return TOKEN_MAP_F64_TO_F32; }
"texmode_unified"    { yylval->vsigned = TOKEN_UNIFIED; return TOKEN_UNIFIED; }
"texmode_independent" {yylval->vsigned = TOKEN_INDEPENDENT; return TOKEN_INDEPENDENT; }
".address_size"      { yylval->vsigned = TOKEN_ADDRESS_SIZE; return TOKEN_ADDRESS_SIZE; }

".s8"				 { yylval->vsigned = TOKEN_S8;  return TOKEN_S8;  }
".s16"			     { yylval->vsigned = TOKEN_S16; return TOKEN_S16; }
".s32"				 { yylval->vsigned = TOKEN_S32; return TOKEN_S32; }
".s64"			     { yylval->vsigned = TOKEN_S64; return TOKEN_S64; }
".u8"			     { yylval->vsigned = TOKEN_U8;  return TOKEN_U8;  }
".u16"			     { yylval->vsigned = TOKEN_U16; return TOKEN_U16; }
".u32"				 { yylval->vsigned = TOKEN_U32; return TOKEN_U32; }
".u64"			     { yylval->vsigned = TOKEN_U64; return TOKEN_U64; }
".b8"			     { yylval->vsigned = TOKEN_B8;  return TOKEN_B8;  }
".b16"			     { yylval->vsigned = TOKEN_B16; return TOKEN_B16; }
".b32"			     { yylval->vsigned = TOKEN_B32; return TOKEN_B32; }
".b64"			     { yylval->vsigned = TOKEN_B64; return TOKEN_B64; }
".f16"			     { yylval->vsigned = TOKEN_F16; return TOKEN_F16; }
".f32"			     { yylval->vsigned = TOKEN_F32; return TOKEN_F32; }
".f64"			     { yylval->vsigned = TOKEN_F64; return TOKEN_F64; }
".pred"		         { yylval->vsigned = TOKEN_PRED;return TOKEN_PRED; }

".ca"                { yylval->vsigned = TOKEN_CA;  return TOKEN_CA; }
".cg"                { yylval->vsigned = TOKEN_CG;  return TOKEN_CG; }
".cs"                { yylval->vsigned = TOKEN_CS;  return TOKEN_CS; }
".lu"                { yylval->vsigned = TOKEN_LU;  return TOKEN_LU; }
".cv"                { yylval->vsigned = TOKEN_CV;  return TOKEN_CV; }
".wb"                { yylval->vsigned = TOKEN_WB;  return TOKEN_WB; }
".wt"                { yylval->vsigned = TOKEN_WT;  return TOKEN_WT; }

".eq"                { yylval->vsigned = TOKEN_EQ; return TOKEN_EQ; }
".ne"                { yylval->vsigned = TOKEN_NE; return TOKEN_NE; }
".lt"                { yylval->vsigned = TOKEN_LT; return TOKEN_LT; }
".le"                { yylval->vsigned = TOKEN_LE; return TOKEN_LE; }
".gt"                { yylval->vsigned = TOKEN_GT; return TOKEN_GT; }
".ge"                { yylval->vsigned = TOKEN_GE; return TOKEN_GE; }
".ls"                { yylval->vsigned = TOKEN_LS; return TOKEN_LS; }
".hs"                { yylval->vsigned = TOKEN_HS; return TOKEN_HS; }
".equ"               { yylval->vsigned = TOKEN_EQU; return TOKEN_EQU; }
".neu"               { yylval->vsigned = TOKEN_NEU; return TOKEN_NEU; }
".ltu"               { yylval->vsigned = TOKEN_LTU; return TOKEN_LTU; }
".leu"               { yylval->vsigned = TOKEN_LEU; return TOKEN_LEU; }
".gtu"               { yylval->vsigned = TOKEN_GTU; return TOKEN_GTU; }
".geu"               { yylval->vsigned = TOKEN_GEU; return TOKEN_GEU; }
".num"               { yylval->vsigned = TOKEN_NUM; return TOKEN_NUM; }
".nan"               { yylval->vsigned = TOKEN_NAN; return TOKEN_NAN; }

".and"               { yylval->vsigned = TOKEN_AND;return TOKEN_AND; }
".or"                { yylval->vsigned = TOKEN_OR; return TOKEN_OR; }
".xor"               { yylval->vsigned = TOKEN_XOR;return TOKEN_XOR; }
".popc"              { yylval->vsigned = TOKEN_POPC; return TOKEN_POPC; }

".hi"                { yylval->vsigned = TOKEN_HI; return TOKEN_HI; }
".lo"                { yylval->vsigned = TOKEN_LO; return TOKEN_LO; }
".rn"                { yylval->vsigned = TOKEN_RN; return TOKEN_RN; }
".rm"                { yylval->vsigned = TOKEN_RM; return TOKEN_RM; }
".rz"                { yylval->vsigned = TOKEN_RZ; return TOKEN_RZ; }
".rp"                { yylval->vsigned = TOKEN_RP; return TOKEN_RP; }
".rni"               { yylval->vsigned = TOKEN_RNI;return TOKEN_RNI; }
".rmi"               { yylval->vsigned = TOKEN_RMI;return TOKEN_RMI; }
".rzi"               { yylval->vsigned = TOKEN_RZI;return TOKEN_RZI; }
".rpi"               { yylval->vsigned = TOKEN_RPI;return TOKEN_RPI; }
".sat"               { yylval->vsigned = TOKEN_SAT;return TOKEN_SAT; }
".ftz"               { yylval->vsigned = TOKEN_FTZ;return TOKEN_FTZ; }
".approx"            { yylval->vsigned = TOKEN_APPROX;  return TOKEN_APPROX; }
".full"              { yylval->vsigned = TOKEN_FULL;    return TOKEN_FULL;   }

".uni"               { yylval->vsigned = TOKEN_UNI; return TOKEN_UNI; }
".byte"              { yylval->vsigned = TOKEN_BYTE;return TOKEN_BYTE; }
".wide"              { yylval->vsigned = TOKEN_WIDE;return TOKEN_WIDE; }

".v2"                { yylval->vsigned = TOKEN_V2; return TOKEN_V2; }
".v4"                { yylval->vsigned = TOKEN_V4; return TOKEN_V4; }

".x"                 { yylval->vsigned = TOKEN_X; return TOKEN_X; }
".y"                 { yylval->vsigned = TOKEN_Y; return TOKEN_Y; }
".z"                 { yylval->vsigned = TOKEN_Z; return TOKEN_Z; }
".w"                 { yylval->vsigned = TOKEN_W; return TOKEN_W; }
".r"                 { yylval->vsigned = TOKEN_X; return TOKEN_X; }
".g"                 { yylval->vsigned = TOKEN_Y; return TOKEN_Y; }
".b"                 { yylval->vsigned = TOKEN_Z; return TOKEN_Z; }
".a"                 { yylval->vsigned = TOKEN_W; return TOKEN_W; }

".min"               { yylval->vsigned = TOKEN_MIN; return TOKEN_MIN; }
".max"               { yylval->vsigned = TOKEN_MAX; return TOKEN_MAX; }
".dec"               { yylval->vsigned = TOKEN_DEC; return TOKEN_DEC; }
".inc"               { yylval->vsigned = TOKEN_INC; return TOKEN_INC; }
".add"               { yylval->vsigned = TOKEN_ADD; return TOKEN_ADD; }
".cas"               { yylval->vsigned = TOKEN_CAS; return TOKEN_CAS; }
".exch"              { yylval->vsigned = TOKEN_EXCH;return TOKEN_EXCH; }

".1d"                { yylval->vsigned = TOKEN_1D; return TOKEN_1D; }
".2d"                { yylval->vsigned = TOKEN_2D; return TOKEN_2D; }
".3d"                { yylval->vsigned = TOKEN_3D; return TOKEN_3D; }
".a1d"              { yylval->vsigned = TOKEN_A1D;      return TOKEN_A1D;   }
".a2d"              { yylval->vsigned = TOKEN_A2D;      return TOKEN_A2D;   }
".cube"             { yylval->vsigned = TOKEN_CUBE;     return TOKEN_CUBE;  }
".acube"            { yylval->vsigned = TOKEN_ACUBE;    return TOKEN_ACUBE; }

".sync"              { yylval->vsigned = TOKEN_SYNC; return TOKEN_SYNC; }
".arrive"            { yylval->vsigned = TOKEN_ARRIVE; return TOKEN_ARRIVE; }
".red"               { yylval->vsigned = TOKEN_RED;  return TOKEN_RED; }

".generic"           { yylval->vsigned = TOKEN_GENERIC; return TOKEN_GENERIC; }
".to"                { yylval->vsigned = TOKEN_TO;      return TOKEN_TO; }

".cta"              { yylval->vsigned = TOKEN_MCTA;     return TOKEN_MCTA;  }
".gl"               { yylval->vsigned = TOKEN_MGL;      return TOKEN_MGL;   }
".sys"              { yylval->vsigned = TOKEN_MSYS;     return TOKEN_MSYS;  }

".mask"             { yylval->vsigned = TOKEN_MASK;     return TOKEN_MASK;  }

".all"              { yylval->vsigned = TOKEN_ALL;      return TOKEN_ALL;       }
".any"              { yylval->vsigned = TOKEN_ANY;      return TOKEN_ANY;       }
".ballot"           { yylval->vsigned = TOKEN_BALLOT;   return TOKEN_BALLOT;    }

".cc"               { yylval->vsigned = TOKEN_CARRY;    return TOKEN_CARRY;     }
".shiftamt"         { yylval->vsigned = TOKEN_SHIFTAMT; return TOKEN_SHIFTAMT;  }
".finite"           { yylval->vsigned = TOKEN_FINITE;   return TOKEN_FINITE;    }
".infinite"         { yylval->vsigned = TOKEN_INFINITE; return TOKEN_INFINITE;  }
".number"           { yylval->vsigned = TOKEN_NUMBER;   return TOKEN_NUMBER;    }
".notanumber"       { yylval->vsigned = TOKEN_NOTANUMBER; return TOKEN_NOTANUMBER; }
".normal"           { yylval->vsigned = TOKEN_NORMAL;   return TOKEN_NORMAL;    }
".subnormal"        { yylval->vsigned = TOKEN_SUBNORMAL;return TOKEN_SUBNORMAL; }

".f4e"              { yylval->vsigned = TOKEN_F4E;      return TOKEN_F4E;    }
".b4e"              { yylval->vsigned = TOKEN_B4E;      return TOKEN_B4E;    }
".rc8"              { yylval->vsigned = TOKEN_RC8;      return TOKEN_RC8;    }
".ecl"              { yylval->vsigned = TOKEN_ECL;      return TOKEN_ECL;    }
".ecr"              { yylval->vsigned = TOKEN_ECR;      return TOKEN_ECR;    }
".rc16"             { yylval->vsigned = TOKEN_RC16;     return TOKEN_RC16;   }
".L1"               { yylval->vsigned = TOKEN_L1;       return TOKEN_L1;     }
".L2"               { yylval->vsigned = TOKEN_L2;       return TOKEN_L2;     }

".width"                { yylval->vsigned = TOKEN_WIDTH;    return TOKEN_WIDTH;     }
".height"               { yylval->vsigned = TOKEN_HEIGHT;   return TOKEN_HEIGHT;    }
".depth"                { yylval->vsigned = TOKEN_DEPTH;    return TOKEN_DEPTH;     }
".channel_data_type"    { yylval->vsigned = TOKEN_CDATATYPE;return TOKEN_CDATATYPE; }
".channel_order"        { yylval->vsigned = TOKEN_CORDER;   return TOKEN_CORDER;    }
".normalized_coords"    { yylval->vsigned = TOKEN_NORMCOORD;return TOKEN_NORMCOORD; }
".force_unnormalized_coords" { yylval->vsigned = TOKEN_FUNNORM; return TOKEN_FUNNORM;    }
".filter_mode"          { yylval->vsigned = TOKEN_FILTERMODE;   return TOKEN_FILTERMODE; }
".addr_mode_0"          { yylval->vsigned = TOKEN_ADDRMODE0;    return TOKEN_ADDRMODE0;  }
".addr_mode_1"          { yylval->vsigned = TOKEN_ADDRMODE1;    return TOKEN_ADDRMODE1;  }
".addr_mode_2"          { yylval->vsigned = TOKEN_ADDRMODE2;    return TOKEN_ADDRMODE2;  }

{CONSTANT_DECIMAL}   { yylval->vsigned = boost::lexical_cast<int64_t>(yytext); \
                        return TOKEN_CONSTANT_DECIMAL; }
{CONSTANT_HEX}       { yylval->vsigned = boost::lexical_cast<int64_t>(yytext); \
                        return TOKEN_CONSTANT_DECIMAL; }
{CONSTANT_OCT}       { yylval->vsigned = boost::lexical_cast<int64_t>(yytext); \
                        return TOKEN_CONSTANT_DECIMAL; }

{CONSTANT_FLOAT}     {
/**
 * Why use std::stringstream here?
 *
 * lexical_cast<> has default stream semantics (read: decimal).  We have to
 * manually specify hex.
 */
                       yytext[1] = 'x'; std::stringstream ss; \
                       ss << std::hex << yytext; \
                       ss >> yylval->vunsigned;  \
                       return TOKEN_CONSTANT_FLOAT; }
{CONSTANT_DOUBLE}    { yytext[1] = 'x'; std::stringstream ss; \
                       ss << std::hex << yytext; \
                       ss >> yylval->vunsigned;  \
                       return TOKEN_CONSTANT_DOUBLE; }

{IDENTIFIER}         {
    if (strcmp(yytext, "_") == 0) {
        return TOKEN_UNDERSCORE;
    } else {
        strlcpy(yylval->text, yytext, sizeof(yylval->text));
        return TOKEN_IDENTIFIER;
    }
}

@{IDENTIFIER}        { strlcpy(yylval->text, yytext + 1, sizeof(yylval->text)); return TOKEN_PREDICATE; }
@!{IDENTIFIER}       { strlcpy(yylval->text, yytext + 2, sizeof(yylval->text)); return TOKEN_NEG_PREDICATE; }

{STRING}             { strlcpy(yylval->text, yytext + 1, strlen(yytext) - 2); return TOKEN_STRING; }

{COMMENT}            {  }
{TAB}                {  }
{SPACE}              {  }
{NEW_LINE}           {  }

","                  { yylval->vsigned = TOKEN_COMMA;       return TOKEN_COMMA; }
"."                  { yylval->vsigned = TOKEN_PERIOD;      return TOKEN_PERIOD; }
":"                  { yylval->vsigned = TOKEN_COLON;       return TOKEN_COLON; }
";"                  { yylval->vsigned = TOKEN_SEMICOLON;   return TOKEN_SEMICOLON; }
"{"                  { yylval->vsigned = TOKEN_LBRACE;      return TOKEN_LBRACE; }
"}"                  { yylval->vsigned = TOKEN_RBRACE;      return TOKEN_RBRACE; }
"["                  { yylval->vsigned = TOKEN_LBRACKET;    return TOKEN_LBRACKET; }
"]"                  { yylval->vsigned = TOKEN_RBRACKET;    return TOKEN_RBRACKET; }
"("                  { yylval->vsigned = TOKEN_LPAREN;      return TOKEN_LPAREN; }
")"                  { yylval->vsigned = TOKEN_RPAREN;      return TOKEN_RPAREN; }
"<"                  { yylval->vsigned = TOKEN_LANGLE;      return TOKEN_LANGLE; }
">"                  { yylval->vsigned = TOKEN_RANGLE;      return TOKEN_RANGLE; }
"+"                  { yylval->vsigned = TOKEN_PLUS;        return TOKEN_PLUS; }
"-"                  { yylval->vsigned = TOKEN_MINUS;       return TOKEN_MINUS; }
"!"                  { yylval->vsigned = TOKEN_NOT;         return TOKEN_NOT; }
"="                  { yylval->vsigned = TOKEN_EQUAL;       return TOKEN_EQUAL; }
"|"                  { yylval->vsigned = TOKEN_PIPE;        return TOKEN_PIPE; }

%%

/******************************************************************************/
/* USER CODE                                                                  */

int yyFlexLexer::yywrap() { return 1; }

namespace {
    size_t strlcpy(char * dst, const char * src, size_t size) {
        if (size == 0) {
            return 0;
        }

        const char * start  = src;
        const char * end    = src + size - 1;
        for (; src != end && *src; src++, dst++) {
            *dst = *src;
        }

        *dst = '\0';
        return src - start + 1;
    }
}

#endif

/******************************************************************************/

