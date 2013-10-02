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

#include <climits>
#include <gtest/gtest.h>
#include <ptx_io/ptx_formatter.h>
#include <ptx_io/ptx_ir.h>
#include <ptx_io/ptx_parser.h>
#include <tests/ptx_io/test_utilities.h>

using namespace panoptes;

class ParserTest : public ::testing::Test {
public:
    ptx_checker validator;
    ptx_t ir;
    ptx_parser parser;

    void SetUp() { }
    void TearDown() { }
};

std::string program1(
    ".version 1.4\n"
    ".target sm_10, map_f64_to_f32\n\n"
    ".entry k(.param .u64 pptr) {\n"
    "   .reg .u64 ptr;\n"
    "   ld.param.u64 ptr, [pptr];\n"
    "   st.global.s32 [ptr], 5;\n"
    "}\n");

TEST_F(ParserTest, PTXtoIR1) {
    EXPECT_TRUE(validator.check(program1));

    parser.parse(program1, &ir);
    EXPECT_EQ(1u, ir.version_major);
    EXPECT_EQ(4u, ir.version_minor);
    EXPECT_EQ(SM10, ir.sm);
    EXPECT_TRUE(ir.map_f64_to_f32);
    /* Unspecified .address_size should default to the host size. */
    EXPECT_EQ(sizeof(void *) * CHAR_BIT, ir.address_size);
    EXPECT_EQ(0u, ir.textures.size());
    EXPECT_EQ(0u, ir.variables.size());

    ASSERT_EQ(1u, ir.entries.size());
    const function_t & entry = *ir.entries[0];
    EXPECT_EQ(linkage_default, entry.linkage);
    EXPECT_TRUE(entry.entry);
    EXPECT_EQ("k", entry.entry_name);
    EXPECT_EQ(&ir, entry.parent);
    EXPECT_FALSE(entry.has_return_value);

    ASSERT_EQ(1u, entry.params.size());
    const param_t & param = entry.params[0];
    EXPECT_EQ(linkage_default, param.linkage);
    EXPECT_EQ(param_space, param.space);
    EXPECT_EQ(v1, param.vector_size);
    EXPECT_EQ(u64_type, param.type);
    EXPECT_FALSE(param.is_ptr);
    EXPECT_EQ("pptr", param.name);
    EXPECT_FALSE(param.has_suffix);
    EXPECT_FALSE(param.is_array);
    EXPECT_FALSE(param.has_initializer);
    EXPECT_FALSE(param.initializer_vector);

    EXPECT_FALSE(entry.no_body);
    const block_t & block = entry.scope;
    ASSERT_EQ(block_scope, block.block_type);
    EXPECT_EQ(NULL, block.parent);
    EXPECT_EQ(&entry, block.fparent);

    const scope_t & scope = *block.scope;
    ASSERT_EQ(1u, scope.variables.size());
    const variable_t & var = scope.variables[0];
    EXPECT_EQ(reg_space, var.space);
    EXPECT_EQ(v1, var.vector_size);
    EXPECT_EQ(u64_type, var.type);
    EXPECT_FALSE(var.is_ptr);
    EXPECT_FALSE(var.has_align);
    EXPECT_EQ("ptr", var.name);
    EXPECT_FALSE(var.has_suffix);
    EXPECT_FALSE(var.is_array);
    EXPECT_FALSE(var.has_initializer);
    EXPECT_FALSE(var.initializer_vector);
    EXPECT_EQ(8u, var.size());

    ASSERT_EQ(2u, scope.blocks.size());

    scope_t::block_vt::const_iterator it = scope.blocks.begin();
    const block_t & block0 = **it;
    ++it;
    const block_t & block1 = **it;

    ASSERT_EQ(block_statement, block0.block_type);
    ASSERT_EQ(block_statement, block1.block_type);

    const statement_t & statement0 = *block0.statement;
    EXPECT_FALSE(statement0.has_predicate);
    EXPECT_FALSE(statement0.is_negated);
    EXPECT_FALSE(statement0.has_ppredicate);
    EXPECT_FALSE(statement0.has_qpredicate);
    EXPECT_FALSE(statement0.saturating);
    EXPECT_FALSE(statement0.uniform);
    EXPECT_EQ(op_ld, statement0.op);
    EXPECT_FALSE(statement0.is_volatile);
    EXPECT_FALSE(statement0.carry_out);
    EXPECT_EQ(param_space, statement0.space);
    EXPECT_EQ(u64_type, statement0.type);
    EXPECT_FALSE(statement0.ftz);
    EXPECT_FALSE(statement0.is_to);
    EXPECT_FALSE(statement0.has_return_value);
    EXPECT_FALSE(statement0.mask);
    EXPECT_FALSE(statement0.shiftamt);

    ASSERT_EQ(2u, statement0.operands.size());
    EXPECT_EQ(operand_t::make_identifier("ptr"), statement0.operands[0]);
    EXPECT_EQ(operand_t::make_addressable("pptr", 0), statement0.operands[1]);

    const statement_t & statement1 = *block1.statement;
    EXPECT_FALSE(statement1.has_predicate);
    EXPECT_FALSE(statement1.is_negated);
    EXPECT_FALSE(statement1.has_ppredicate);
    EXPECT_FALSE(statement1.has_qpredicate);
    EXPECT_FALSE(statement1.saturating);
    EXPECT_FALSE(statement1.uniform);
    EXPECT_EQ(op_st, statement1.op);
    EXPECT_FALSE(statement1.is_volatile);
    EXPECT_FALSE(statement1.carry_out);
    EXPECT_EQ(global_space, statement1.space);
    EXPECT_EQ(s32_type, statement1.type);
    EXPECT_FALSE(statement1.ftz);
    EXPECT_FALSE(statement1.is_to);
    EXPECT_FALSE(statement1.has_return_value);
    EXPECT_FALSE(statement1.mask);
    EXPECT_FALSE(statement1.shiftamt);

    ASSERT_EQ(2u, statement1.operands.size());
    EXPECT_EQ(operand_t::make_addressable("ptr", 0), statement1.operands[0]);
    EXPECT_EQ(operand_t::make_iconstant(5), statement1.operands[1]);
}

TEST_F(ParserTest, FixedPoint1) {
    parser.parse(program1, &ir);

    std::stringstream ss;
    ss << ir;
    std::string program1p = ss.str();

    ptx_t irp;
    parser.parse(program1p, &irp);

    std::stringstream ss2;
    ss2 << irp;
    std::string program1pp = ss2.str();

    EXPECT_EQ(program1p, program1pp);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
