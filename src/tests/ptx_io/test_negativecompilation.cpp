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

#include <gtest/gtest.h>
#include <tests/ptx_io/test_utilities.h>

using namespace panoptes;

class NegativeCompilation : public ::testing::Test {
public:
    ptx_checker checker;
    void SetUp() { }
    void TearDown() { }
};

TEST_F(NegativeCompilation, NegativeOffsets) {
    std::vector<std::string> args;
    args.push_back("-arch");
    args.push_back("sm_12");

    std::string control(
        ".version 1.4\n"
        ".target sm_12, map_f64_to_f32\n"
        ".entry underrun () {\n"
        "    .shared .u32 s[1];\n"
        "    red.shared.and.b32 [s+0], 0;\n"
        "}\n");

    EXPECT_TRUE(checker.check(control, args));

    std::string test(
        ".version 1.4\n"
        ".target sm_12, map_f64_to_f32\n"
        ".entry underrun () {\n"
        "    .shared .u32 s[1];\n"
        "    red.shared.and.b32 [s-4], 0;\n"
        "}\n");

    EXPECT_FALSE(checker.check(test, args));
}

TEST_F(NegativeCompilation, MultidimensionalParameters) {
    std::string control(
        ".version 1.4\n"
        ".target sm_10, map_f64_to_f32\n\n"
        ".entry foo (.param .align 8 .b8 bar[3]) {\n"
        "    exit;\n"
        "}\n");

    EXPECT_TRUE(checker.check(control));

    std::string test(
        ".version 1.4\n"
        ".target sm_10, map_f64_to_f32\n\n"
        ".entry foo (.param .align 8 .b8 bar[3][3]) {\n"
        "    exit;\n"
        "}\n");

    EXPECT_FALSE(checker.check(test));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
