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

#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>

texture<int32_t, 1, cudaReadModeElementType> tex_src;

TEST(GetTextureReference, NullArguments) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    EXPECT_EXIT(
        cudaGetTextureReference(NULL, "tex_src"),
        ::testing::KilledBySignal(SIGSEGV), "");

    int version;
    cudaError_t ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);
    if (version >= 5000 /* 5.0 */) {
        EXPECT_EXIT(
            cudaGetTextureReference(NULL, NULL),
            ::testing::KilledBySignal(SIGSEGV), "");
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
