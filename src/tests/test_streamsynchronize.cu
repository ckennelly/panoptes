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

#include <cuda.h>
#include <gtest/gtest.h>

TEST(StreamSynchronize, InvalidStream) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    cudaStream_t stream;

    /**
     * Without this compound statement, EXPECT_EXIT fails.
     * Without the EXPECT_EXIT wrapper, SIGSEGV happens.
     *
     * From appearances, it seems that gtest does not properly execute
     * both cudaStreamCreate and cudaStreamDestroy on stream when entering
     * the death test for this lone statement as to properly "initialize"
     * stream.
     */
    EXPECT_EXIT(
        {
            cudaStreamCreate(&stream);
            cudaStreamDestroy(stream);
            cudaStreamSynchronize(stream);
        },
        ::testing::KilledBySignal(SIGSEGV), "");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
