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

#include "fixture_bindtexture2d.h"
#include <gtest/gtest.h>
#include "scoped_allocations.h"
#include <stdint.h>

TEST_F(BindTexture2D, NullArguments) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    cudaError_t ret;

    unsigned width  = prop.textureAlignment;
    unsigned height = prop.textureAlignment;

    scoped_allocation<int32_t> data(width * height);

    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int32_t>();
    size_t offset;
    const size_t pitch = width * sizeof(*data);
    /* Control test. */
    ret = cudaBindTexture2D(&offset, texref, data, &desc, width, height, pitch);
    EXPECT_EQ(cudaSuccess, ret);

    EXPECT_EXIT(
        cudaBindTexture2D(&offset, texref, data, NULL,  width, height, pitch),
        ::testing::KilledBySignal(SIGSEGV), "");

    EXPECT_EXIT(
        cudaBindTexture2D(&offset, texref, NULL, NULL,  width, height, pitch),
        ::testing::KilledBySignal(SIGSEGV), "");

    EXPECT_EXIT(
        cudaBindTexture2D(&offset, NULL,   data, NULL,  width, height, pitch),
        ::testing::KilledBySignal(SIGSEGV), "");

    EXPECT_EXIT(
        cudaBindTexture2D(&offset, NULL,   NULL, NULL,  width, height, pitch),
        ::testing::KilledBySignal(SIGSEGV), "");

    EXPECT_EXIT(
        cudaBindTexture2D(NULL,    texref, data, NULL,  width, height, pitch),
        ::testing::KilledBySignal(SIGSEGV), "");

    EXPECT_EXIT(
        cudaBindTexture2D(NULL,    texref, NULL, NULL,  width, height, pitch),
        ::testing::KilledBySignal(SIGSEGV), "");

    EXPECT_EXIT(
        cudaBindTexture2D(NULL,    NULL,   data, NULL,  width, height, pitch),
        ::testing::KilledBySignal(SIGSEGV), "");

    EXPECT_EXIT(
        cudaBindTexture2D(NULL,    NULL,   NULL, NULL,  width, height, pitch),
        ::testing::KilledBySignal(SIGSEGV), "");

    ret = cudaUnbindTexture(texref);
    EXPECT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
