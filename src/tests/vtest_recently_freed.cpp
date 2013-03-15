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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

TEST(Deallocate, RecentlyFreed) {
    /**
     * This test allocates a buffer, frees it, and then tries to copy from it.
     * While this should yield an error, it should not crash the program.
     */
    const size_t N = 256;

    cudaError_t ret;
    char *dptr, *hptr;
    ret = cudaMalloc((void **) &dptr, N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMallocHost((void **) &hptr, N);
    ASSERT_EQ(cudaSuccess, ret);

    /* Control. */
    ret = cudaMemcpy(hptr, dptr, N, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(dptr, hptr, N, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaSuccess, ret);

    /* Free */
    ret = cudaFree(dptr);
    ASSERT_EQ(cudaSuccess, ret);

    /* Test. */
    ret = cudaMemcpy(hptr, dptr, N, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMemcpy(dptr, hptr, N, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaFreeHost(hptr);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
