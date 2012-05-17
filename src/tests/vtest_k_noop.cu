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

#include <boost/scoped_array.hpp>
#include <boost/static_assert.hpp>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>
#include <cstdio>

extern "C" __global__ void k_noop() {

}

TEST(kNOOP, ZeroThreads) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_noop<<<1, 0, 0, stream>>>();

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kNOOP, GlobalZeroThreads) {
    cudaError_t ret;

    k_noop<<<1, 0, 0>>>();

    ret = cudaDeviceSynchronize();
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(kNOOP, ZeroBlocks) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_noop<<<0, 1, 0, stream>>>();

    ret = cudaPeekAtLastError();
    EXPECT_EQ(cudaErrorInvalidConfiguration, ret);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaPeekAtLastError();
    EXPECT_EQ(cudaErrorInvalidConfiguration, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaGetLastError();
    EXPECT_EQ(cudaErrorInvalidConfiguration, ret);
}

TEST(kNOOP, GlobalZeroBlocks) {
    cudaError_t ret;

    k_noop<<<0, 1, 0>>>();

    ret = cudaPeekAtLastError();
    EXPECT_EQ(cudaErrorInvalidConfiguration, ret);

    /**
     * Why the error doesn't show up at this call isn't clear.
     */
    ret = cudaDeviceSynchronize();
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaGetLastError();
    EXPECT_EQ(cudaErrorInvalidConfiguration, ret);
}



TEST(kNOOP, ExplicitStream) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_noop<<<1, 1, 0, stream>>>();

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kNOOP, GlobalStream) {
    cudaError_t ret;

    k_noop<<<1, 1, 0>>>();

    ret = cudaDeviceSynchronize();
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(kNOOP, ExplicitGlobalStream) {
    cudaError_t ret;

    k_noop<<<1, 1, 0, NULL>>>();

    ret = cudaDeviceSynchronize();
    EXPECT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
