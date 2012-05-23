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
#include <stdint.h>

__device__ uint32_t src;

extern "C" __global__ void k_uniform(uint32_t * dst, int32_t ints) {
    const uint32_t value = src;

    for (int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            idx < ints; idx += blockDim.x * gridDim.x) {
        dst[idx] = value;
    }
}

TEST(kUniform, ExplicitStream) {
    cudaError_t ret;
    cudaStream_t stream;

    const int32_t ints = 1 << 10;
    uint32_t *dst;

    ret = cudaMalloc((void **) &dst, sizeof(*dst) * ints);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_uniform<<<1, 1, 0, stream>>>(dst, ints);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(dst);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
