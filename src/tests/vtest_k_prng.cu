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

/**
 * The XOR128 PRNG (G. Marsaglia.  Xorshift RNGs.  J. Stat. Soft., 8:1-6,
 * 2003).
 *
 * Since the state cannot be easily shared, we have each thread maintain its
 * own state.  This will lead to duplicate random numbers.  (We're mostly after
 * testing the performance impact and functionality of Panoptes, so the data
 * itself is less relevant.)
 */
extern "C" __global__ void k_xor128(uint32_t * out, int32_t n) {
    uint32_t x = 123456789;
    uint32_t y = 362436069;
    uint32_t z = 521288629;
    uint32_t w =  88675123;

    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        uint32_t t;
        t = (x ^ (x << 11));
        x = y; y = z; z = w;
        out[i] = w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }
}

TEST(kPRNG, XOR128) {
    cudaError_t ret;
    cudaStream_t stream;

    const int32_t ints = 1 << 20;
    uint32_t * out;
    ret = cudaMalloc((void **) &out, sizeof(*out) * ints);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_xor128<<<256, 16, 0, stream>>>(out, ints);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
