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

__global__ void k_mul24(const int32_t * x, const int32_t * y,
        int32_t * out, int32_t n) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        out[i] = __mul24(x[i], y[i]);
    }
}

__global__ void k_umul24(const uint32_t * x, const uint32_t * y,
        uint32_t * out, int32_t n) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        out[i] = __umul24(x[i], y[i]);
    }
}

TEST(kMul24, Signed) {
    cudaError_t ret;
    cudaStream_t stream;

    const int32_t n = 1 << 24;
    int32_t * x;
    int32_t * y;
    int32_t * out;

    ret = cudaMalloc((void **) &x, sizeof(*x) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &y, sizeof(*y) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_mul24<<<256, 16, 0, stream>>>(x, y, out, n);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(x);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(y);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kMul24, Unsigned) {
    cudaError_t ret;
    cudaStream_t stream;

    const int32_t n = 1 << 24;
    uint32_t * x;
    uint32_t * y;
    uint32_t * out;

    ret = cudaMalloc((void **) &x, sizeof(*x) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &y, sizeof(*y) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_umul24<<<256, 16, 0, stream>>>(x, y, out, n);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(x);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(y);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
