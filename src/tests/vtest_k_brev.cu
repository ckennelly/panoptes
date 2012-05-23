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

extern "C" __global__ void k_brev(const int32_t * in, int * out, int n) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < n; idx += blockDim.x * gridDim.x) {
        out[idx] = __brev(in[idx]);
    }
}

extern "C" __global__ void k_brevll(const int64_t * in, int * out, int n) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < n; idx += blockDim.x * gridDim.x) {
        out[idx] = __brevll(in[idx]);
    }
}

TEST(kBREV, Int32) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    int32_t * in;
    int * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_brev<<<256, n_blocks, 0, stream>>>(in, out, N);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kBREV, Int64) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    int64_t * in;
    int * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_brevll<<<256, n_blocks, 0, stream>>>(in, out, N);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
