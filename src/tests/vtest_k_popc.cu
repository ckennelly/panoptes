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

extern "C" __global__ void k_popc(const unsigned * data, const int N,
        int * popc_values) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        popc_values[idx] = __popc(data[idx]);
    }
}

extern "C" __global__ void k_popcll(const unsigned long long * data,
        const int N, int * popc_values) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        popc_values[idx] = __popc(data[idx]);
    }
}

TEST(kPOPC, POPC) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    unsigned * data;
    int * popc_values;

    ret = cudaMalloc((void **) &data, sizeof(*data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &popc_values, sizeof(*popc_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_popc<<<256, n_blocks, 0, stream>>>(data, N, popc_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(popc_values);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kPOPC, POPCLL) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    unsigned long long * data;
    int * popc_values;

    ret = cudaMalloc((void **) &data, sizeof(*data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &popc_values, sizeof(*popc_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_popcll<<<256, n_blocks, 0, stream>>>(data, N, popc_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(popc_values);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
