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

#include <boost/static_assert.hpp>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>
#include <cstdio>

extern "C" __global__ void k_divmod(const int * a_values, const int * b_values,
        const int N, int * div_values, int * mod_values) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        const int a = a_values[idx];
        const int b = b_values[idx];

        div_values[idx] = a / b;
        mod_values[idx] = a % b;
    }
}

extern "C" __global__ void k_fdiv(const float * a_values,
        const float * b_values, const int N, float * div_values) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        const int a = a_values[idx];
        const int b = b_values[idx];

        div_values[idx] = a / b;
    }
}

extern "C" __global__ void k_fdivf(const float * a_values,
        const float * b_values, const int N, float * div_values) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        const int a = a_values[idx];
        const int b = b_values[idx];

        div_values[idx] = __fdividef(a, b);
    }
}

TEST(kDIVMOD, ExplicitStream) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    int * a_data;
    int * b_data;
    int * div_values;
    int * mod_values;

    ret = cudaMalloc((void **) &a_data, sizeof(*a_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &b_data, sizeof(*b_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &div_values, sizeof(*div_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &mod_values, sizeof(*mod_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_divmod<<<256, n_blocks, 0, stream>>>(a_data, b_data, N, div_values,
        mod_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(a_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(b_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(div_values);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(mod_values);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kDIVMOD, FloatingPoint) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    float * a_data;
    float * b_data;
    float * div_values;

    ret = cudaMalloc((void **) &a_data, sizeof(*a_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &b_data, sizeof(*b_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &div_values, sizeof(*div_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_fdiv<<<256, n_blocks, 0, stream>>>(a_data, b_data, N, div_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(a_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(b_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(div_values);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kDIVMOD, FloatingPointFast) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    float * a_data;
    float * b_data;
    float * div_values;

    ret = cudaMalloc((void **) &a_data, sizeof(*a_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &b_data, sizeof(*b_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &div_values, sizeof(*div_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_fdivf<<<256, n_blocks, 0, stream>>>(a_data, b_data, N, div_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(a_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(b_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(div_values);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
