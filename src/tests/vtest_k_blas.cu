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

/**
 * Computes the dot product of x^T y.  The result is reduced on a per block
 * basis to a single value using the reduction technique in the CUDA SDK
 * example "CUDA Parallel Reduction," reduction_kernel.cu, reduce6.
 *
 * out must be a memory location of at least gridDim.x T's.
 *
 * Unlike the SDK example, we require a power of two block size.
 */
template<typename T, unsigned block_size>
__global__ void k_dot(const T * x, const T * y, const int32_t n, T * out) {
    /* Enforce power of two requirement. */
    BOOST_STATIC_ASSERT((block_size & (block_size - 1u)) == 0);

    extern __shared__ T __smem[];
    const uint tid = threadIdx.x;

    T sum = 0.;
    for (int32_t i = tid + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }

    // Store the thread-local sum into shared memory.
    __smem[tid] = sum;
    __syncthreads();

    // Reduce within shared memory.
    if (block_size >= 512) {
        if (tid < 256) {
            __smem[tid] = sum = sum + __smem[tid + 256];
        }
        __syncthreads();
    }

    if (block_size >= 256) {
        if (tid < 128) {
            __smem[tid] = sum = sum + __smem[tid + 128];
        }
        __syncthreads();
    }

    if (block_size >= 128) {
        if (tid < 64) {
            __smem[tid] = sum = sum + __smem[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        // Reduce within the last warp.
        volatile T * smem = __smem;
        if (block_size >= 64) { smem[tid] = sum = sum + smem[tid + 32]; }
        if (block_size >= 32) { smem[tid] = sum = sum + smem[tid + 16]; }
        if (block_size >= 16) { smem[tid] = sum = sum + smem[tid +  8]; }
        if (block_size >=  8) { smem[tid] = sum = sum + smem[tid +  4]; }
        if (block_size >=  4) { smem[tid] = sum = sum + smem[tid +  2]; }
        if (block_size >=  2) { smem[tid] = sum = sum + smem[tid +  1]; }
    }

    // Write the block reduced sum to the output.
    if (tid == 0) {
        out[blockIdx.x] = __smem[0];
    }
}

TEST(kBLAS, DDOT) {
    cudaError_t ret;
    cudaStream_t stream;

    const int32_t n = 1 << 24;
    double *x, *y, *out;

    ret = cudaMalloc((void **) &x, sizeof(*x) * n);
    ASSERT_EQ(cudaSuccess, ret);
    ret = cudaMalloc((void **) &y, sizeof(*y) * n);
    ASSERT_EQ(cudaSuccess, ret);

    const uint n_blocks = 512;
    ret = cudaMalloc((void **) &out, sizeof(*out) * n_blocks);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const uint shmem = sizeof(*out) * n_blocks;

    k_dot<double, 256>
        <<<256, 16, shmem, stream>>>(x, y, n, out);

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

/**
 * Computes y = ax + y.
 */
template<typename T>
__global__ void k_axpy(T * y, const T a, const T * x, int32_t n) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

TEST(kBLAS, AXPY) {
    cudaError_t ret;
    cudaStream_t stream;

    const int32_t n = 1 << 24;
    float *x, *y;
    const float a = 0.5;
    ret = cudaMalloc((void **) &x, sizeof(*x) * n);
    ASSERT_EQ(cudaSuccess, ret);
    ret = cudaMalloc((void **) &y, sizeof(*y) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_axpy<<<256, 16, 0, stream>>>(y, a, x, n);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(x);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(y);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
