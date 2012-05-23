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

__device__ unsigned int count = 0;

extern "C" __global__ void k_global_all_evens(const int * in, bool * out,
        bool * reduction_buffer, const int N) {
    bool local = true;

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < N;
            idx += blockDim.x * gridDim.x) {
        local = local & (in[idx] % 2 == 0);
    }

    /**
     * This is the turnstile-based global reduction algorithm described on
     * pages 90-91 (Appendix B) of the CUDA C Programming Guide version 4.1.
     */
    bool block = __syncthreads_and(local);

    __shared__ bool is_last_block_done;

    if (threadIdx.x == 0) {
        reduction_buffer[blockIdx.x] = block;

        /* Make result visible to all other threads. */
        __threadfence();

        /* Enter turnstile. */
        unsigned int value = atomicInc(&count, gridDim.x);

        /* Determine if we're last. */
        is_last_block_done = (value == (gridDim.x - 1));
    }

    /* Wait for the other threads of the block to see the update. */
    __syncthreads();

    if (is_last_block_done) {
        /* Reduce across the results of each block. */
        local = true;
        for (int idx = threadIdx.x; idx < gridDim.x; idx += blockDim.x) {
            local = local & reduction_buffer[idx];
        }

        /* Reduce across all threads. */
        bool global = __syncthreads_and(local);

        /* Write result. */
        if (threadIdx.x == 0) {
            *out = global;
        }
    }
}

extern "C" __global__ void k_global_count_evens(const int * in, int * out,
        const int N) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    bool val = false;
    if (idx < N) {
        val = (in[idx] % 2 == 0);
    }

    int block =  __syncthreads_count(val);
    if (threadIdx.x == 0) {
        /**
         * Since "(void) atomicAdd(out, block)" does not issue a red
         * instruction, we use inline PTX.
         */
        asm("red.global.add.s32 [%0], %1;" : : "l"(out), "r"(block));
    }
}

TEST(kGlobalReduction, AllEvens) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;
    const int block_size = 32; /* Must be warp size. */

    int * in;
    bool * reduction_buffer;
    bool * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &reduction_buffer,
        sizeof(*reduction_buffer) * n_blocks);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_global_all_evens<<<block_size, n_blocks, 0, stream>>>
        (in, out, reduction_buffer, N);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(reduction_buffer);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kGlobalReduction, CountEvens) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int block_size = 256;
    const int n_blocks = 32;

    int * in;
    int * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_global_count_evens<<<block_size, n_blocks, 0, stream>>>(in, out, N);

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
