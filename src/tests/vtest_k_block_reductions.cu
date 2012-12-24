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
#include <gtest/gtest.h>
#include <valgrind/memcheck.h>

extern "C" __global__ void k_all_evens(const int * in, bool * out,
        const int N) {
    bool local = true;

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < N;
            idx += blockDim.x * gridDim.x) {
        local = local & (in[idx] % 2 == 0);
    }

    out[blockIdx.x] = __syncthreads_and(local);
}

extern "C" __global__ void k_const_all(bool * out) {
    int tmp;
    asm("{ .reg .pred %tmp;\n"
        "bar.red.and.pred %tmp, 0, 1;\n"
        "selp.s32 %0, 1, 0, %tmp;\n}" : "=r"(tmp));
    out[blockIdx.x] = tmp;
}

extern "C" __global__ void k_any_evens(const int * in, bool * out,
        const int N) {
    bool local = true;

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < N;
            idx += blockDim.x * gridDim.x) {
        local = local & (in[idx] % 2 == 0);
    }

    out[blockIdx.x] = __syncthreads_or(local);
}

extern "C" __global__ void k_const_any(bool * out) {
    int tmp;
    asm("{ .reg .pred %tmp;\n"
        "bar.red.or.pred %tmp, 0, 1;\n"
        "selp.s32 %0, 1, 0, %tmp;\n}" : "=r"(tmp));
    out[blockIdx.x] = tmp;
}

extern "C" __global__ void k_count_evens(const int * in, int * out,
        const int N) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    bool val = false;
    if (idx < N) {
        val = (in[idx] % 2 == 0);
    }

    out[blockIdx.x] = __syncthreads_count(val);
}

extern "C" __global__ void k_const_count(int * out) {
    int tmp;
    asm("bar.red.popc.u32 %0, 0, 1;" : "=r"(tmp));
    out[blockIdx.x] = tmp;
}

TEST(kSyncThreads, AllEvens) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;
    const int block_size = 256;

    int * in;
    bool * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, 2 * sizeof(*out) * n_blocks);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_all_evens<<<n_blocks, block_size, 0, stream>>>(in, out, N);
    k_const_all<<<n_blocks, block_size, 0, stream>>>(out + n_blocks);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    /* Avoid the std::vector<bool> specialization. */
    std::vector<char> hout(2 * n_blocks);
    BOOST_STATIC_ASSERT(sizeof(hout[0]) == sizeof(*out));
    ret = cudaMemcpy(&hout[0], out, 2 * sizeof(*out) * n_blocks,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    if (RUNNING_ON_VALGRIND) {
        std::vector<unsigned char> vout(2 * n_blocks);
        BOOST_STATIC_ASSERT(sizeof(hout[0]) == sizeof(vout[0]));
        int v = VALGRIND_GET_VBITS(&hout[0], &vout[0],
            2 * sizeof(hout[0]) * n_blocks);
        ASSERT_EQ(1, v);

        bool error = false;
        for (int i = 0; i < n_blocks; i++) {
            /**
             * setp is imprecise and marks the validity bits to be entirely
             * invalid, even when the propagation of uninitialized state may
             * cause a single bitflip.
             */
            error |= vout[i] != 0xFF;
        }
        EXPECT_FALSE(error);

        for (int i = n_blocks; i < 2 * n_blocks; i++) {
            error |= vout[i] != 0x0;
        }
        EXPECT_FALSE(error);
    }

    bool error = false;
    for (int i = n_blocks; i < 2 * n_blocks; i++) {
        error |= hout[i] != 1;
    }
    EXPECT_FALSE(error);
}

TEST(kSyncThreads, AnyEvens) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;
    const int block_size = 256;

    int * in;
    bool * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, 2 * sizeof(*out) * n_blocks);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_any_evens<<<n_blocks, block_size, 0, stream>>>(in, out, N);
    k_const_any<<<n_blocks, block_size, 0, stream>>>(out + n_blocks);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    /* Avoid the std::vector<bool> specialization. */
    std::vector<char> hout(2 * n_blocks);
    BOOST_STATIC_ASSERT(sizeof(hout[0]) == sizeof(*out));
    ret = cudaMemcpy(&hout[0], out, 2 * sizeof(*out) * n_blocks,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    if (RUNNING_ON_VALGRIND) {
        std::vector<unsigned char> vout(2 * n_blocks);
        BOOST_STATIC_ASSERT(sizeof(hout[0]) == sizeof(vout[0]));
        int v = VALGRIND_GET_VBITS(&hout[0], &vout[0],
            2 * sizeof(hout[0]) * n_blocks);
        ASSERT_EQ(1, v);

        bool error = false;
        for (int i = 0; i < n_blocks; i++) {
            /**
             * setp is imprecise and marks the validity bits to be entirely
             * invalid, even when the propagation of uninitialized state may
             * cause a single bitflip.
             */
            error |= vout[i] != 0xFF;
        }
        EXPECT_FALSE(error);

        for (int i = n_blocks; i < 2 * n_blocks; i++) {
            error |= vout[i] != 0x0;
        }
        EXPECT_FALSE(error);
    }

    bool error = false;
    for (int i = n_blocks; i < 2 * n_blocks; i++) {
        error |= hout[i] != 1;
    }
    EXPECT_FALSE(error);
}

TEST(kSyncThreads, CountEvens) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int block_size = 256;
    const int n_blocks = (N + block_size - 1) / block_size;

    int * in;
    int * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, 2 * sizeof(*out) * n_blocks);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_count_evens<<<n_blocks, block_size, 0, stream>>>(in, out, N);
    k_const_count<<<n_blocks, block_size, 0, stream>>>(out + n_blocks);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    std::vector<int> hout(2 * n_blocks);
    ret = cudaMemcpy(&hout[0], out, 2 * sizeof(*out) * n_blocks,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    if (RUNNING_ON_VALGRIND) {
        std::vector<unsigned> vout(2 * n_blocks);
        BOOST_STATIC_ASSERT(sizeof(hout[0]) == sizeof(vout[0]));
        int v = VALGRIND_GET_VBITS(&hout[0], &vout[0],
            2 * sizeof(hout[0]) * n_blocks);
        ASSERT_EQ(1, v);

        /* Round block_size to next power of two. */
        unsigned rbs = block_size;
        rbs--;
        rbs |= rbs >> 1;
        rbs |= rbs >> 2;
        rbs |= rbs >> 4;
        rbs |= rbs >> 8;
        rbs |= rbs >> 16;
        const unsigned mask = (rbs + 1) | rbs;

        bool error = false;
        for (int i = 0; i < n_blocks; i++) {
            EXPECT_EQ(mask, vout[i]);
            error |= vout[i] != mask;
        }
        EXPECT_FALSE(error);

        for (int i = n_blocks; i < 2 * n_blocks; i++) {
            error |= vout[i] != 0;
        }
        EXPECT_FALSE(error);
    }

    bool error = false;
    for (int i = n_blocks; i < 2 * n_blocks; i++) {
        error |= hout[i] != block_size;
    }
    EXPECT_FALSE(error);
}

__global__ void k_partial(unsigned * out, const unsigned * _value) {
    const unsigned value = *_value;
    unsigned tmp;

    asm("{.reg .pred %temp;\n"
        "setp.u32.ne %temp, 0, %1;\n"
        "bar.red.popc.u32 %0, 1, 32, %temp;\n}" : "=r"(tmp) : "r"(value));
    out[threadIdx.x] = tmp;
    tmp = 0;
    out += blockDim.x;
    __syncthreads();

    if (blockDim.x % 64 == 0) {
        asm("{.reg .pred %temp;\n"
            "setp.u32.ne %temp, 0, %1;\n"
            "bar.red.popc.u32 %0, 2, 64, %temp;\n}" : "=r"(tmp) : "r"(value));
    }

    out[threadIdx.x] = tmp;
    tmp = 0;
    out += blockDim.x;
    __syncthreads();

    if (blockDim.x % 96 == 0) {
        asm("{.reg .pred %temp;\n"
            "setp.u32.ne %temp, 0, %1;\n"
            "bar.red.popc.u32 %0, 3, 96, %temp;\n}" : "=r"(tmp) : "r"(value));
    }

    out[threadIdx.x] = tmp;
    tmp = 0;
    out += blockDim.x;
    __syncthreads();

    if (blockDim.x % 128 == 0) {
        asm("{.reg .pred %temp;\n"
            "setp.u32.ne %temp, 0, %1;\n"
            "bar.red.popc.u32 %0, 4, 128, %temp;\n}" : "=r"(tmp) : "r"(value));
    }

    out[threadIdx.x] = tmp;
    tmp = 0;
    out += blockDim.x;
    __syncthreads();

    if (blockDim.x % 160 == 0) {
        asm("{.reg .pred %temp;\n"
            "setp.u32.ne %temp, 0, %1;\n"
            "bar.red.popc.u32 %0, 5, 160, %temp;\n}" : "=r"(tmp) : "r"(value));
    }

    out[threadIdx.x] = tmp;
    tmp = 0;
    out += blockDim.x;
    __syncthreads();

    if (blockDim.x % 192 == 0) {
        asm("{.reg .pred %temp;\n"
            "setp.u32.ne %temp, 0, %1;\n"
            "bar.red.popc.u32 %0, 6, 192, %temp;\n}" : "=r"(tmp) : "r"(value));
    }

    out[threadIdx.x] = tmp;
    tmp = 0;
    out += blockDim.x;
    __syncthreads();

    if (blockDim.x % 224 == 0) {
        asm("{.reg .pred %temp;\n"
            "setp.u32.ne %temp, 0, %1;\n"
            "bar.red.popc.u32 %0, 7, 224, %temp;\n}" : "=r"(tmp) : "r"(value));
    }

    out[threadIdx.x] = tmp;
    tmp = 0;
    out += blockDim.x;
    __syncthreads();

    if (blockDim.x % 256 == 0) {
        asm("{.reg .pred %temp;\n"
            "setp.u32.ne %temp, 0, %1;\n"
            "bar.red.popc.u32 %0, 8, 256, %temp;\n}" : "=r"(tmp) : "r"(value));
    }

    out[threadIdx.x] = tmp;
    tmp = 0;
    out += blockDim.x;
}

TEST(BlockReductions, PartialBlocks) {
    cudaError_t ret;
    const size_t block_size = 256;
    const size_t warp_size  = 32;
    const size_t rounds     = block_size / warp_size;

    unsigned * out;
    ret = cudaMalloc((void **) &out, 2 * sizeof(*out) * block_size * rounds);
    ASSERT_EQ(cudaSuccess, ret);

    unsigned * uninitialized;
    ret = cudaMalloc((void **) &uninitialized, sizeof(*uninitialized));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_partial<<<1, block_size, 0, stream>>>(out + block_size * rounds,
        uninitialized);

    unsigned * initialized;
    ret = cudaMalloc((void **) &initialized, sizeof(*initialized));
    ASSERT_EQ(cudaSuccess, ret);

    unsigned tmp = 1;
    cudaMemcpy(initialized, &tmp, sizeof(tmp), cudaMemcpyHostToDevice);
    k_partial<<<1, block_size, 0, stream>>>(out, initialized);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(uninitialized);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(initialized);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    std::vector<unsigned> hout(2 * block_size * rounds);
    ret = cudaMemcpy(&hout[0], out, 2 * sizeof(*out) * block_size * rounds,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    for (size_t round = 0; round < rounds; round++) {
        const unsigned size = (round + 1) * warp_size;
        /*
         * Partial block reductions are fraught with peril.  The best guarantee
         * we have is that at least size theads will yield size in their
         * reduction.
         *
         * For block_size % size != 0, all threads yield 0.
         */
        unsigned e_value, e_count;
        if (block_size % size == 0) {
            e_value     = size;
            e_count     = size;
        } else {
            e_value     = 0;
            e_count     = block_size;
        }

        unsigned count = 0, m_count = 0;
        for (size_t i = 0; i < block_size; i++) {
            const unsigned value = hout[block_size * round + i];
            if (value == e_value) {
                count++;
            } else if (value == e_value - warp_size) {
                m_count++;
            }
        }

        EXPECT_LE(e_count,  count);
        EXPECT_GE(block_size, count);
        EXPECT_EQ(block_size - count, m_count);
    }

    std::vector<unsigned> vout(2 * block_size * rounds);
    int v = VALGRIND_GET_VBITS(&hout[0], &vout[0],
        2 * block_size * rounds * sizeof(hout[0]));
    if (v == 1) {
        /* Values will be initialized. */
        for (size_t round = 0; round < rounds; round++) {
            for (size_t i = 0; i < block_size; i++) {
                EXPECT_EQ(0, vout[round * block_size + i]);
            }
        }

        /* Values will be uninitialized where block_size % width. */
        for (size_t round = 0; round < rounds; round++) {
            const unsigned width = (round + 1) * warp_size;
            unsigned expected;
            if (block_size % width != 0) {
                expected = 0;
            } else if (block_size == width) {
                expected = width - 1;
                expected |= expected >> 1;
                expected |= expected >> 2;
                expected |= expected >> 4;
                expected |= expected >> 8;
                expected |= expected >> 16;
                expected |= expected + 1;
            } else {
                /* No guarantees are made for partial block reductions. */
                continue;
            }

            const size_t offset = round + rounds;
            for (size_t i = 0; i < block_size; i++) {
                EXPECT_EQ(expected, vout[block_size * offset + i]);
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
