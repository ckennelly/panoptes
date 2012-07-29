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

extern "C" __global__ void k_minmax(const int * data, const int N,
        int * min_value, int * max_value) {
    int local_max = INT_MIN;
    int local_min = INT_MAX;

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        const int d = data[idx];

        local_max = max(local_max, d);
        local_min = min(local_min, d);
    }

    max_value[blockIdx.x] = local_max;
    min_value[blockIdx.x] = local_min;
}

TEST(kMinMax, ExplicitStream) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    int * data;
    int * max_values;
    int * min_values;

    ret = cudaMalloc((void **) &data, sizeof(*data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &max_values, n_blocks);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &min_values, n_blocks);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_minmax<<<256, n_blocks, 0, stream>>>(data, N, max_values, min_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(max_values);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(min_values);
    ASSERT_EQ(cudaSuccess, ret);
}

static __global__ void k_max_A5(int * out, int A) {
    int _out;
    asm("max.s32 %0, %1, 5;\n" : "=r"(_out) : "r"(A));
    *out = _out;
}

static __global__ void k_max_5B(int * out, int B) {
    int _out;
    asm("max.s32 %0, 5, %1;\n" : "=r"(_out) : "r"(B));
    *out = _out;
}

static __global__ void k_max_57(int * out) {
    int _out;
    asm("max.s32 %0, 5, 7;\n" : "=r"(_out));
    *out = _out;
}

static __global__ void k_min_A5(int * out, int A) {
    int _out;
    asm("min.s32 %0, %1, 5;\n" : "=r"(_out) : "r"(A));
    *out = _out;
}

static __global__ void k_min_5B(int * out, int B) {
    int _out;
    asm("min.s32 %0, 5, %1;\n" : "=r"(_out) : "r"(B));
    *out = _out;
}

static __global__ void k_min_57(int * out) {
    int _out;
    asm("min.s32 %0, 5, 7;\n" : "=r"(_out));
    *out = _out;
}

TEST(kMinMax, Constants) {
    cudaError_t ret;
    cudaStream_t stream;

    int * out;
    ret = cudaMalloc((void **) &out, 10 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const int A = 10;
    const int B = 2;
    int A_invalid = A;
    int B_invalid = B;
    VALGRIND_MAKE_MEM_UNDEFINED(&A_invalid, sizeof(A_invalid));
    VALGRIND_MAKE_MEM_UNDEFINED(&B_invalid, sizeof(B_invalid));

    k_max_A5<<<1, 1, 0, stream>>>(out + 0, A);
    k_max_A5<<<1, 1, 0, stream>>>(out + 1, A_invalid);
    k_max_5B<<<1, 1, 0, stream>>>(out + 2, B);
    k_max_5B<<<1, 1, 0, stream>>>(out + 3, B_invalid);
    k_max_57<<<1, 1, 0, stream>>>(out + 4);
    k_min_A5<<<1, 1, 0, stream>>>(out + 5, A);
    k_min_A5<<<1, 1, 0, stream>>>(out + 6, A_invalid);
    k_min_5B<<<1, 1, 0, stream>>>(out + 7, B);
    k_min_5B<<<1, 1, 0, stream>>>(out + 8, B_invalid);
    k_min_57<<<1, 1, 0, stream>>>(out + 9);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    int hout[10];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(std::max(A, 5), hout[0]);
    EXPECT_EQ(std::max(5, B), hout[2]);
    EXPECT_EQ(std::max(5, 7), hout[4]);
    EXPECT_EQ(std::min(A, 5), hout[5]);
    EXPECT_EQ(std::min(5, B), hout[7]);
    EXPECT_EQ(std::min(5, 7), hout[9]);

    uint32_t vout[10];
    BOOST_STATIC_ASSERT(sizeof(hout) == sizeof(vout));
    const int vret = VALGRIND_GET_VBITS(hout, vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout[0]);
        EXPECT_EQ(0xFFFFFFFF, vout[1]);
        EXPECT_EQ(0x00000000, vout[2]);
        EXPECT_EQ(0xFFFFFFFF, vout[3]);
        EXPECT_EQ(0x00000000, vout[4]);
        EXPECT_EQ(0x00000000, vout[5]);
        EXPECT_EQ(0xFFFFFFFF, vout[6]);
        EXPECT_EQ(0x00000000, vout[7]);
        EXPECT_EQ(0xFFFFFFFF, vout[8]);
        EXPECT_EQ(0x00000000, vout[9]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
