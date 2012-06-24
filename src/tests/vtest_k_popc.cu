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
#include <limits>
#include <stdint.h>
#include <valgrind/memcheck.h>

template<typename T>
static __device__ int popc(const T & t) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
    return 0;
}

template<>
static __device__ int popc<uint32_t>(const uint32_t & t) {
    return __popc(t);
}

template<>
static __device__ int popc<uint64_t>(const uint64_t & t) {
    return __popcll(t);
}

template<typename T>
__global__ void k_popc(const T * data, const int N,
        int * popc_values) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        popc_values[idx] = popc(data[idx]);
    }
}

TEST(kPOPC, POPC) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    uint32_t * data;
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

    uint64_t * data;
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

template<typename T>
static __global__ void k_popc_single(const T t, int * out) {
    *out = popc(t);
}

TEST(kPOPC, Validity) {
    if (!(RUNNING_ON_VALGRIND)) {
        /* Skip the test without Valgrind. */
        return;
    }

    cudaError_t ret;
    cudaStream_t stream;

    uint32_t u32 = std::numeric_limits<uint32_t>::max();
    uint64_t u64 = std::numeric_limits<uint64_t>::max();
    /* Clear validity */
    VALGRIND_MAKE_MEM_UNDEFINED(&u32, sizeof(u32));
    VALGRIND_MAKE_MEM_UNDEFINED(&u64, sizeof(u64));

    int * out;
    ret = cudaMalloc((void **) &out, sizeof(*out) * 2);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_popc_single<<<1, 1, 0, stream>>>(u32, out + 0);
    k_popc_single<<<1, 1, 0, stream>>>(u64, out + 1);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    int hout[2];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t vout[2];
    VALGRIND_GET_VBITS(&hout[0], &vout[0], sizeof(hout[0]));
    VALGRIND_GET_VBITS(&hout[1], &vout[1], sizeof(hout[1]));
    /* The largest value of popc/popcll is 32 and 64 respectively.  All
     * higher bits should be known to be 0. */
    EXPECT_EQ(0x0000003F, vout[0]);
    EXPECT_EQ(0x0000007F, vout[1]);

    VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(hout, sizeof(hout));

    /* Mask off lower bits. */
    EXPECT_EQ(0, hout[0] & 0xFFFFFFC0);
    EXPECT_EQ(0, hout[1] & 0xFFFFFF80);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
