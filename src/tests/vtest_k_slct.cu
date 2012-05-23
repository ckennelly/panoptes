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

template<typename S, typename T>
static __device__ __inline__ T slct(const T & a, const T & b, const S & c) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__device__ __inline__ uint32_t slct(const uint32_t & a, const uint32_t & b,
        const int32_t & c) {
    uint32_t ret;
    asm volatile("slct.u32.s32 %0, %1, %2, %3;\n" :
        "=r"(ret) : "r"(a), "r"(b), "r"(c));
    return ret;
}

template<>
__device__ __inline__ uint32_t slct(const uint32_t & a, const uint32_t & b,
        const float & c) {
    uint32_t ret;
    asm volatile("slct.u32.f32 %0, %1, %2, %3;\n" :
        "=r"(ret) : "r"(a), "r"(b), "f"(c));
    return ret;
}

template<typename S, typename T>
__global__ void k_slct(T * d, const T * a, const T * b, const S * c,
        int N) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N;
            i += blockDim.x * gridDim.x) {
        d[i] = slct(a[i], b[i], c[i]);
    }
}

template<typename S, typename T>
__global__ void k_slct_single(T * d, const T a, const T b, const S c) {
    *d = slct(a, b, c);
}

TEST(SelectTest, Single) {
    cudaError_t ret;
    cudaStream_t stream;

    uint32_t hd[2];
    uint32_t * d;
    BOOST_STATIC_ASSERT(sizeof(hd[0]) == sizeof(d[0]));
    ret = cudaMalloc((void **) &d, sizeof(hd));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const uint32_t a    = 3;
    const uint32_t b    = 4;
    const int32_t  cs   = -5;
    const float    cf   = 1e-5;
    const uint32_t exps = b;
    const uint32_t expf = a;

    k_slct_single<<<1, 1, 0, stream>>>(d + 0, a, b, cs);
    k_slct_single<<<1, 1, 0, stream>>>(d + 1, a, b, cf);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(exps, hd[0]);
    EXPECT_EQ(expf, hd[1]);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

/**
 * TODO:  Add a validity check to see that we propagate invalid bits.
 */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
