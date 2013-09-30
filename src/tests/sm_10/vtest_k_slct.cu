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

__global__ void k_slct_constants(int32_t * d, int32_t a, int32_t b, int c) {
    int32_t ret;

    asm("slct.s32.s32 %0, 3, -5, 0;" : "=r"(ret));
    d[0] = ret;

    asm("slct.s32.s32 %0, %1, -5, 0;" : "=r"(ret) : "r"(a));
    d[1] = ret;

    asm("slct.s32.s32 %0, 3, %1, 0;" : "=r"(ret) : "r"(b));
    d[2] = ret;

    asm("slct.s32.s32 %0, %1, %2, 0;" : "=r"(ret) : "r"(a), "r"(b));
    d[3] = ret;

    asm("slct.s32.s32 %0, 3, -5, %1;" : "=r"(ret) : "r"(c));
    d[4] = ret;
}

__global__ void k_slct_constants(int64_t * d, int64_t a, int64_t b, int c) {
    int64_t ret;

    asm("slct.s64.s32 %0, 3, -5, 0;" : "=l"(ret));
    d[0] = ret;

    asm("slct.s64.s32 %0, %1, -5, 0;" : "=l"(ret) : "l"(a));
    d[1] = ret;

    asm("slct.s64.s32 %0, 3, %1, 0;" : "=l"(ret) : "l"(b));
    d[2] = ret;

    asm("slct.s64.s32 %0, %1, %2, 0;" : "=l"(ret) : "l"(a), "l"(b));
    d[3] = ret;

    asm("slct.s64.s32 %0, 3, -5, %1;" : "=l"(ret) : "r"(c));
    d[4] = ret;
}

TEST(SelectTest, Constants) {
    cudaError_t ret;

    int32_t hout32[15];
    int64_t hout64[15];

    int32_t *out32;
    int64_t *out64;
    ret = cudaMalloc((void **) &out32, sizeof(hout32));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out64, sizeof(hout64));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    int32_t a32, b32, c32;
    int64_t a64, b64;
    a32 = a64 = 3;
    b32 = b64 = -5;
    c32 = 0;

    const uint32_t va32 = 0xDEADBEEF;
    const uint32_t vb32 = 0xFFFE7F7E;
    const uint32_t vc32 = 0x80000000;
    const uint64_t va64 = 0xDEADBEEFDEADBEEFULL;
    const uint64_t vb64 = 0xFFFE7F7EFFFE7F7EULL;

    const int valgrind = VALGRIND_SET_VBITS(&a32, &va32, sizeof(a32));
    (void)               VALGRIND_SET_VBITS(&b32, &vb32, sizeof(b32));
    (void)               VALGRIND_SET_VBITS(&a64, &va64, sizeof(a64));
    (void)               VALGRIND_SET_VBITS(&b64, &vb64, sizeof(b64));
    ASSERT_GE(1, valgrind);

    k_slct_constants<<<1, 1, 0, stream>>>(out32, a32, b32, c32);
    k_slct_constants<<<1, 1, 0, stream>>>(out64, a64, b64, c32);

    c32 = -1;
    k_slct_constants<<<1, 1, 0, stream>>>(out32 + 5, a32, b32, c32);
    k_slct_constants<<<1, 1, 0, stream>>>(out64 + 5, a64, b64, c32);

    if (valgrind) {
        (void) VALGRIND_SET_VBITS(&c32, &vc32, sizeof(c32));

        k_slct_constants<<<1, 1, 0, stream>>>(out32 + 10, a32, b32, c32);
        k_slct_constants<<<1, 1, 0, stream>>>(out64 + 10, a64, b64, c32);
    }

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(hout32, out32, sizeof(hout32), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(hout64, out64, sizeof(hout64), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out32);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out64);
    ASSERT_EQ(cudaSuccess, ret);

    if (valgrind) {
        uint32_t vout32[15];
        uint64_t vout64[15];
        BOOST_STATIC_ASSERT(sizeof(vout32) == sizeof(hout32));
        BOOST_STATIC_ASSERT(sizeof(vout64) == sizeof(hout64));

        VALGRIND_GET_VBITS(hout32, vout32, sizeof(hout32));
        VALGRIND_GET_VBITS(hout64, vout64, sizeof(hout64));

        const uint32_t good32 = 0;
        const uint64_t good64 = 0;
        const uint32_t bad32  = 0xFFFFFFFF;
        const uint64_t bad64  = 0xFFFFFFFFFFFFFFFFULL;
        EXPECT_EQ(good32, vout32[ 0]);
        EXPECT_EQ(va32,   vout32[ 1]);
        EXPECT_EQ(good32, vout32[ 2]);
        EXPECT_EQ(va32,   vout32[ 3]);
        EXPECT_EQ(good32, vout32[ 4]);

        EXPECT_EQ(good32, vout32[ 5]);
        EXPECT_EQ(va32,   vout32[ 6]);
        EXPECT_EQ(good32, vout32[ 7]);
        EXPECT_EQ(va32,   vout32[ 8]);
        EXPECT_EQ(good32, vout32[ 9]);

        EXPECT_EQ(good32, vout32[10]);
        EXPECT_EQ(va32,   vout32[11]);
        EXPECT_EQ(good32, vout32[12]);
        EXPECT_EQ(va32,   vout32[13]);
        EXPECT_EQ(bad32,  vout32[14]);

        EXPECT_EQ(good64, vout64[ 0]);
        EXPECT_EQ(va64,   vout64[ 1]);
        EXPECT_EQ(good64, vout64[ 2]);
        EXPECT_EQ(va64,   vout64[ 3]);
        EXPECT_EQ(good64, vout64[ 4]);

        EXPECT_EQ(good64, vout64[ 5]);
        EXPECT_EQ(va64,   vout64[ 6]);
        EXPECT_EQ(good64, vout64[ 7]);
        EXPECT_EQ(va64,   vout64[ 8]);
        EXPECT_EQ(good64, vout64[ 9]);

        EXPECT_EQ(good64, vout64[10]);
        EXPECT_EQ(va64,   vout64[11]);
        EXPECT_EQ(good64, vout64[12]);
        EXPECT_EQ(va64,   vout64[13]);
        EXPECT_EQ(bad64,  vout64[14]);

        /* Mark memory as defined so it can be checked. */
        VALGRIND_MAKE_MEM_DEFINED(hout32, sizeof(hout32));
        VALGRIND_MAKE_MEM_DEFINED(hout64, sizeof(hout64));
        VALGRIND_MAKE_MEM_DEFINED(&a32,   sizeof(a32));
        VALGRIND_MAKE_MEM_DEFINED(&b32,   sizeof(b32));
        VALGRIND_MAKE_MEM_DEFINED(&a64,   sizeof(a64));
        VALGRIND_MAKE_MEM_DEFINED(&b64,   sizeof(b64));
    }

    EXPECT_EQ(a32, hout32[0]);
    EXPECT_EQ(a32, hout32[1]);
    EXPECT_EQ(a32, hout32[2]);
    EXPECT_EQ(a32, hout32[3]);
    EXPECT_EQ(a32, hout32[4]);

    EXPECT_EQ(a32, hout32[5]);
    EXPECT_EQ(a32, hout32[6]);
    EXPECT_EQ(a32, hout32[7]);
    EXPECT_EQ(a32, hout32[8]);
    EXPECT_EQ(b32, hout32[9]);

    if (valgrind) {
        EXPECT_EQ(a32, hout32[10]);
        EXPECT_EQ(a32, hout32[11]);
        EXPECT_EQ(a32, hout32[12]);
        EXPECT_EQ(a32, hout32[13]);
        EXPECT_EQ(b32, hout32[14]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
