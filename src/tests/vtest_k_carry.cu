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
#include <valgrind/memcheck.h>

template<typename T>
static __device__ __inline__ T wide_add(const T & a, const T & b) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__device__ __inline__ uint2 wide_add(const uint2 & a, const uint2 & b) {
    uint2 ret;
    asm volatile(
        "add.cc.u32 %0, %2, %4;\n"
        "addc.u32 %1, %3, %5;\n" : "=r"(ret.x), "=r"(ret.y) :
        "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
    return ret;
}

template<typename T>
__global__ void k_wide_add(T * d, const T a, const T b) {
    *d = wide_add(a, b);
}

TEST(CarryTest, AddSingle) {
    cudaError_t ret;
    cudaStream_t stream;

    uint2 * d;
    ret = cudaMalloc((void **) &d, sizeof(*d));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const uint2 a   = make_uint2(0xFFFFFFFF, 0x0);
    const uint2 b   = make_uint2(0x00000002, 0x0);
    const uint2 exp = make_uint2(0x00000001, 0x1);

    k_wide_add<<<1, 1, 0, stream>>>(d, a, b);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint2 hd;
    BOOST_STATIC_ASSERT(sizeof(hd) == sizeof(*d));

    ret = cudaMemcpy(&hd, d, sizeof(*d), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * TODO:  Do not suppress validity bits.
     */
    (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&hd, sizeof(hd));
    EXPECT_EQ(exp.x, hd.x);
    EXPECT_EQ(exp.y, hd.y);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

/**
 * TODO:  Add a validity check to see that we propagate invalid bits
 * during a carry operation.
 */

template<typename T>
static __device__ __inline__ T wide_sub(const T & a, const T & b) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__device__ __inline__ uint2 wide_sub(const uint2 & a, const uint2 & b) {
    uint2 ret;
    asm volatile(
        "sub.cc.u32 %0, %2, %4;\n"
        "subc.u32 %1, %3, %5;\n" : "=r"(ret.x), "=r"(ret.y) :
        "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
    return ret;
}

template<typename T>
__global__ void k_wide_sub(T * d, const T a, const T b) {
    *d = wide_sub(a, b);
}

TEST(CarryTest, SubSingle) {
    cudaError_t ret;
    cudaStream_t stream;

    uint2 * d;
    ret = cudaMalloc((void **) &d, sizeof(*d));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const uint2 a   = make_uint2(0x0000000F, 0x1);
    const uint2 b   = make_uint2(0x00000010, 0x0);
    const uint2 exp = make_uint2(0xFFFFFFFF, 0x0);

    k_wide_sub<<<1, 1, 0, stream>>>(d, a, b);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint2 hd;
    BOOST_STATIC_ASSERT(sizeof(hd) == sizeof(*d));

    ret = cudaMemcpy(&hd, d, sizeof(*d), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * TODO:  Do not suppress validity bits.
     */
    (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&hd, sizeof(hd));
    EXPECT_EQ(exp.x, hd.x);
    EXPECT_EQ(exp.y, hd.y);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

template<typename S, typename T>
static __device__ __inline__ S wide_mul(const T & a, const T & b) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__device__ __forceinline__ uint4 wide_mul(const uint2 & a, const uint2 & b) {
    /**
     * This is based on the extended precision multiplication PTX given
     * in the PTX version 3.0 ISA documentation for madc.
     *
     * [r3, r2, r1, r0] = [r5, r4] * [r7, r6]
     */
    uint4 ret;
    asm volatile(
        "mul.lo.u32 %0, %4, %6;\n"
        "mul.hi.u32 %1, %4, %6;\n"
        "mad.lo.cc.u32 %1, %5, %6, %1;\n"
        "madc.hi.u32 %2, %5, %6, 0;\n"
        "mad.lo.cc.u32 %1, %4, %7, %1;\n"
        "madc.hi.cc.u32 %2, %4, %7, %2;\n"
        "addc.u32 %3, 0, 0;\n"
        "mad.lo.cc.u32 %2, %5, %7, %2;\n"
        "madc.hi.u32 %3, %5, %7, %3;\n" :
        "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) :
        "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
    return ret;
}

template<typename S, typename T>
__global__ void k_wide_mul(S * d, const T a, const T b) {
    *d = wide_mul<S, T>(a, b);
}

TEST(CarryTest, MulSingle) {
    cudaError_t ret;
    cudaStream_t stream;

    uint4 * d;
    ret = cudaMalloc((void **) &d, sizeof(*d));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const uint2 a   = make_uint2(0x00000002, 0x00000001);
    const uint2 b   = make_uint2(0xFFFFFFFF, 0x00000000);
    const uint4 exp = make_uint4(0xFFFFFFFE, 0x00000000, 0x1, 0x0);

    k_wide_mul<<<1, 1, 0, stream>>>(d, a, b);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint4 hd;
    BOOST_STATIC_ASSERT(sizeof(hd) == sizeof(*d));

    ret = cudaMemcpy(&hd, d, sizeof(*d), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * TODO:  Do not suppress validity bits.
     */
    (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&hd, sizeof(hd));
    EXPECT_EQ(exp.x, hd.x);
    EXPECT_EQ(exp.y, hd.y);
    EXPECT_EQ(exp.z, hd.z);
    EXPECT_EQ(exp.w, hd.w);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
