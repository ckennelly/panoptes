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

#include <boost/type_traits/make_unsigned.hpp>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

template<typename T>
__global__ void k_shr_single(T * d, const T a, const T b) {
    *d = a >> b;
}

template<typename T>
__global__ void k_shl_single(T * d, const T a, const T b) {
    *d = a << b;
}

/**
 * nvcc generates code that causes ptxas to fail with an error when b is a
 * uint16_t/int16_t.  We convert to an intermediate int32_t to resolve it.
 */
template<>
__global__ void k_shl_single(int16_t * d, const int16_t a, const int16_t b) {
    int32_t _b = b;
    int16_t _d;
    asm volatile("shl.b16 %0, %1, %2;\n" : "=h"(_d) : "h"(a), "r"(_b));
    *d = _d;
}

template<>
__global__ void k_shl_single(uint16_t * d, const uint16_t a,
        const uint16_t b) {
    uint32_t _b = b;
    uint16_t _d;
    asm volatile("shl.b16 %0, %1, %2;\n" : "=h"(_d) : "h"(a), "r"(_b));
    *d = _d;
}

template<typename T>
class ShiftTest : public ::testing::Test {
public:
    ShiftTest() { }
    ~ShiftTest() { }

    void SetUp() {
        cudaError_t ret;
        ret = cudaMalloc((void **) &d, sizeof(*d));
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaStreamCreate(&stream);
        ASSERT_EQ(cudaSuccess, ret);

        a = 5;
        b = 3;
    }

    void TearDown() {
        cudaError_t ret;
        ret = cudaFree(d);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaStreamDestroy(stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    T * d;
    T a;
    T b;
    cudaStream_t stream;
};

typedef ::testing::Types<int16_t, uint16_t, int32_t, uint32_t,
    int64_t, uint64_t> MyTypes;
TYPED_TEST_CASE(ShiftTest, MyTypes);

TYPED_TEST(ShiftTest, Left) {
    k_shl_single<<<1, 1, 0, this->stream>>>(this->d, this->a, this->b);

    /**
     * nvcc and gtest do not play well, so we cannot apparently use the
     * gtest assertions here.
     */
    cudaError_t ret;
    ret = cudaStreamSynchronize(this->stream);
    assert(cudaSuccess == ret);

    TypeParam hd;
    const TypeParam exp = this->a << this->b;
    ret = cudaMemcpy(&hd, this->d, sizeof(hd), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == ret);

    assert(exp == hd);
}

TYPED_TEST(ShiftTest, LeftClamp) {
    /**
     * This exceeds the bit width of the type by one.
     */
    const TypeParam clamp =
        static_cast<TypeParam>(sizeof(TypeParam)) * CHAR_BIT + 1;

    k_shl_single<<<1, 1, 0, this->stream>>>(this->d, this->a, clamp);

    /**
     * nvcc and gtest do not play well, so we cannot apparently use the
     * gtest assertions here.
     */
    cudaError_t ret;
    ret = cudaStreamSynchronize(this->stream);
    assert(cudaSuccess == ret);

    TypeParam hd;
    const TypeParam exp = 0;
    ret = cudaMemcpy(&hd, this->d, sizeof(hd), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == ret);

    assert(exp == hd);
}

TYPED_TEST(ShiftTest, Right) {
    k_shr_single<<<1, 1, 0, this->stream>>>(this->d, this->a, this->b);

    /**
     * nvcc and gtest do not play well, so we cannot apparently use the
     * gtest assertions here.
     */
    cudaError_t ret;
    ret = cudaStreamSynchronize(this->stream);
    assert(cudaSuccess == ret);

    TypeParam hd;
    const TypeParam exp = this->a >> this->b;
    ret = cudaMemcpy(&hd, this->d, sizeof(hd), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == ret);

    assert(exp == hd);
}

template<typename T>
__global__ void k_shr_single_consta(T * d, const uint32_t b) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_shr_single_consta<int16_t>(int16_t * d, const uint32_t b) {
    int16_t _d;
    asm volatile("shr.s16 %0, 5, %1;\n" : "=h"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shr_single_consta<uint16_t>(uint16_t * d, const uint32_t b) {
    uint16_t _d;
    asm volatile("shr.u16 %0, 5, %1;\n" : "=h"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shr_single_consta<int32_t>(int32_t * d, const uint32_t b) {
    int32_t _d;
    asm volatile("shr.s32 %0, 5, %1;\n" : "=r"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shr_single_consta<uint32_t>(uint32_t * d, const uint32_t b) {
    uint32_t _d;
    asm volatile("shr.u32 %0, 5, %1;\n" : "=r"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shr_single_consta<int64_t>(int64_t * d, const uint32_t b) {
    int64_t _d;
    asm volatile("shr.s64 %0, 5, %1;\n" : "=l"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shr_single_consta<uint64_t>(uint64_t * d, const uint32_t b) {
    uint64_t _d;
    asm volatile("shr.u64 %0, 5, %1;\n" : "=l"(_d) : "r"(b));
    *d = _d;
}

TYPED_TEST(ShiftTest, RightConstA) {
    k_shr_single_consta<<<1, 1, 0, this->stream>>>(this->d, this->b);

    /**
     * nvcc and gtest do not play well, so we cannot apparently use the
     * gtest assertions here.
     */
    cudaError_t ret;
    ret = cudaStreamSynchronize(this->stream);
    assert(cudaSuccess == ret);

    TypeParam hd;
    const TypeParam exp = 5 >> this->b;
    ret = cudaMemcpy(&hd, this->d, sizeof(hd), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == ret);

    assert(exp == hd);

    if (RUNNING_ON_VALGRIND) {
        TypeParam b = 3;
        VALGRIND_MAKE_MEM_UNDEFINED(&b, sizeof(b));
        k_shr_single_consta<<<1, 1, 0, this->stream>>>(this->d, b);

        /**
         * nvcc and gtest do not play well, so we cannot apparently use the
         * gtest assertions here.
         */
        ret = cudaStreamSynchronize(this->stream);
        assert(cudaSuccess == ret);

        TypeParam hd;
        ret = cudaMemcpy(&hd, this->d, sizeof(hd), cudaMemcpyDeviceToHost);
        assert(cudaSuccess == ret);

        typedef typename boost::make_unsigned<TypeParam>::type validity_t;

        validity_t vd;
        const int vret = VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd));
        assert(vret != 1 || static_cast<validity_t>(-1) == vd);
    }
}

template<typename T>
__global__ void k_shr_single_constab(T * d) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_shr_single_constab<int16_t>(int16_t * d) {
    int16_t _d;
    asm volatile("shr.s16 %0, 5, 3;\n" : "=h"(_d));
    *d = _d;
}

template<>
__global__ void k_shr_single_constab<uint16_t>(uint16_t * d) {
    uint16_t _d;
    asm volatile("shr.u16 %0, 5, 3;\n" : "=h"(_d));
    *d = _d;
}

template<>
__global__ void k_shr_single_constab<int32_t>(int32_t * d) {
    int32_t _d;
    asm volatile("shr.s32 %0, 5, 3;\n" : "=r"(_d));
    *d = _d;
}

template<>
__global__ void k_shr_single_constab<uint32_t>(uint32_t * d) {
    uint32_t _d;
    asm volatile("shr.u32 %0, 5, 3;\n" : "=r"(_d));
    *d = _d;
}

template<>
__global__ void k_shr_single_constab<int64_t>(int64_t * d) {
    int64_t _d;
    asm volatile("shr.s64 %0, 5, 3;\n" : "=l"(_d));
    *d = _d;
}

template<>
__global__ void k_shr_single_constab<uint64_t>(uint64_t * d) {
    uint64_t _d;
    asm volatile("shr.u64 %0, 5, 3;\n" : "=l"(_d));
    *d = _d;
}

TYPED_TEST(ShiftTest, RightConstAB) {
    k_shr_single_constab<<<1, 1, 0, this->stream>>>(this->d);

    /**
     * nvcc and gtest do not play well, so we cannot apparently use the
     * gtest assertions here.
     */
    cudaError_t ret;
    ret = cudaStreamSynchronize(this->stream);
    assert(cudaSuccess == ret);

    TypeParam hd;
    const TypeParam exp = 5 >> 3;
    ret = cudaMemcpy(&hd, this->d, sizeof(hd), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == ret);

    assert(exp == hd);

    if (RUNNING_ON_VALGRIND) {
        typedef typename boost::make_unsigned<TypeParam>::type validity_t;

        validity_t vd;
        const int vret = VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd));
        assert(vret != 1 || 0 == vd);
    }
}

template<typename T>
__global__ void k_shl_single_consta(T * d, const uint32_t b) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_shl_single_consta<int16_t>(int16_t * d, const uint32_t b) {
    int16_t _d;
    asm volatile("shl.b16 %0, 5, %1;\n" : "=h"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shl_single_consta<uint16_t>(uint16_t * d, const uint32_t b) {
    uint16_t _d;
    asm volatile("shl.b16 %0, 5, %1;\n" : "=h"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shl_single_consta<int32_t>(int32_t * d, const uint32_t b) {
    int32_t _d;
    asm volatile("shl.b32 %0, 5, %1;\n" : "=r"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shl_single_consta<uint32_t>(uint32_t * d, const uint32_t b) {
    uint32_t _d;
    asm volatile("shl.b32 %0, 5, %1;\n" : "=r"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shl_single_consta<int64_t>(int64_t * d, const uint32_t b) {
    int64_t _d;
    asm volatile("shl.b64 %0, 5, %1;\n" : "=l"(_d) : "r"(b));
    *d = _d;
}

template<>
__global__ void k_shl_single_consta<uint64_t>(uint64_t * d, const uint32_t b) {
    uint64_t _d;
    asm volatile("shl.b64 %0, 5, %1;\n" : "=l"(_d) : "r"(b));
    *d = _d;
}

TYPED_TEST(ShiftTest, LeftConstA) {
    k_shl_single_consta<<<1, 1, 0, this->stream>>>(this->d, this->b);

    /**
     * nvcc and gtest do not play well, so we cannot apparently use the
     * gtest assertions here.
     */
    cudaError_t ret;
    ret = cudaStreamSynchronize(this->stream);
    assert(cudaSuccess == ret);

    TypeParam hd;
    const TypeParam exp = 5 << this->b;
    ret = cudaMemcpy(&hd, this->d, sizeof(hd), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == ret);

    assert(exp == hd);

    if (RUNNING_ON_VALGRIND) {
        TypeParam b = 3;
        VALGRIND_MAKE_MEM_UNDEFINED(&b, sizeof(b));
        k_shl_single_consta<<<1, 1, 0, this->stream>>>(this->d, b);

        /**
         * nvcc and gtest do not play well, so we cannot apparently use the
         * gtest assertions here.
         */
        ret = cudaStreamSynchronize(this->stream);
        assert(cudaSuccess == ret);

        TypeParam hd;
        ret = cudaMemcpy(&hd, this->d, sizeof(hd), cudaMemcpyDeviceToHost);
        assert(cudaSuccess == ret);

        typedef typename boost::make_unsigned<TypeParam>::type validity_t;

        validity_t vd;
        const int vret = VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd));
        assert(vret != 1 || static_cast<validity_t>(-1) == vd);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
