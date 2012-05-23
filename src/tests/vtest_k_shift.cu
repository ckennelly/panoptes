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
#include <stdint.h>

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
