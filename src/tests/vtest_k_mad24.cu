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

template<typename T, bool Hi>
__device__ __inline__ T mad24(const T & a, const T & b, const T & c) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<typename T, bool Hi>
__device__ __inline__ int32_t mad24(const int32_t & a, const int32_t & b,
        const int32_t & c) {
    int32_t ret;
    if (Hi) {
        asm volatile("mad24.hi.s32 %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a),
            "r"(b), "r"(c));
    } else {
        asm volatile("mad24.lo.s32 %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a),
            "r"(b), "r"(c));
    }
    return ret;
}

template<typename T, bool Hi>
__device__ __inline__ uint32_t mad24(const uint32_t & a, const uint32_t & b,
        const uint32_t & c) {
    uint32_t ret;
    if (Hi) {
        asm volatile("mad24.hi.u32 %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a),
            "r"(b), "r"(c));
    } else {
        asm volatile("mad24.lo.u32 %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a),
            "r"(b), "r"(c));
    }
    return ret;
}

template<typename T>
__global__ void k_mad24(T * d, const T * a, const T * b, const T * c, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        d[i] = mad24<T, false>(a[i], b[i], c[i]);
    }
}

template<typename T>
__global__ void k_mad24hi(T * d, const T * a, const T * b, const T * c,
        int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        d[i] = mad24<T, true>(a[i], b[i], c[i]);
    }
}

template<typename T>
class MadTest : public ::testing::Test {
public:
    MadTest() { }
    ~MadTest() { }

    void SetUp() {
        cudaError_t ret;
        ret = cudaMalloc((void **) &d, sizeof(*d) * n);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaMalloc((void **) &a, sizeof(*a) * n);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaMalloc((void **) &b, sizeof(*b) * n);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaMalloc((void **) &c, sizeof(*c) * n);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaStreamCreate(&stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    void TearDown() {
        cudaError_t ret;
        ret = cudaFree(d);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaFree(a);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaFree(b);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaFree(c);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaStreamSynchronize(stream);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaStreamDestroy(stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    static const int32_t n = 1 << 24;
    T * d;
    T * a;
    T * b;
    T * c;
    cudaStream_t stream;
};

TYPED_TEST_CASE_P(MadTest);

TYPED_TEST_P(MadTest, Low) {
    k_mad24<<<256, 16, 0, this->stream>>>(this->d, this->a, this->b,
        this->c, this->n);
}

TYPED_TEST_P(MadTest, Hi) {
    k_mad24hi<<<256, 16, 0, this->stream>>>(this->d, this->a, this->b,
        this->c, this->n);
}

REGISTER_TYPED_TEST_CASE_P(MadTest, Low, Hi);

typedef ::testing::Types<int32_t, uint32_t> MyTypes;
INSTANTIATE_TYPED_TEST_CASE_P(My, MadTest, MyTypes);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
