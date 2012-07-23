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

template<typename T>
class BitfieldTestFixture : public ::testing::Test {
public:
    BitfieldTestFixture() { }
    ~BitfieldTestFixture() { }

    void SetUp() {
        cudaError_t ret;
        ret = cudaMalloc((void **) &d, sizeof(*d) * n);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaMalloc((void **) &a, sizeof(*a) * n);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaMalloc((void **) &f, sizeof(*a) * n);
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

        ret = cudaFree(f);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaStreamSynchronize(stream);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaStreamDestroy(stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    static const int32_t n = 1 << 20;
    /**
     * When testing insertions, we use d as field b.
     */
    T * d;
    T * a;
    T * f;
    T tmp;
    cudaStream_t stream;
};

TYPED_TEST_CASE_P(BitfieldTestFixture);

template<typename T>
__global__ void k_bfi(T * f, const T * a, const T * b, uint32_t c,
        uint32_t d, int N) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_bfi(uint32_t * f, const uint32_t * a, const uint32_t * b,
        uint32_t c, uint32_t d, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t _f;
        uint32_t _a = a[idx];
        uint32_t _b = b[idx];
        asm volatile("bfi.b32 %0, %1, %2, %3, %4;\n" : "=r"(_f) : "r"(_a),
            "r"(_b), "r"(c), "r"(d));
        f[idx] = _f;
    }
}

template<>
__global__ void k_bfi(uint64_t * f, const uint64_t * a, const uint64_t * b,
        uint32_t c, uint32_t d, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint64_t _f;
        uint64_t _a = a[idx];
        uint64_t _b = b[idx];
        asm volatile("bfi.b64 %0, %1, %2, %3, %4;\n" : "=l"(_f) : "l"(_a),
            "l"(_b), "r"(c), "r"(d));
        f[idx] = _f;
    }
}

template<>
__global__ void k_bfi(int32_t * f, const int32_t * a, const int32_t * b,
        uint32_t c, uint32_t d, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int32_t _f;
        int32_t _a = a[idx];
        int32_t _b = b[idx];
        asm volatile("bfi.b32 %0, %1, %2, %3, %4;\n" : "=r"(_f) : "r"(_a),
            "r"(_b), "r"(c), "r"(d));
        f[idx] = _f;
    }
}

template<>
__global__ void k_bfi(int64_t * f, const int64_t * a, const int64_t * b,
        uint32_t c, uint32_t d, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int64_t _f;
        int64_t _a = a[idx];
        int64_t _b = b[idx];
        asm volatile("bfi.b64 %0, %1, %2, %3, %4;\n" : "=l"(_f) : "l"(_a),
            "l"(_b), "r"(c), "r"(d));
        f[idx] = _f;
    }
}

TYPED_TEST_P(BitfieldTestFixture, Insert) {
    uint32_t c = 1;
    uint32_t d = 33;
    k_bfi<<<256, 16, 0, this->stream>>>(
        this->f, this->a, this->d, c, d, this->n);
}

template<typename T>
__global__ void k_bfi_const(T * f, const T * a, const T * b, int N) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_bfi_const(uint32_t * f, const uint32_t * a,
        const uint32_t * b, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t _f;
        uint32_t _a = a[idx];
        uint32_t _b = b[idx];
        asm volatile("bfi.b32 %0, %1, %2, 1, 5;\n" : "=r"(_f) :
            "r"(_a), "r"(_b));
        f[idx] = _f;
    }
}

template<>
__global__ void k_bfi_const(uint64_t * f, const uint64_t * a,
        const uint64_t * b, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint64_t _f;
        uint64_t _a = a[idx];
        uint64_t _b = b[idx];
        asm volatile("bfi.b64 %0, %1, %2, 1, 5;\n" : "=l"(_f) :
            "l"(_a), "l"(_b));
        f[idx] = _f;
    }
}

template<>
__global__ void k_bfi_const(int32_t * f, const int32_t * a,
        const int32_t * b, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int32_t _f;
        int32_t _a = a[idx];
        int32_t _b = b[idx];
        asm volatile("bfi.b32 %0, %1, %2, 1, 5;\n" : "=r"(_f) :
            "r"(_a), "r"(_b));
        f[idx] = _f;
    }
}

template<>
__global__ void k_bfi_const(int64_t * f, const int64_t * a,
        const int64_t * b, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int64_t _f;
        int64_t _a = a[idx];
        int64_t _b = b[idx];
        asm volatile("bfi.b64 %0, %1, %2, 1, 5;\n" : "=l"(_f) :
            "l"(_a), "l"(_b));
        f[idx] = _f;
    }
}

TYPED_TEST_P(BitfieldTestFixture, InsertConstant) {
    k_bfi_const<<<256, 16, 0, this->stream>>>(
        this->f, this->a, this->d, this->n);
}

template<typename T>
__global__ void k_bfi_constd(T * f, const T * a, const T * b,
        uint32_t c, int N) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_bfi_constd(uint32_t * f, const uint32_t * a,
        const uint32_t * b, uint32_t c, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t _f;
        uint32_t _a = a[idx];
        uint32_t _b = b[idx];
        asm volatile("bfi.b32 %0, %1, %2, %3, 5;\n" : "=r"(_f) : "r"(_a),
            "r"(_b), "r"(c));
        f[idx] = _f;
    }
}

template<>
__global__ void k_bfi_constd(uint64_t * f, const uint64_t * a,
        const uint64_t * b, uint32_t c, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint64_t _f;
        uint64_t _a = a[idx];
        uint64_t _b = b[idx];
        asm volatile("bfi.b64 %0, %1, %2, %3, 5;\n" : "=l"(_f) : "l"(_a),
            "l"(_b), "r"(c));
        f[idx] = _f;
    }
}

template<>
__global__ void k_bfi_constd(int32_t * f, const int32_t * a,
        const int32_t * b, uint32_t c, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int32_t _f;
        int32_t _a = a[idx];
        int32_t _b = b[idx];
        asm volatile("bfi.b32 %0, %1, %2, %3, 5;\n" : "=r"(_f) : "r"(_a),
            "r"(_b), "r"(c));
        f[idx] = _f;
    }
}

template<>
__global__ void k_bfi_constd(int64_t * f, const int64_t * a,
        const int64_t * b, uint32_t c, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int64_t _f;
        int64_t _a = a[idx];
        int64_t _b = b[idx];
        asm volatile("bfi.b64 %0, %1, %2, %3, 5;\n" : "=l"(_f) : "l"(_a),
            "l"(_b), "r"(c));
        f[idx] = _f;
    }
}

TYPED_TEST_P(BitfieldTestFixture, InsertConstantD) {
    const uint32_t c = 5;
    k_bfi_constd<<<256, 16, 0, this->stream>>>(this->f, this->a, this->d, c,
        this->n);
}

REGISTER_TYPED_TEST_CASE_P(BitfieldTestFixture,
    Insert,  InsertConstant,  InsertConstantD);

typedef ::testing::Types<int32_t, uint32_t, int64_t, uint64_t> MyTypes;
INSTANTIATE_TYPED_TEST_CASE_P(My, BitfieldTestFixture, MyTypes);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
