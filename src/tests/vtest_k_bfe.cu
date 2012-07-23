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
__global__ void k_bfe(T * d, const T * a, uint32_t b, uint32_t c, int N) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_bfe(uint32_t * d, const uint32_t * a, uint32_t b,
        uint32_t c, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t _d;
        uint32_t _a = a[idx];
        asm volatile("bfe.u32 %0, %1, %2, %3;\n" : "=r"(_d) : "r"(_a), "r"(b),
            "r"(c));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe(uint64_t * d, const uint64_t * a, uint32_t b,
        uint32_t c, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint64_t _d;
        uint64_t _a = a[idx];
        asm volatile("bfe.u64 %0, %1, %2, %3;\n" : "=l"(_d) : "l"(_a), "r"(b),
            "r"(c));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe(int32_t * d, const int32_t * a, uint32_t b,
        uint32_t c, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int32_t _d;
        int32_t _a = a[idx];
        asm volatile("bfe.s32 %0, %1, %2, %3;\n" : "=r"(_d) : "r"(_a), "r"(b),
            "r"(c));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe(int64_t * d, const int64_t * a, uint32_t b,
        uint32_t c, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int64_t _d;
        int64_t _a = a[idx];
        asm volatile("bfe.s64 %0, %1, %2, %3;\n" : "=l"(_d) : "l"(_a), "r"(b),
            "r"(c));
        d[idx] = _d;
    }
}

TYPED_TEST_P(BitfieldTestFixture, Extract) {
    uint32_t b = 1;
    uint32_t c = 33;
    k_bfe<<<256, 16, 0, this->stream>>>(this->d, this->a, b, c, this->n);
}

template<typename T>
__global__ void k_bfe_const(T * d, const T * a, int N) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_bfe_const(uint32_t * d, const uint32_t * a, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t _d;
        uint32_t _a = a[idx];
        asm volatile("bfe.u32 %0, %1, 1, 5;\n" : "=r"(_d) : "r"(_a));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_const(uint64_t * d, const uint64_t * a, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint64_t _d;
        uint64_t _a = a[idx];
        asm volatile("bfe.u64 %0, %1, 1, 5;\n" : "=l"(_d) : "l"(_a));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_const(int32_t * d, const int32_t * a, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int32_t _d;
        int32_t _a = a[idx];
        asm volatile("bfe.s32 %0, %1, 1, 5;\n" : "=r"(_d) : "r"(_a));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_const(int64_t * d, const int64_t * a, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int64_t _d;
        int64_t _a = a[idx];
        asm volatile("bfe.s64 %0, %1, 1, 5;\n" : "=l"(_d) : "l"(_a));
        d[idx] = _d;
    }
}

TYPED_TEST_P(BitfieldTestFixture, ExtractConstant) {
    k_bfe_const<<<256, 16, 0, this->stream>>>(this->d, this->a, this->n);
}

template<typename T>
__global__ void k_bfe_constc(T * d, const T * a, uint32_t b, int N) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_bfe_constc(uint32_t * d, const uint32_t * a, uint32_t b,
        int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t _d;
        uint32_t _a = a[idx];
        asm volatile("bfe.u32 %0, %1, %2, 5;\n" : "=r"(_d) : "r"(_a), "r"(b));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_constc(uint64_t * d, const uint64_t * a, uint32_t b,
        int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint64_t _d;
        uint64_t _a = a[idx];
        asm volatile("bfe.u64 %0, %1, %2, 5;\n" : "=l"(_d) : "l"(_a), "r"(b));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_constc(int32_t * d, const int32_t * a, uint32_t b,
        int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int32_t _d;
        int32_t _a = a[idx];
        asm volatile("bfe.s32 %0, %1, %2, 5;\n" : "=r"(_d) : "r"(_a), "r"(b));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_constc(int64_t * d, const int64_t * a, uint32_t b,
        int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int64_t _d;
        int64_t _a = a[idx];
        asm volatile("bfe.s64 %0, %1, %2, 5;\n" : "=l"(_d) : "l"(_a), "r"(b));
        d[idx] = _d;
    }
}

TYPED_TEST_P(BitfieldTestFixture, ExtractConstantC) {
    const uint32_t b = 5;
    k_bfe_constc<<<256, 16, 0, this->stream>>>(this->d, this->a, b, this->n);
}

template<typename T>
__global__ void k_bfe_constbc(T * d, const T * a, int N) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_bfe_constbc(uint32_t * d, const uint32_t * a, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint32_t _d;
        uint32_t _a = a[idx];
        asm volatile("bfe.u32 %0, %1, 1, 5;\n" : "=r"(_d) : "r"(_a));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_constbc(uint64_t * d, const uint64_t * a, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        uint64_t _d;
        uint64_t _a = a[idx];
        asm volatile("bfe.u64 %0, %1, 1, 5;\n" : "=l"(_d) : "l"(_a));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_constbc(int32_t * d, const int32_t * a, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int32_t _d;
        int32_t _a = a[idx];
        asm volatile("bfe.s32 %0, %1, 1, 5;\n" : "=r"(_d) : "r"(_a));
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_constbc(int64_t * d, const int64_t * a, int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        int64_t _d;
        int64_t _a = a[idx];
        asm volatile("bfe.s64 %0, %1, 1, 5;\n" : "=l"(_d) : "l"(_a));
        d[idx] = _d;
    }
}

TYPED_TEST_P(BitfieldTestFixture, ExtractConstantBC) {
    k_bfe_constbc<<<256, 16, 0, this->stream>>>(this->d, this->a, this->n);
}

template<typename T>
__global__ void k_bfe_consta(T * d, uint32_t b, uint32_t c, int N) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_bfe_consta(uint32_t * d, uint32_t b, uint32_t c, int N) {
    uint32_t _d;
    asm volatile("bfe.u32 %0, 3735928559, %1, %2;\n" : "=r"(_d) : "r"(b), "r"(c));

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_consta(int32_t * d, uint32_t b, uint32_t c, int N) {
    int32_t _d;
    asm volatile("bfe.s32 %0, 3735928559, %1, %2;\n" : "=r"(_d) : "r"(b), "r"(c));

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_consta(uint64_t * d, uint32_t b, uint32_t c, int N) {
    uint64_t _d;
    asm volatile("bfe.u64 %0, 3735928559, %1, %2;\n" : "=l"(_d) : "r"(b), "r"(c));

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_consta(int64_t * d, uint32_t b, uint32_t c, int N) {
    int64_t _d;
    asm volatile("bfe.s64 %0, 3735928559, %1, %2;\n" : "=l"(_d) : "r"(b), "r"(c));

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        d[idx] = _d;
    }
}

TYPED_TEST_P(BitfieldTestFixture, ExtractConstantA) {
    const int b = 1;
    const int c = 5;
    k_bfe_consta<<<256, 16, 0, this->stream>>>(this->d, b, c, this->n);
}

template<typename T>
__global__ void k_bfe_constabc(T * d, int N) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_bfe_constabc(uint32_t * d, int N) {
    uint32_t _d;
    asm volatile("bfe.u32 %0, 3735928559, 1, 5;\n" : "=r"(_d));

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_constabc(int32_t * d, int N) {
    int32_t _d;
    asm volatile("bfe.s32 %0, 3735928559, 1, 5;\n" : "=r"(_d));

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_constabc(uint64_t * d, int N) {
    uint64_t _d;
    asm volatile("bfe.u64 %0, 3735928559, 1, 5;\n" : "=l"(_d));

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        d[idx] = _d;
    }
}

template<>
__global__ void k_bfe_constabc(int64_t * d, int N) {
    int64_t _d;
    asm volatile("bfe.s64 %0, 3735928559, 1, 5;\n" : "=l"(_d));

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        d[idx] = _d;
    }
}

TYPED_TEST_P(BitfieldTestFixture, ExtractConstantABC) {
    k_bfe_constabc<<<256, 16, 0, this->stream>>>(this->d, this->n);
}

REGISTER_TYPED_TEST_CASE_P(BitfieldTestFixture,
    Extract, ExtractConstant, ExtractConstantA, ExtractConstantC,
    ExtractConstantABC, ExtractConstantBC);

typedef ::testing::Types<int32_t, uint32_t, int64_t, uint64_t> MyTypes;
INSTANTIATE_TYPED_TEST_CASE_P(My, BitfieldTestFixture, MyTypes);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
