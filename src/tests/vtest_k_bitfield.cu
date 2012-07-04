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

    static const int32_t n = 1 << 24;
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

template<typename T>
static __device__ __inline__ uint32_t bitfind(const T & a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
    return 0; /* Suppress warning */
}

template<>
__device__ __inline__ uint32_t bitfind(const uint32_t & a) {
    uint32_t ret;
    asm volatile("bfind.u32 %0, %1;\n" : "=r"(ret) : "r"(a));
    return ret;
}

template<>
__device__ __inline__ uint32_t bitfind(const uint64_t & a) {
    uint32_t ret;
    asm volatile("bfind.u64 %0, %1;\n" : "=r"(ret) : "l"(a));
    return ret;
}

template<>
__device__ __inline__ uint32_t bitfind(const int32_t & a) {
    uint32_t ret;
    asm volatile("bfind.s32 %0, %1;\n" : "=r"(ret) : "r"(a));
    return ret;
}

template<>
__device__ __inline__ uint32_t bitfind(const int64_t & a) {
    uint32_t ret;
    asm volatile("bfind.s64 %0, %1;\n" : "=r"(ret) : "l"(a));
    return ret;
}

template<typename T>
static __device__ __inline__ uint32_t bitfindshift(const T & a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
    return 0; /* Suppress warning */
}

template<>
__device__ __inline__ uint32_t bitfindshift(const uint32_t & a) {
    uint32_t ret;
    asm volatile("bfind.shiftamt.u32 %0, %1;\n" : "=r"(ret) : "r"(a));
    return ret;
}

template<>
__device__ __inline__ uint32_t bitfindshift(const uint64_t & a) {
    uint32_t ret;
    asm volatile("bfind.shiftamt.u64 %0, %1;\n" : "=r"(ret) : "l"(a));
    return ret;
}

template<>
__device__ __inline__ uint32_t bitfindshift(const int32_t & a) {
    uint32_t ret;
    asm volatile("bfind.shiftamt.s32 %0, %1;\n" : "=r"(ret) : "r"(a));
    return ret;
}

template<>
__device__ __inline__ uint32_t bitfindshift(const int64_t & a) {
    uint32_t ret;
    asm volatile("bfind.shiftamt.s64 %0, %1;\n" : "=r"(ret) : "l"(a));
    return ret;
}

template<typename T>
__global__ void k_bitfind(uint32_t * d, T a) {
    uint32_t normal = bitfind(a);
    uint32_t shift  = bitfindshift(a);
    d[0] = normal;
    d[1] = sizeof(T) * CHAR_BIT - (1 + shift + normal);
}

TYPED_TEST_P(BitfieldTestFixture, FindSingle) {
    cudaError_t ret;

    this->tmp = 5;
    uint32_t exp[2] = {2, 0};
    uint32_t * d;
    ret = cudaMalloc((void **) &d, sizeof(exp));
    assert(ret == cudaSuccess);

    k_bitfind<<<1, 1, 0, this->stream>>>(d, this->tmp);

    ret = cudaStreamSynchronize(this->stream);
    assert(ret == cudaSuccess);

    uint32_t hd[2];
    BOOST_STATIC_ASSERT(sizeof(hd[0]) == sizeof(d[0]));
    BOOST_STATIC_ASSERT(sizeof(hd) == sizeof(exp));
    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    assert(ret == cudaSuccess);

    assert(hd[0] == exp[0]);
    assert(hd[1] == exp[1]);

    ret = cudaFree(d);
    assert(ret == cudaSuccess);
}

REGISTER_TYPED_TEST_CASE_P(BitfieldTestFixture,
    Extract, ExtractConstant, ExtractConstantC,
    Insert,  InsertConstant,  InsertConstantD,
    FindSingle);

typedef ::testing::Types<int32_t, uint32_t, int64_t, uint64_t> MyTypes;
INSTANTIATE_TYPED_TEST_CASE_P(My, BitfieldTestFixture, MyTypes);

static __global__ void k_bitfind_const(uint4 * out) {
    uint4 _out;
    asm volatile(
        "bfind.u32 %0, 1;\n"
        "bfind.s32 %1, -1;\n"
        "bfind.u64 %2, 4;\n"
        "bfind.s64 %3, -2;\n" : "=r"(_out.x),  "=r"(_out.y), 
                                  "=r"(_out.z),  "=r"(_out.w));
    *out = _out;
}

TEST(Bitfind, Constant) {
    cudaError_t ret;
    cudaStream_t stream;

    uint4 * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_bitfind_const<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint4 hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(0x00000000, hout.x);
    EXPECT_EQ(0xFFFFFFFF, hout.y);
    EXPECT_EQ(0x00000002, hout.z);
    EXPECT_EQ(0x00000000, hout.w);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
