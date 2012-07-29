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

extern "C" __global__ void k_divmod(const int * a_values, const int * b_values,
        const int N, int * div_values, int * mod_values) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        const int a = a_values[idx];
        const int b = b_values[idx];

        div_values[idx] = a / b;
        mod_values[idx] = a % b;
    }
}

extern "C" __global__ void k_fdiv(const float * a_values,
        const float * b_values, const int N, float * div_values) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        const int a = a_values[idx];
        const int b = b_values[idx];

        div_values[idx] = a / b;
    }
}

extern "C" __global__ void k_fdivf(const float * a_values,
        const float * b_values, const int N, float * div_values) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        const int a = a_values[idx];
        const int b = b_values[idx];

        div_values[idx] = __fdividef(a, b);
    }
}

TEST(kDIVMOD, ExplicitStream) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    int * a_data;
    int * b_data;
    int * div_values;
    int * mod_values;

    ret = cudaMalloc((void **) &a_data, sizeof(*a_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &b_data, sizeof(*b_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &div_values, sizeof(*div_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &mod_values, sizeof(*mod_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_divmod<<<256, n_blocks, 0, stream>>>(a_data, b_data, N, div_values,
        mod_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(a_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(b_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(div_values);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(mod_values);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kDIVMOD, FloatingPoint) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    float * a_data;
    float * b_data;
    float * div_values;

    ret = cudaMalloc((void **) &a_data, sizeof(*a_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &b_data, sizeof(*b_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &div_values, sizeof(*div_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_fdiv<<<256, n_blocks, 0, stream>>>(a_data, b_data, N, div_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(a_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(b_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(div_values);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kDIVMOD, FloatingPointFast) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    float * a_data;
    float * b_data;
    float * div_values;

    ret = cudaMalloc((void **) &a_data, sizeof(*a_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &b_data, sizeof(*b_data) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &div_values, sizeof(*div_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_fdivf<<<256, n_blocks, 0, stream>>>(a_data, b_data, N, div_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(a_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(b_data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(div_values);
    ASSERT_EQ(cudaSuccess, ret);
}

template<typename T>
class DivisionFixture : public ::testing::Test {
public:
    DivisionFixture() { }
    ~DivisionFixture() { }

    void SetUp() {
        cudaError_t ret;
        ret = cudaStreamCreate(&stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    void TearDown() {
        cudaError_t ret;
        ret = cudaStreamDestroy(stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    cudaStream_t stream;
};

TYPED_TEST_CASE_P(DivisionFixture);

template<typename T>
static __global__ void k_divA5(T * out, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
static __global__ void k_divA5<>(int16_t * out, const int16_t a) {
    int16_t _out;
    asm("div.s16 %0, %1, 5;\n" : "=h"(_out) : "h"(a));
    *out = _out;
}

template<>
static __global__ void k_divA5<>(uint16_t * out, const uint16_t a) {
    uint16_t _out;
    asm("div.u16 %0, %1, 5;\n" : "=h"(_out) : "h"(a));
    *out = _out;
}

template<>
static __global__ void k_divA5<>(int32_t * out, const int32_t a) {
    int32_t _out;
    asm("div.s32 %0, %1, 5;\n" : "=r"(_out) : "r"(a));
    *out = _out;
}

template<>
static __global__ void k_divA5<>(uint32_t * out, const uint32_t a) {
    uint32_t _out;
    asm("div.u32 %0, %1, 5;\n" : "=r"(_out) : "r"(a));
    *out = _out;
}

template<>
static __global__ void k_divA5<>(int64_t * out, const int64_t a) {
    int64_t _out;
    asm("div.s64 %0, %1, 5;\n" : "=l"(_out) : "l"(a));
    *out = _out;
}

template<>
static __global__ void k_divA5<>(uint64_t * out, const uint64_t a) {
    uint64_t _out;
    asm("div.u64 %0, %1, 5;\n" : "=l"(_out) : "l"(a));
    *out = _out;
}

template<>
static __global__ void k_divA5<>(float * out, const float a) {
    float _out;
    asm("div.approx.f32 %0, %1, 0f40A00000;\n" : "=f"(_out) : "f"(a));
    *out = _out;
}

template<>
static __global__ void k_divA5<>(double * out, const double a) {
    double _out;
    asm("div.rn.f64 %0, %1, 0d4014000000000000;\n" : "=d"(_out) : "d"(a));
    *out = _out;
}

template<typename T>
static __global__ void k_div10B(T * out, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
static __global__ void k_div10B(int16_t * out, const int16_t a) {
    int16_t _out;
    asm("div.s16 %0, 10, %1;\n" : "=h"(_out) : "h"(a));
    *out = _out;
}

template<>
static __global__ void k_div10B(uint16_t * out, const uint16_t a) {
    uint16_t _out;
    asm("div.u16 %0, 10, %1;\n" : "=h"(_out) : "h"(a));
    *out = _out;
}

template<>
static __global__ void k_div10B(int32_t * out, const int32_t a) {
    int32_t _out;
    asm("div.s32 %0, 10, %1;\n" : "=r"(_out) : "r"(a));
    *out = _out;
}

template<>
static __global__ void k_div10B(uint32_t * out, const uint32_t a) {
    uint32_t _out;
    asm("div.u32 %0, 10, %1;\n" : "=r"(_out) : "r"(a));
    *out = _out;
}

template<>
static __global__ void k_div10B(int64_t * out, const int64_t a) {
    int64_t _out;
    asm("div.s64 %0, 10, %1;\n" : "=l"(_out) : "l"(a));
    *out = _out;
}

template<>
static __global__ void k_div10B(uint64_t * out, const uint64_t a) {
    uint64_t _out;
    asm("div.u64 %0, 10, %1;\n" : "=l"(_out) : "l"(a));
    *out = _out;
}

template<>
static __global__ void k_div10B(float * out, const float a) {
    float _out;
    asm("div.approx.f32 %0, 0f41200000, %1;\n" : "=f"(_out) : "f"(a));
    *out = _out;
}

template<>
static __global__ void k_div10B(double * out, const double a) {
    double _out;
    asm("div.rn.f64 %0, 0d4024000000000000, %1;\n" : "=d"(_out) : "d"(a));
    *out = _out;
}

template<typename T>
static __global__ void k_div105(T * out) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
static __global__ void k_div105(int16_t * out) {
    int16_t _out;
    asm("div.s16 %0, 10, 5;\n" : "=h"(_out));
    *out = _out;
}

template<>
static __global__ void k_div105(uint16_t * out) {
    uint16_t _out;
    asm("div.u16 %0, 10, 5;\n" : "=h"(_out));
    *out = _out;
}

template<>
static __global__ void k_div105(int32_t * out) {
    int32_t _out;
    asm("div.s32 %0, 10, 5;\n" : "=r"(_out));
    *out = _out;
}

template<>
static __global__ void k_div105(uint32_t * out) {
    uint32_t _out;
    asm("div.u32 %0, 10, 5;\n" : "=r"(_out));
    *out = _out;
}

template<>
static __global__ void k_div105(int64_t * out) {
    int64_t _out;
    asm("div.s64 %0, 10, 5;\n" : "=l"(_out));
    *out = _out;
}

template<>
static __global__ void k_div105(uint64_t * out) {
    uint64_t _out;
    asm("div.u64 %0, 10, 5;\n" : "=l"(_out));
    *out = _out;
}

template<>
static __global__ void k_div105(float * out) {
    float _out;
    asm("div.approx.f32 %0, 0f41200000, 0f40A00000;\n" : "=f"(_out));
    *out = _out;
}

template<>
static __global__ void k_div105(double * out) {
    double _out;
    asm("div.rn.f64 %0, 0d4024000000000000, 0d4014000000000000;\n" : "=d"(_out));
    *out = _out;
}

template<size_t N>
struct unsigned_of {
    typedef void type;
    BOOST_STATIC_ASSERT(N == 0);
};

template<>
struct unsigned_of<2> {
    typedef uint16_t type;
};

template<>
struct unsigned_of<4> {
    typedef uint32_t type;
};

template<>
struct unsigned_of<8> {
    typedef uint64_t type;
};

TYPED_TEST_P(DivisionFixture, ConstantDivision) {
    cudaError_t ret;

    TypeParam * out;
    ret = cudaMalloc((void **) &out, 5 * sizeof(*out));
    assert(cudaSuccess == ret);

    const TypeParam a = 10;
    const TypeParam b =  5;
    TypeParam a_invalid = a;
    TypeParam b_invalid = b;

    VALGRIND_MAKE_MEM_UNDEFINED(&a_invalid, sizeof(a_invalid));
    VALGRIND_MAKE_MEM_UNDEFINED(&b_invalid, sizeof(b_invalid));

    k_divA5 <<<1, 1, 0, this->stream>>>(out + 0, a);
    k_divA5 <<<1, 1, 0, this->stream>>>(out + 1, a_invalid);
    k_div10B<<<1, 1, 0, this->stream>>>(out + 2, b);
    k_div10B<<<1, 1, 0, this->stream>>>(out + 3, b_invalid);
    k_div105<<<1, 1, 0, this->stream>>>(out + 4);

    ret = cudaStreamSynchronize(this->stream);
    assert(cudaSuccess == ret);

    TypeParam hout[5];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == ret);

    ret = cudaFree(out);
    assert(cudaSuccess == ret);

    const TypeParam expected = a / b;
    assert(expected == hout[0]);
    assert(expected == hout[2]);
    assert(expected == hout[4]);

    typedef typename unsigned_of<sizeof(TypeParam)>::type unsigned_t;
    BOOST_STATIC_ASSERT(sizeof(unsigned_t) == sizeof(TypeParam));
    unsigned_t vout[5];
    BOOST_STATIC_ASSERT(sizeof(hout) == sizeof(vout));
    const int vret = VALGRIND_GET_VBITS(hout, vout, sizeof(vout));
    if (vret == 1) {
        const unsigned_t invalid = static_cast<unsigned_t>(-1);

        assert(      0 == vout[0]);
        assert(invalid == vout[1]);
        assert(      0 == vout[2]);
        assert(invalid == vout[3]);
        assert(      0 == vout[4]);
    }
}

REGISTER_TYPED_TEST_CASE_P(DivisionFixture, ConstantDivision);

typedef ::testing::Types<int16_t, uint16_t, int32_t, uint32_t, int64_t,
    uint64_t, float, double> MyTypes;
INSTANTIATE_TYPED_TEST_CASE_P(My, DivisionFixture, MyTypes);

template<typename T>
class RemainderFixture : public ::testing::Test {
public:
    RemainderFixture() { }
    ~RemainderFixture() { }

    void SetUp() {
        cudaError_t ret;
        ret = cudaStreamCreate(&stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    void TearDown() {
        cudaError_t ret;
        ret = cudaStreamDestroy(stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    cudaStream_t stream;
};

TYPED_TEST_CASE_P(RemainderFixture);

template<typename T>
static __global__ void k_remA5(T * out, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
static __global__ void k_remA5<>(int16_t * out, const int16_t a) {
    int16_t _out;
    asm("rem.s16 %0, %1, 5;\n" : "=h"(_out) : "h"(a));
    *out = _out;
}

template<>
static __global__ void k_remA5<>(uint16_t * out, const uint16_t a) {
    uint16_t _out;
    asm("rem.u16 %0, %1, 5;\n" : "=h"(_out) : "h"(a));
    *out = _out;
}

template<>
static __global__ void k_remA5<>(int32_t * out, const int32_t a) {
    int32_t _out;
    asm("rem.s32 %0, %1, 5;\n" : "=r"(_out) : "r"(a));
    *out = _out;
}

template<>
static __global__ void k_remA5<>(uint32_t * out, const uint32_t a) {
    uint32_t _out;
    asm("rem.u32 %0, %1, 5;\n" : "=r"(_out) : "r"(a));
    *out = _out;
}

template<>
static __global__ void k_remA5<>(int64_t * out, const int64_t a) {
    int64_t _out;
    asm("rem.s64 %0, %1, 5;\n" : "=l"(_out) : "l"(a));
    *out = _out;
}

template<>
static __global__ void k_remA5<>(uint64_t * out, const uint64_t a) {
    uint64_t _out;
    asm("rem.u64 %0, %1, 5;\n" : "=l"(_out) : "l"(a));
    *out = _out;
}

template<typename T>
static __global__ void k_rem10B(T * out, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
static __global__ void k_rem10B(int16_t * out, const int16_t a) {
    int16_t _out;
    asm("rem.s16 %0, 10, %1;\n" : "=h"(_out) : "h"(a));
    *out = _out;
}

template<>
static __global__ void k_rem10B(uint16_t * out, const uint16_t a) {
    uint16_t _out;
    asm("rem.u16 %0, 10, %1;\n" : "=h"(_out) : "h"(a));
    *out = _out;
}

template<>
static __global__ void k_rem10B(int32_t * out, const int32_t a) {
    int32_t _out;
    asm("rem.s32 %0, 10, %1;\n" : "=r"(_out) : "r"(a));
    *out = _out;
}

template<>
static __global__ void k_rem10B(uint32_t * out, const uint32_t a) {
    uint32_t _out;
    asm("rem.u32 %0, 10, %1;\n" : "=r"(_out) : "r"(a));
    *out = _out;
}

template<>
static __global__ void k_rem10B(int64_t * out, const int64_t a) {
    int64_t _out;
    asm("rem.s64 %0, 10, %1;\n" : "=l"(_out) : "l"(a));
    *out = _out;
}

template<>
static __global__ void k_rem10B(uint64_t * out, const uint64_t a) {
    uint64_t _out;
    asm("rem.u64 %0, 10, %1;\n" : "=l"(_out) : "l"(a));
    *out = _out;
}

template<typename T>
static __global__ void k_rem105(T * out) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
static __global__ void k_rem105(int16_t * out) {
    int16_t _out;
    asm("rem.s16 %0, 10, 5;\n" : "=h"(_out));
    *out = _out;
}

template<>
static __global__ void k_rem105(uint16_t * out) {
    uint16_t _out;
    asm("rem.u16 %0, 10, 5;\n" : "=h"(_out));
    *out = _out;
}

template<>
static __global__ void k_rem105(int32_t * out) {
    int32_t _out;
    asm("rem.s32 %0, 10, 5;\n" : "=r"(_out));
    *out = _out;
}

template<>
static __global__ void k_rem105(uint32_t * out) {
    uint32_t _out;
    asm("rem.u32 %0, 10, 5;\n" : "=r"(_out));
    *out = _out;
}

template<>
static __global__ void k_rem105(int64_t * out) {
    int64_t _out;
    asm("rem.s64 %0, 10, 5;\n" : "=l"(_out));
    *out = _out;
}

template<>
static __global__ void k_rem105(uint64_t * out) {
    uint64_t _out;
    asm("rem.u64 %0, 10, 5;\n" : "=l"(_out));
    *out = _out;
}

TYPED_TEST_P(RemainderFixture, ConstantDivision) {
    cudaError_t ret;

    TypeParam * out;
    ret = cudaMalloc((void **) &out, 5 * sizeof(*out));
    assert(cudaSuccess == ret);

    const TypeParam a = 10;
    const TypeParam b =  5;
    TypeParam a_invalid = a;
    TypeParam b_invalid = b;

    VALGRIND_MAKE_MEM_UNDEFINED(&a_invalid, sizeof(a_invalid));
    VALGRIND_MAKE_MEM_UNDEFINED(&b_invalid, sizeof(b_invalid));

    k_remA5 <<<1, 1, 0, this->stream>>>(out + 0, a);
    k_remA5 <<<1, 1, 0, this->stream>>>(out + 1, a_invalid);
    k_rem10B<<<1, 1, 0, this->stream>>>(out + 2, b);
    k_rem10B<<<1, 1, 0, this->stream>>>(out + 3, b_invalid);
    k_rem105<<<1, 1, 0, this->stream>>>(out + 4);

    ret = cudaStreamSynchronize(this->stream);
    assert(cudaSuccess == ret);

    TypeParam hout[5];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == ret);

    ret = cudaFree(out);
    assert(cudaSuccess == ret);

    const TypeParam expected = a % b;
    assert(expected == hout[0]);
    assert(expected == hout[2]);
    assert(expected == hout[4]);

    typedef typename unsigned_of<sizeof(TypeParam)>::type unsigned_t;
    BOOST_STATIC_ASSERT(sizeof(unsigned_t) == sizeof(TypeParam));
    unsigned_t vout[5];
    BOOST_STATIC_ASSERT(sizeof(hout) == sizeof(vout));
    const int vret = VALGRIND_GET_VBITS(hout, vout, sizeof(vout));
    if (vret == 1) {
        const unsigned_t invalid = static_cast<unsigned_t>(-1);

        assert(      0 == vout[0]);
        assert(invalid == vout[1]);
        assert(      0 == vout[2]);
        assert(invalid == vout[3]);
        assert(      0 == vout[4]);
    }
}

REGISTER_TYPED_TEST_CASE_P(RemainderFixture, ConstantDivision);

typedef ::testing::Types<int16_t, uint16_t, int32_t, uint32_t, int64_t,
    uint64_t> IntTypes;
INSTANTIATE_TYPED_TEST_CASE_P(My, RemainderFixture, IntTypes);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
