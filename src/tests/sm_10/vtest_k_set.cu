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

__global__ void k_set_with_immediate(uint32_t * out) {
    uint32_t out_;
    asm volatile("set.ne.u32.u32 %0, 0, 0;\n" : "=r"(out_));
    *out = out_;
}

TEST(Regression, SetWithImmediates) {
    cudaError_t ret;

    uint32_t * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_set_with_immediate<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(0x0, hout);
}

__global__ void k_cpred(bool * out) {
    uint32_t out_;

    asm volatile(
        "{\n"
        "  .reg .pred %tmp;\n"
        "  mov.pred %tmp, 1;\n"
        "  set.ne.or.u32.u32 %0, 0, 0, %tmp;\n"
        "}\n" : "=r"(out_));

    *out = out_;
}

TEST(Regression, cPredicate) {
    cudaError_t ret;

    bool * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_cpred<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_TRUE(hout);
}

__global__ void k_negatedoperand(bool * out) {
    uint32_t out_;

    asm volatile(
        "{\n"
        "  .reg .pred %tmp;\n"
        "  mov.pred %tmp, 0;\n"
        "  set.ne.or.u32.u32 %0, 0, 0, !%tmp;\n"
        "}\n" : "=r"(out_));

    *out = out_;
}

TEST(Regression, NegatedOperand) {
    cudaError_t ret;

    bool * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_negatedoperand<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_TRUE(hout);
}

template<typename T>
class SetTypeParameterized : public ::testing::Test { };

TYPED_TEST_CASE_P(SetTypeParameterized);

template<typename T>
__global__ void k_set_single(uint32_t * out, const T a, const T b) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_set_single<int16_t>(uint32_t * out, const int16_t a,
        const int16_t b) {
    uint32_t tmp;
    asm("set.eq.u32.s16 %0, %1, %2;" : "=r"(tmp) : "h"(a), "h"(b));
    *out = tmp;
}

template<>
__global__ void k_set_single<int32_t>(uint32_t * out, const int32_t a,
        const int32_t b) {
    uint32_t tmp;
    asm("set.eq.u32.s32 %0, %1, %2;" : "=r"(tmp) : "r"(a), "r"(b));
    *out = tmp;
}

template<>
__global__ void k_set_single<int64_t>(uint32_t * out, const int64_t a,
        const int64_t b) {
    uint32_t tmp;
    asm("set.eq.u32.s64 %0, %1, %2;" : "=r"(tmp) : "l"(a), "l"(b));
    *out = tmp;
}

template<>
__global__ void k_set_single<uint16_t>(uint32_t * out, const uint16_t a,
        const uint16_t b) {
    uint32_t tmp;
    asm("set.eq.u32.u16 %0, %1, %2;" : "=r"(tmp) : "h"(a), "h"(b));
    *out = tmp;
}

template<>
__global__ void k_set_single<uint32_t>(uint32_t * out, const uint32_t a,
        const uint32_t b) {
    uint32_t tmp;
    asm("set.eq.u32.u32 %0, %1, %2;" : "=r"(tmp) : "r"(a), "r"(b));
    *out = tmp;
}

template<>
__global__ void k_set_single<uint64_t>(uint32_t * out, const uint64_t a,
        const uint64_t b) {
    uint32_t tmp;
    asm("set.eq.u32.u64 %0, %1, %2;" : "=r"(tmp) : "l"(a), "l"(b));
    *out = tmp;
}

template<>
__global__ void k_set_single<float>(uint32_t * out, const float a,
        const float b) {
    uint32_t tmp;
    asm("set.eq.u32.f32 %0, %1, %2;" : "=r"(tmp) : "f"(a), "f"(b));
    *out = tmp;
}

template<>
__global__ void k_set_single<double>(uint32_t * out, const double a,
        const double b) {
    uint32_t tmp;
    asm("set.eq.u32.f64 %0, %1, %2;" : "=r"(tmp) : "d"(a), "d"(b));
    *out = tmp;
}

TYPED_TEST_P(SetTypeParameterized, Single) {
    TypeParam a = 3;
    TypeParam b = 5;

    uint32_t hout;
    uint32_t *out;
    (void) cudaMalloc((void **) &out, sizeof(hout));

    cudaStream_t stream;
    (void) cudaStreamCreate(&stream);

    k_set_single<<<1, 1, 0, stream>>>(out, a, b);

    (void) cudaStreamSynchronize(stream);
    (void) cudaStreamDestroy(stream);
    (void) cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    (void) cudaFree(out);

    assert(hout == 0);
};

template<typename T>
__global__ void k_set_single(uint32_t * out, const T a, const T b, bool c) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_set_single<int16_t>(uint32_t * out, const int16_t a,
        const int16_t b, bool _c) {
    uint32_t tmp;
    const uint32_t c = _c;
    asm("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %3, 0;\n"
        "set.eq.or.u32.s16 %0, %1, %2, %temp; }" : "=r"(tmp) : "h"(a), "h"(b), "r"(c));
    *out = tmp;
}

template<>
__global__ void k_set_single<int32_t>(uint32_t * out, const int32_t a,
        const int32_t b, bool _c) {
    uint32_t tmp;
    const uint32_t c = _c;
    asm("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %3, 0;\n"
        "set.eq.or.u32.s32 %0, %1, %2, %temp; }" : "=r"(tmp) : "r"(a), "r"(b), "r"(c));
    *out = tmp;
}

template<>
__global__ void k_set_single<int64_t>(uint32_t * out, const int64_t a,
        const int64_t b, bool _c) {
    uint32_t tmp;
    const uint32_t c = _c;
    asm("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %3, 0;\n"
        "set.eq.or.u32.s64 %0, %1, %2, %temp; }" : "=r"(tmp) : "l"(a), "l"(b), "r"(c));
    *out = tmp;
}

template<>
__global__ void k_set_single<uint16_t>(uint32_t * out, const uint16_t a,
        const uint16_t b, bool _c) {
    uint32_t tmp;
    const uint32_t c = _c;
    asm("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %3, 0;\n"
        "set.eq.or.u32.u16 %0, %1, %2, %temp; }" : "=r"(tmp) : "h"(a), "h"(b), "r"(c));
    *out = tmp;
}

template<>
__global__ void k_set_single<uint32_t>(uint32_t * out, const uint32_t a,
        const uint32_t b, bool _c) {
    uint32_t tmp;
    const uint32_t c = _c;
    asm("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %3, 0;\n"
        "set.eq.or.u32.u32 %0, %1, %2, %temp; }" : "=r"(tmp) : "r"(a), "r"(b), "r"(c));
    *out = tmp;
}

template<>
__global__ void k_set_single<uint64_t>(uint32_t * out, const uint64_t a,
        const uint64_t b, bool _c) {
    uint32_t tmp;
    const uint32_t c = _c;
    asm("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %3, 0;\n"
        "set.eq.or.u32.u64 %0, %1, %2, %temp; }" : "=r"(tmp) : "l"(a), "l"(b), "r"(c));
    *out = tmp;
}

template<>
__global__ void k_set_single<float>(uint32_t * out, const float a,
        const float b, bool _c) {
    uint32_t tmp;
    const uint32_t c = _c;
    asm("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %3, 0;\n"
        "set.eq.or.u32.f32 %0, %1, %2, %temp; }" : "=r"(tmp) : "f"(a), "f"(b), "r"(c));
    *out = tmp;
}

template<>
__global__ void k_set_single<double>(uint32_t * out, const double a,
        const double b, bool _c) {
    uint32_t tmp;
    const uint32_t c = _c;
    asm("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %3, 0;\n"
        "set.eq.or.u32.f64 %0, %1, %2, %temp; }"
        : "=r"(tmp) : "d"(a), "d"(b), "r"(c));
    *out = tmp;
}

TYPED_TEST_P(SetTypeParameterized, MixInC) {
    TypeParam a = 3;
    TypeParam b = 5;

    uint32_t hout[3];
    uint32_t *out;
    (void) cudaMalloc((void **) &out, sizeof(hout));

    cudaStream_t stream;
    (void) cudaStreamCreate(&stream);

    bool c = true;
    k_set_single<<<1, 1, 0, stream>>>(out + 0, a, b, c);

    c = false;
    k_set_single<<<1, 1, 0, stream>>>(out + 1, a, b, c);

    VALGRIND_MAKE_MEM_UNDEFINED(&c, sizeof(c));
    k_set_single<<<1, 1, 0, stream>>>(out + 2, a, b, c);

    (void) cudaStreamSynchronize(stream);
    (void) cudaStreamDestroy(stream);
    (void) cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    (void) cudaFree(out);

    uint32_t vout[3];
    int valgrind = VALGRIND_GET_VBITS(hout, vout, sizeof(hout));
    assert(valgrind <= 1);
    if (valgrind == 1) {
        assert(vout[0] == 0);
        assert(vout[1] == 0);
        assert(vout[2] == 0xFFFFFFFF);
    }

    assert(hout[0] == 0xFFFFFFFF);
    assert(hout[1] == 0x0);
};

template<typename T>
__global__ void k_set_aonly(uint32_t * out, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_set_aonly<int16_t>(uint32_t * out, const int16_t a) {
    uint32_t tmp;
    asm("set.eq.u32.s16 %0, %1, 5;" : "=r"(tmp) : "h"(a));
    *out = tmp;
}

template<>
__global__ void k_set_aonly<int32_t>(uint32_t * out, const int32_t a) {
    uint32_t tmp;
    asm("set.eq.u32.s32 %0, %1, 5;" : "=r"(tmp) : "r"(a));
    *out = tmp;
}

template<>
__global__ void k_set_aonly<int64_t>(uint32_t * out, const int64_t a) {
    uint32_t tmp;
    asm("set.eq.u32.s64 %0, %1, 5;" : "=r"(tmp) : "l"(a));
    *out = tmp;
}

template<>
__global__ void k_set_aonly<uint16_t>(uint32_t * out, const uint16_t a) {
    uint32_t tmp;
    asm("set.eq.u32.u16 %0, %1, 5;" : "=r"(tmp) : "h"(a));
    *out = tmp;
}

template<>
__global__ void k_set_aonly<uint32_t>(uint32_t * out, const uint32_t a) {
    uint32_t tmp;
    asm("set.eq.u32.u32 %0, %1, 5;" : "=r"(tmp) : "r"(a));
    *out = tmp;
}

template<>
__global__ void k_set_aonly<uint64_t>(uint32_t * out, const uint64_t a) {
    uint32_t tmp;
    asm("set.eq.u32.u64 %0, %1, 5;" : "=r"(tmp) : "l"(a));
    *out = tmp;
}

template<>
__global__ void k_set_aonly<float>(uint32_t * out, const float a) {
    uint32_t tmp;
    asm("set.eq.u32.f32 %0, %1, 0f40A00000;" : "=r"(tmp) : "f"(a));
    *out = tmp;
}

template<>
__global__ void k_set_aonly<double>(uint32_t * out, const double a) {
    uint32_t tmp;
    asm("set.eq.u32.f64 %0, %1, 0d4014000000000000;" : "=r"(tmp) : "d"(a));
    *out = tmp;
}

template<typename T>
__global__ void k_set_bonly(uint32_t * out, const T b) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_set_bonly<int16_t>(uint32_t * out, const int16_t b) {
    uint32_t tmp;
    asm("set.eq.u32.s16 %0, 3, %1;" : "=r"(tmp) : "h"(b));
    *out = tmp;
}

template<>
__global__ void k_set_bonly<int32_t>(uint32_t * out, const int32_t b) {
    uint32_t tmp;
    asm("set.eq.u32.s32 %0, 3, %1;" : "=r"(tmp) : "r"(b));
    *out = tmp;
}

template<>
__global__ void k_set_bonly<int64_t>(uint32_t * out, const int64_t b) {
    uint32_t tmp;
    asm("set.eq.u32.s64 %0, 3, %1;" : "=r"(tmp) : "l"(b));
    *out = tmp;
}

template<>
__global__ void k_set_bonly<uint16_t>(uint32_t * out, const uint16_t b) {
    uint32_t tmp;
    asm("set.eq.u32.u16 %0, 3, %1;" : "=r"(tmp) : "h"(b));
    *out = tmp;
}

template<>
__global__ void k_set_bonly<uint32_t>(uint32_t * out, const uint32_t b) {
    uint32_t tmp;
    asm("set.eq.u32.u32 %0, 3, %1;" : "=r"(tmp) : "r"(b));
    *out = tmp;
}

template<>
__global__ void k_set_bonly<uint64_t>(uint32_t * out, const uint64_t b) {
    uint32_t tmp;
    asm("set.eq.u32.u64 %0, 3, %1;" : "=r"(tmp) : "l"(b));
    *out = tmp;
}

template<>
__global__ void k_set_bonly<float>(uint32_t * out, const float b) {
    uint32_t tmp;
    asm("set.eq.u32.f32 %0, 0f40400000, %1;" : "=r"(tmp) : "f"(b));
    *out = tmp;
}

template<>
__global__ void k_set_bonly<double>(uint32_t * out, const double b) {
    uint32_t tmp;
    asm("set.eq.u32.f64 %0, %1, 0d4008000000000000;" : "=r"(tmp) : "d"(b));
    *out = tmp;
}

TYPED_TEST_P(SetTypeParameterized, ABOnly) {
    TypeParam a = 3;
    TypeParam b = 5;

    uint32_t hout[4];
    uint32_t *out;
    (void) cudaMalloc((void **) &out, sizeof(hout));

    cudaStream_t stream;
    (void) cudaStreamCreate(&stream);

    k_set_aonly<<<1, 1, 0, stream>>>(out + 0, a);

    VALGRIND_MAKE_MEM_UNDEFINED(&a, sizeof(a));
    k_set_aonly<<<1, 1, 0, stream>>>(out + 1, a);

    k_set_bonly<<<1, 1, 0, stream>>>(out + 2, b);

    VALGRIND_MAKE_MEM_UNDEFINED(&b, sizeof(b));
    k_set_bonly<<<1, 1, 0, stream>>>(out + 3, b);

    (void) cudaStreamSynchronize(stream);
    (void) cudaStreamDestroy(stream);
    (void) cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    (void) cudaFree(out);

    uint32_t vout[4];
    int valgrind = VALGRIND_GET_VBITS(hout, vout, sizeof(hout));
    assert(valgrind <= 1);
    if (valgrind == 1) {
        assert(vout[0] == 0);
        assert(vout[1] == 0xFFFFFFFF);
        assert(vout[2] == 0);
        assert(vout[3] == 0xFFFFFFFF);
    }

    assert(hout[0] == 0x0);
    assert(hout[2] == 0x0);
};

REGISTER_TYPED_TEST_CASE_P(SetTypeParameterized, Single, MixInC, ABOnly);

typedef ::testing::Types<int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, float, double> ParameterTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Inst, SetTypeParameterized, ParameterTypes);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
