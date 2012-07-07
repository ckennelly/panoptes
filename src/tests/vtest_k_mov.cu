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
static __device__ __inline__ S zip(const T a, const T b) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__device__ __inline__ uint32_t zip(const uint16_t a,
        const uint16_t b) {
    uint32_t ret;
    asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(ret) : "h"(a), "h"(b));
    return ret;
}

template<>
__device__ __inline__ uint64_t zip(const uint32_t a,
        const uint32_t b) {
    uint64_t ret;
    asm volatile("mov.b64 %0, {%1, %2};\n" : "=l"(ret) : "r"(a), "r"(b));
    return ret;
}

template<typename S, typename T>
__global__ void k_zip(S * d, const T a, const T b) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_zip(uint32_t * d, const uint16_t a, const uint16_t b) {
    *d = zip<uint32_t, uint16_t>(a, b);
}

template<>
__global__ void k_zip(uint64_t * d, const uint32_t a, const uint32_t b) {
    *d = zip<uint64_t, uint32_t>(a, b);
}

TEST(MovTest, Zip32) {
    typedef uint16_t src_t;
    typedef uint32_t dst_t;

    cudaError_t ret;

    dst_t * d;
    ret = cudaMalloc((void **) &d, sizeof(*d));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const src_t a   = 5;
    const src_t b   = 3;
    const dst_t exp = (((dst_t) b) << (sizeof(src_t) * CHAR_BIT)) | ((dst_t) a);

    k_zip<<<1, 1, 0, stream>>>(d, a, b);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    dst_t hd;
    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(exp, hd);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(MovTest, Zip64) {
    typedef uint32_t src_t;
    typedef uint64_t dst_t;

    cudaError_t ret;

    dst_t * d;
    ret = cudaMalloc((void **) &d, sizeof(*d));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const src_t a   = 5;
    const src_t b   = 3;
    const dst_t exp = (((dst_t) b) << (sizeof(src_t) * CHAR_BIT)) | ((dst_t) a);

    k_zip<<<1, 1, 0, stream>>>(d, a, b);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    dst_t hd;
    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(exp, hd);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

/**
 * TODO: Test unzip.
 */

/**
 * Validate that Panoptes supports moving predicate values.
 */
__global__ void k_mov_pred(uint32_t * out, uint32_t in) {
    uint32_t out_;
    asm volatile("{\n"
        " .reg .pred %tmp;\n"
        " mov.pred %tmp, 0;\n"
        " selp.u32 %0, 0, %1, %tmp;\n"
        "}\n" : "=r"(out_) : "r"(in));
    *out = out_;
}

TEST(Regression, MovePredicate) {
    cudaError_t ret;

    uint32_t * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const uint32_t expected = 0xDEADBEEF;

    k_mov_pred<<<1, 1, 0, stream>>>(out, expected);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(expected, hout);
}

/**
 * We move one known-constant identifier and one variable identifier in the
 * same operation to validate that Panoptes generates valid instrumentation.
 */
__global__ void k_mov_dual(uint2 * out, const unsigned int in) {
    uint2 _out;
    asm("{ .reg .b64 %tmp;\n"
        "mov.b64 %tmp, {%clock, %2};\n"
        "mov.b64 {%0, %1}, %tmp;}\n" : "=r"(_out.x), "=r"(_out.y) : "r"(in));
    *out = _out;
}

TEST(MovTest, Dual) {
    cudaError_t ret;

    uint2 * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    unsigned int in = 0xDEADBEEF;
    k_mov_dual<<<1, 1, 0, stream>>>(out, in);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint2 hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(in, hout.y);
    uint2 vout;
    int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout.x);
        EXPECT_EQ(0x00000000, vout.y);

        VALGRIND_MAKE_MEM_UNDEFINED(&in, sizeof(in));
        k_mov_dual<<<1, 1, 0, stream>>>(out, in);

        ret = cudaStreamSynchronize(stream);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
        ASSERT_EQ(cudaSuccess, ret);

        vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
        ASSERT_EQ(1, vret);

        EXPECT_EQ(0x00000000, vout.x);
        EXPECT_EQ(0xFFFFFFFF, vout.y);
    }

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

/**
 * Move two known values together
 */
__global__ void k_mov_dual_const(uint2 * out) {
    uint2 _out;
    asm("{ .reg .b64 %tmp;\n"
        "mov.b64 %tmp, {%clock, %clock};\n"
        "mov.b64 {%0, %1}, %tmp;}\n" : "=r"(_out.x), "=r"(_out.y));
    *out = _out;
}

TEST(MovTest, DualConst) {
    cudaError_t ret;

    uint2 * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_mov_dual_const<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint2 hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    uint2 vout;
    int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout.x);
        EXPECT_EQ(0x00000000, vout.y);
    }

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
