/**
 * Panoptes - A Binary Translation Framework for CUDA
 * (c) 2011-2013 Chris Kennelly <chris@ckennelly.com>
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

#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

/**
 * Per the PTX documentation, PTX ISA 3.0 was released with CUDA runtime
 * version 4.1.  As carry-flag related instructions appeared with this ISA, we
 * restrict compilation of the bulk of these tests to runtime versions 4.1 and
 * newer.
 */
#if CUDART_VERSION >= 4010 /* 4.1 */
static __global__ void k_mad_carry_in(uint32_t * out, uint32_t in) {
    uint32_t _out;
    asm volatile(
        "{ .reg .u32 %tmp;\n"
        "mad.lo.cc.u32 %tmp, 1, 2147483647, %1;\n"
        "madc.lo.u32 %0, 0, 0, 0;\n}" : "=r"(_out) : "r"(in));
    *out = _out;
}

TEST(Mad, ConstantArgumentsCarryIn) {
    cudaError_t ret;
    cudaStream_t stream;

    uint32_t * out;
    ret = cudaMalloc((void **) &out, 4 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const uint32_t in0 = 1;
    const uint32_t in1 = UINT_MAX;
          uint32_t in2 = in0;
          uint32_t in3 = in1;
    VALGRIND_MAKE_MEM_UNDEFINED(&in2, sizeof(in2));
    VALGRIND_MAKE_MEM_UNDEFINED(&in3, sizeof(in3));

    k_mad_carry_in<<<1, 1, 0, stream>>>(out + 0, in0);
    k_mad_carry_in<<<1, 1, 0, stream>>>(out + 1, in1);
    k_mad_carry_in<<<1, 1, 0, stream>>>(out + 2, in2);
    k_mad_carry_in<<<1, 1, 0, stream>>>(out + 3, in3);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hout[4];
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(0, hout[0]);
    EXPECT_EQ(1, hout[1]);

    uint32_t vout[4];
    const int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout[0]);
        EXPECT_EQ(0x00000000, vout[1]);
        EXPECT_EQ(0xFFFFFFFF, vout[2]);
        EXPECT_EQ(0xFFFFFFFF, vout[3]);
    }
}

static __global__ void k_mad_const_out(uint32_t * out) {
    uint32_t _out[2];
    asm volatile(
        "mad.lo.cc.u32 %0, 1, 4294967295, 1;\n"
        "madc.lo.u32 %1, 0, 0, 0;\n" : "=r"(_out[0]), "=r"(_out[1]));
    out[0] = _out[0];
    out[1] = _out[1];
}

TEST(Mad, ConstantArgumentsCarryOut) {
    cudaError_t ret;
    cudaStream_t stream;

    uint32_t * out;
    ret = cudaMalloc((void **) &out, 2 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_mad_const_out<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hout[2];
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(0, hout[0]);
    EXPECT_EQ(1, hout[1]);

    uint32_t vout[2];
    const int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout[0]);
        EXPECT_EQ(0x00000000, vout[1]);
    }
}
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
