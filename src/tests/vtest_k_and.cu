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
#include <valgrind/memcheck.h>

static __global__ void k_and_const11(uint32_t * out) {
    uint32_t _out;
    asm("and.b32 %0, 1234567890, 987654321;\n" : "=r"(_out));
    *out = _out;
}

static __global__ void k_and_constA1(uint32_t * out, uint32_t in) {
    uint32_t _out;
    asm("and.b32 %0, %1, 1234567890;\n" : "=r"(_out) : "r"(in));
    *out = _out;
}

static __global__ void k_and_const1B(uint32_t * out, uint32_t in) {
    uint32_t _out;
    asm("and.b32 %0, 1234567890, %1;\n" : "=r"(_out) : "r"(in));
    *out = _out;
}

TEST(And, BinaryConstant) {
    cudaError_t ret;
    cudaStream_t stream;

    uint32_t * out;
    ret = cudaMalloc((void **) &out, 5 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const uint32_t in = 987654321;
    uint32_t invalid_in = in;
    VALGRIND_MAKE_MEM_UNDEFINED(&invalid_in, sizeof(invalid_in));
    k_and_const11<<<1, 1, 0, stream>>>(out + 0);
    k_and_constA1<<<1, 1, 0, stream>>>(out + 1, in);
    k_and_constA1<<<1, 1, 0, stream>>>(out + 2, invalid_in);
    k_and_const1B<<<1, 1, 0, stream>>>(out + 3, in);
    k_and_const1B<<<1, 1, 0, stream>>>(out + 4, invalid_in);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hout[5];
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(1234567890 & 987654321, hout[0]);
    EXPECT_EQ(1234567890 & 987654321, hout[1]);
    EXPECT_EQ(1234567890 & 987654321, hout[1]);

    uint32_t vout[5];
    const int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout[0]);
        EXPECT_EQ(0x00000000, vout[1]);
        EXPECT_EQ(0xFFFFFFFF, vout[2]);
        EXPECT_EQ(0x00000000, vout[3]);
        EXPECT_EQ(0xFFFFFFFF, vout[4]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
