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
#include <limits>
#include <stdint.h>
#include <valgrind/memcheck.h>

/**
 * Loads a value from a hard-coded, constant address.
 */
static __global__ void k_ld_const(int2 * out, int in) {
    int2 _out;
    asm volatile(
        "{ .local .u32 l[1];\n"
        "st.local.u32 [0], %2;\n"
        "ld.local.u32 %0, [0];\n"
        "ld.local.u32 %1, [4];\n}" : "=r"(_out.x), "=r"(_out.y) : "r"(in));
    *out = _out;
}

TEST(Load, Constant) {
    cudaError_t ret;
    cudaStream_t stream;

    int2 * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const int expected = 5;
    k_ld_const<<<1, 1, 0, stream>>>(out, expected);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    int2 hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(expected, hout.x);

    int2 vout;
    const int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout.x);
        EXPECT_EQ(0xFFFFFFFF, vout.y);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
