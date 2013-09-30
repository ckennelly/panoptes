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

#include <gtest/gtest.h>
#include <valgrind/memcheck.h>

/**
 * Read the pm registers.
 */
static __global__ void k_read_pm(uint4 * out) {
    uint4 _out;
    asm volatile(
        "mov.u32 %0, %pm0;\n"
        "mov.u32 %1, %pm1;\n"
        "mov.u32 %2, %pm2;\n"
        "mov.u32 %3, %pm3;\n" : "=r"(_out.x), "=r"(_out.y),
                                "=r"(_out.z), "=r"(_out.w));
    *out = _out;
}

TEST(PM, Validity) {
    if (!(RUNNING_ON_VALGRIND)) {
        return;
    }

    cudaError_t ret;
    cudaStream_t stream;

    uint4 * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_read_pm<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint4 hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    uint4 vout;
    const int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout.x);
        EXPECT_EQ(0x00000000, vout.y);
        EXPECT_EQ(0x00000000, vout.z);
        EXPECT_EQ(0x00000000, vout.w);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
