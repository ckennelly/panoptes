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
#include <stdint.h>
#include <valgrind/memcheck.h>

/**
 * Read the %envreg<32> registers.
 */
static __global__ void k_read_envreg(uint32_t * out) {
    uint32_t _out[32];
    asm("mov.u32 %0, %envreg0;\n" : "=r"(_out[0]));
    asm("mov.u32 %0, %envreg1;\n" : "=r"(_out[1]));
    asm("mov.u32 %0, %envreg2;\n" : "=r"(_out[2]));
    asm("mov.u32 %0, %envreg3;\n" : "=r"(_out[3]));
    asm("mov.u32 %0, %envreg4;\n" : "=r"(_out[4]));
    asm("mov.u32 %0, %envreg5;\n" : "=r"(_out[5]));
    asm("mov.u32 %0, %envreg6;\n" : "=r"(_out[6]));
    asm("mov.u32 %0, %envreg7;\n" : "=r"(_out[7]));
    asm("mov.u32 %0, %envreg8;\n" : "=r"(_out[8]));
    asm("mov.u32 %0, %envreg9;\n" : "=r"(_out[9]));
    asm("mov.u32 %0, %envreg10;\n" : "=r"(_out[10]));
    asm("mov.u32 %0, %envreg11;\n" : "=r"(_out[11]));
    asm("mov.u32 %0, %envreg12;\n" : "=r"(_out[12]));
    asm("mov.u32 %0, %envreg13;\n" : "=r"(_out[13]));
    asm("mov.u32 %0, %envreg14;\n" : "=r"(_out[14]));
    asm("mov.u32 %0, %envreg15;\n" : "=r"(_out[15]));
    asm("mov.u32 %0, %envreg16;\n" : "=r"(_out[16]));
    asm("mov.u32 %0, %envreg17;\n" : "=r"(_out[17]));
    asm("mov.u32 %0, %envreg18;\n" : "=r"(_out[18]));
    asm("mov.u32 %0, %envreg19;\n" : "=r"(_out[19]));
    asm("mov.u32 %0, %envreg20;\n" : "=r"(_out[20]));
    asm("mov.u32 %0, %envreg21;\n" : "=r"(_out[21]));
    asm("mov.u32 %0, %envreg22;\n" : "=r"(_out[22]));
    asm("mov.u32 %0, %envreg23;\n" : "=r"(_out[23]));
    asm("mov.u32 %0, %envreg24;\n" : "=r"(_out[24]));
    asm("mov.u32 %0, %envreg25;\n" : "=r"(_out[25]));
    asm("mov.u32 %0, %envreg26;\n" : "=r"(_out[26]));
    asm("mov.u32 %0, %envreg27;\n" : "=r"(_out[27]));
    asm("mov.u32 %0, %envreg28;\n" : "=r"(_out[28]));
    asm("mov.u32 %0, %envreg29;\n" : "=r"(_out[29]));
    asm("mov.u32 %0, %envreg30;\n" : "=r"(_out[30]));
    asm("mov.u32 %0, %envreg31;\n" : "=r"(_out[31]));

    #pragma unroll
    for (int i = 0; i < 32; i += 4) {
        *(uint4 *)(out + i) = *(const uint4 *)(_out + i);
    }
}

TEST(EnvRegs, Validity) {
    if (!(RUNNING_ON_VALGRIND)) {
        return;
    }

    cudaError_t ret;
    cudaStream_t stream;

    uint32_t * out;
    ret = cudaMalloc((void **) &out, 32 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_read_envreg<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hout[32];
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t vout[32];
    const int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        for (int i = 0; i < 32; i++) {
            EXPECT_EQ(0x00000000, vout[i]);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
