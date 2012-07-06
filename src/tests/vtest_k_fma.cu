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

static __global__ void k_fma_CCV(float * out, const float in) {
    float _out;
    asm volatile("fma.rn.f32 %0, 0f3f800000, 0f3f800000, %1;\n" :
        "=f"(_out) : "f"(in));
    *out = _out;
}

static __global__ void k_fma_CVC(float * out, const float in) {
    float _out;
    asm volatile("fma.rn.f32 %0, 0f3f800000, %1, 0f3f800000;\n" :
        "=f"(_out) : "f"(in));
    *out = _out;
}

static __global__ void k_fma_VCC(float * out, const float in) {
    float _out;
    asm volatile("fma.rn.f32 %0, %1, 0f3f800000, 0f3f800000;\n" :
        "=f"(_out) : "f"(in));
    *out = _out;
}

static __global__ void k_fma_CCC(float * out) {
    float _out;
    asm volatile("fma.rn.f32 %0, 0f3f800000, 0f3f800000, 0f3f800000;\n" :
        "=f"(_out));
    *out = _out;
}

TEST(FMA, Constants) {
    cudaError_t ret;
    cudaStream_t stream;

    float * out;
    ret = cudaMalloc((void **) &out, 7 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const float in = 2.f;

    /* This value cannot be constant, otherwise, NVCC will constant propagate
     * it into the kernel launches. */
    float invalid_in = in;
    VALGRIND_MAKE_MEM_UNDEFINED(&invalid_in, sizeof(invalid_in));
    k_fma_CCV<<<1, 1, 0, stream>>>(out + 0, in);
    k_fma_CVC<<<1, 1, 0, stream>>>(out + 1, in);
    k_fma_VCC<<<1, 1, 0, stream>>>(out + 2, in);
    k_fma_CCC<<<1, 1, 0, stream>>>(out + 3);
    k_fma_CCV<<<1, 1, 0, stream>>>(out + 4, invalid_in);
    k_fma_CVC<<<1, 1, 0, stream>>>(out + 5, invalid_in);
    k_fma_VCC<<<1, 1, 0, stream>>>(out + 6, invalid_in);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    float hout[7];
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(1.f + in, hout[0]);
    EXPECT_EQ(1.f + in, hout[1]);
    EXPECT_EQ(1.f + in, hout[2]);
    EXPECT_EQ(2.f,      hout[3]);

    uint32_t vout[7];
    const int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout[0]);
        EXPECT_EQ(0x00000000, vout[1]);
        EXPECT_EQ(0x00000000, vout[2]);
        EXPECT_EQ(0x00000000, vout[3]);
        EXPECT_EQ(0xFFFFFFFF, vout[4]);
        EXPECT_EQ(0xFFFFFFFF, vout[5]);
        EXPECT_EQ(0xFFFFFFFF, vout[6]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
