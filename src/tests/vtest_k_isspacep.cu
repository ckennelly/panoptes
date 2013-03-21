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

#include "common.h"
#include <cuda.h>
#include <gtest/gtest.h>

__global__ void k_is_global(bool * out, const void * ptr) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "isspacep.global %tmp, %1;\n"
        "selp.s32 %0, 1, 0, %tmp;}\n" :
        "=r"(ret) : PTRC(ptr));
    *out = ret;
}

__global__ void k_is_local(bool * out, const void * ptr) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "isspacep.local %tmp, %1;\n"
        "selp.s32 %0, 1, 0, %tmp;}\n" :
        "=r"(ret) : PTRC(ptr));
    *out = ret;
}

__global__ void k_is_shared(bool * out, const void * ptr) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "isspacep.shared %tmp, %1;\n"
        "selp.s32 %0, 1, 0, %tmp;}\n" :
        "=r"(ret) : PTRC(ptr));
    *out = ret;
}

__global__ void k_is_global_const(bool * out) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "isspacep.global %tmp, 0;\n"
        "selp.s32 %0, 1, 0, %tmp;}\n" : "=r"(ret));
    *out = ret;
}

TEST(IsSpacePTest, Single) {
    cudaError_t ret;
    cudaStream_t stream;

    bool * d;
    ret = cudaMalloc((void **) &d, 4 * sizeof(*d));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_is_global      <<<1, 1, 0, stream>>>(d + 0, d);
    k_is_global_const<<<1, 1, 0, stream>>>(d + 1);
    k_is_local       <<<1, 1, 0, stream>>>(d + 2, d);
    k_is_shared      <<<1, 1, 0, stream>>>(d + 3, d);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hd[4];
    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    if (version != 5000 /* 5.0 */) {
        /**
         * As of CUDA 5.0, this particular test fails while running NATIVELY.
         * While we could compare against the version and modify our
         * expectation accordingly (for the sake of ensuring that programs
         * cannot tell whether they are being run under Panoptes), the code
         * generated Panoptes (and then compiled via the driver API) *works*.
         *
         * To complicate things further, a very similar test in GPU Ocelot
         * fails with CUDA 5.0 and a GTX480:
         *
         * https://code.google.com/p/gpuocelot/source/browse/trunk/ocelot/ocelot/cuda/test/driver/generic.cpp
         */
        EXPECT_FALSE(hd[0]);
    }
    EXPECT_TRUE(hd[1]);
    EXPECT_FALSE(hd[2]);
    EXPECT_FALSE(hd[3]);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

/**
 * TODO:  Add a validity check to see that we propagate invalid bits.
 */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
