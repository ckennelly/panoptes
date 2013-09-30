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
#include <valgrind/memcheck.h>

extern "C" __global__ void k_cnot(const int * in, const int N,
        int * out) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        const int _in = in[idx];
        int _out;

        /* Compute _out = _in ? 1 : 0 as nvcc does not produce the cnot
           instruction. */
        asm("cnot.b32 %0, %1;" : "=r"(_out) : "r"(_in));

        out[idx] = _out;
    }
}

TEST(kCNOT, ExplicitStream) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    int * in;
    int * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_cnot<<<256, n_blocks, 0, stream>>>(in, N, out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

extern "C" __global__ void k_cnot_const(int32_t * out) {
    int32_t ret0, ret1;
    asm("cnot.b32 %0, 0;" : "=r"(ret0));
    asm("cnot.b32 %0, 1;" : "=r"(ret1));
    out[0] = ret0;
    out[1] = ret1;
}

TEST(kCNOT, Constant) {
    cudaError_t ret;
    cudaStream_t stream;

    int32_t * out;

    ret = cudaMalloc((void **) &out, 2 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_cnot_const<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    int32_t hout[2];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(0x1, hout[0]);
    EXPECT_EQ(0x0, hout[1]);

    int32_t vout[2];
    const int vret = VALGRIND_GET_VBITS(hout, vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0, vout[0]);
        EXPECT_EQ(0, vout[1]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
