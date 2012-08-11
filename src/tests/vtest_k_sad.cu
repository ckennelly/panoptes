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

__global__ void k_sad(const int * x, const int * y, const unsigned int * z,
        unsigned int * out, int32_t n) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        out[i] = __sad(x[i], y[i], z[i]);
    }
}

__global__ void k_sad_allconst(unsigned int * out) {
    unsigned int _out;
    asm("sad.s32 %0, 1, 2, 3;\n" : "=r"(_out));
    *out = _out;
}

TEST(kSAD, SADConstant) {
    cudaError_t ret;
    cudaStream_t stream;

    unsigned int * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_sad_allconst<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    unsigned int hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(4u, hout);
}

TEST(kSAD, SAD) {
    cudaError_t ret;
    cudaStream_t stream;

    const int32_t n = 1 << 24;
    int * x;
    int * y;
    unsigned int * z;
    unsigned int * out;

    ret = cudaMalloc((void **) &x, sizeof(*x) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &y, sizeof(*y) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &z, sizeof(*z) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_sad<<<256, 16, 0, stream>>>(x, y, z, out, n);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(x);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(y);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(z);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

__global__ void k_usad(const unsigned int * x, const unsigned int * y,
        const unsigned int * z, unsigned int * out, int32_t n) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        out[i] = __sad(x[i], y[i], z[i]);
    }
}

TEST(kSAD, USAD) {
    cudaError_t ret;
    cudaStream_t stream;

    const int32_t n = 1 << 24;
    unsigned int * x;
    unsigned int * y;
    unsigned int * z;
    unsigned int * out;

    ret = cudaMalloc((void **) &x, sizeof(*x) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &y, sizeof(*y) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &z, sizeof(*z) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_usad<<<256, 16, 0, stream>>>(x, y, z, out, n);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(x);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(y);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(z);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
