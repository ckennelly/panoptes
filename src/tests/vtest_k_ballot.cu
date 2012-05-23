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

extern "C" __global__ void k_all_evens(const int * in, bool * out,
        const int N) {
    bool local = true;
    const int first_idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int idx = first_idx; idx < N; idx += blockDim.x * gridDim.x) {
        local = local && (in[idx] % 2 == 0);
    }

    out[first_idx / warpSize] = __all(local);
}

extern "C" __global__ void k_any_evens(const int * in, bool * out,
        const int N) {
    bool local = true;
    const int first_idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int idx = first_idx; idx < N; idx += blockDim.x * gridDim.x) {
        local = local && (in[idx] % 2 == 0);
    }

    out[first_idx / warpSize] = __any(local);
}

extern "C" __global__ void k_compact_evens(const int * in, int * out,
        const int N) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        /**
         * The writes of the exact same values will collide, but this will be
         * okay.
         */
        out[idx / warpSize] = __ballot(in[idx] % 2 == 0);
    }
}

TEST(kBallot, AllEvens) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;
    const int block_size = 256;

    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const int warps = n_blocks * block_size / prop.warpSize;

    int * in;
    bool * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * warps);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_all_evens<<<block_size, n_blocks, 0, stream>>>(in, out, N);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kBallot, AnyEvens) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;
    const int block_size = 256;

    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const int warps = n_blocks * block_size / prop.warpSize;

    int * in;
    bool * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * warps);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_any_evens<<<block_size, n_blocks, 0, stream>>>(in, out, N);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kBallot, CompactEvens) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    int * in;
    int * out;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * N /
        (sizeof(int) * CHAR_BIT));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_compact_evens<<<256, n_blocks, 0, stream>>>(in, out, N);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
