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

static __global__ void k_copy(int * out, int in) {
    *out = in;
}

TEST(DeviceReset, Simple) {
    cudaError_t ret;
    cudaStream_t stream;

    int * out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    int in = 1;
    k_copy<<<1, 1, 0, stream>>>(out, in);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    int hout = 0;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);
    ASSERT_EQ(in, hout);

    /* Pervious value of out is now invalidated. */
    ret = cudaDeviceReset();
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    in = 2;
    k_copy<<<1, 1, 0, stream>>>(out, in);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);
    ASSERT_EQ(in, hout);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(DeviceReset, Alias) {
    /*
     * cudaThreadExit aliases cudaDeviceReset.  Calling it should not be
     * harmful.
     */
    (void) cudaGetLastError();

    const cudaError_t ret = cudaThreadExit();
    EXPECT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
