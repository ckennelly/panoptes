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

/**
 * For reasons that are unclear, this test hangs when run under Valgrind (at
 * least when using a Linux workstation that is actively running X).  The
 * assertion fails and produces output, but the launches that ought to fail
 * with cudaErrorAssert fail with cudaErrorLaunchTimeout.
 *
 * TODO:  Figure out what's wrong and fix it (if it's Panoptes' fault).
 */
extern "C" __global__ void k_assert(int N) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    assert(i < N);
}

TEST(kAssert, ExplicitStream) {
    cudaError_t ret;

    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    if (prop.major < 2) {
        /* Assertions are not available. */
        return;
    }

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_assert<<<1, 1, 0, stream>>>(2);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    k_assert<<<256, 16, 0, stream>>>(2);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaErrorAssert, ret);

    ret = cudaStreamDestroy(stream);
    EXPECT_EQ(cudaErrorAssert, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
