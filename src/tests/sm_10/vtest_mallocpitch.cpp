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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

TEST(MallocPitch, NullArguments) {
    cudaError_t ret;

    void *devPtr;
    size_t pitch;

    ret = cudaMallocPitch(&devPtr, NULL,   0, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMallocPitch(&devPtr, NULL,   1, 1);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMallocPitch(NULL,    &pitch, 0, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMallocPitch(NULL,    &pitch, 1, 1);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMallocPitch(NULL,    NULL,   0, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMallocPitch(NULL,    NULL,   1, 1);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
}

TEST(MallocPitch, Simple) {
    /**
     * Allocate and free a small piece of pitched memory to verify nothing
     * explodes.
     */
    cudaError_t ret;
    void *devPtr;
    size_t pitch;
    const size_t width  = 3u;
    const size_t height = 3u;

    ret = cudaMallocPitch(&devPtr, &pitch, width, height);
    ASSERT_EQ(cudaSuccess, ret);
    ASSERT_NE((void *) NULL, devPtr);
    ASSERT_LE(width, pitch);

    ret = cudaFree(devPtr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(MallocPitch, Copy) {
    /**
     * cudaMallocPitch should return a pointer to a linear memory location of
     * size pitch * height.  Allocate a pitched pointer, then allocate another
     * memory location of the same expected size and ping pong the contents
     * of the two.
     */
    cudaError_t ret;

    void *pitchedPtr;
    size_t pitch;
    const size_t width  = 27;
    const size_t height = 1024;

    ret = cudaMallocPitch(&pitchedPtr, &pitch, width, height);
    ASSERT_EQ(cudaSuccess, ret);

    void *linearPtr;
    const size_t linear = pitch * height;
    ret = cudaMalloc(&linearPtr, linear);
    ASSERT_EQ(cudaSuccess, ret);

    /* Copy in each direction. */
    ret = cudaMemcpy(pitchedPtr, linearPtr, linear, cudaMemcpyDeviceToDevice);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(linearPtr, pitchedPtr, linear, cudaMemcpyDeviceToDevice);
    ASSERT_EQ(cudaSuccess, ret);

    /* Cleanup */
    ret = cudaFree(pitchedPtr);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(linearPtr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(MallocPitch, LargePitch) {
    /**
     * Request a width slightly larger than the memPitch specified for the
     * device by cudaGetDeviceProperties.
     */
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const size_t width  = prop.memPitch + 1u;
    const size_t height = 5;

    void *devPtr;
    size_t pitch;

    ret = cudaMallocPitch(&devPtr, &pitch, width, height);
    ASSERT_EQ(cudaErrorMemoryAllocation, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
