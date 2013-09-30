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

TEST(PointerGetAttributes, NonNull) {
    cudaError_t ret;
    struct cudaPointerAttributes attr;
    void * ptr;

    ret = cudaPointerGetAttributes(NULL,  NULL);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaPointerGetAttributes(&attr, NULL);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaPointerGetAttributes(NULL,  &ptr);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaPointerGetAttributes(&attr, &ptr);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
}

TEST(PointerGetAttributes, Malloc) {
    cudaError_t ret;
    void * ptr;

    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc(&ptr, sizeof(1));
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaPointerAttributes attr;
    ret = cudaPointerGetAttributes(&attr, ptr);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(device, attr.device);
    EXPECT_EQ(ptr, attr.devicePointer);
    EXPECT_EQ(NULL, attr.hostPointer);
    EXPECT_EQ(cudaMemoryTypeDevice, attr.memoryType);

    ret = cudaFree(ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(PointerGetAttributes, Array) {
    struct cudaArray * ary;
    cudaError_t ret;

    struct cudaChannelFormatDesc dsc;
    dsc.x = dsc.y = dsc.z = dsc.w = 8;
    dsc.f = cudaChannelFormatKindSigned;

    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMallocArray(&ary, &dsc, 1, 1, 0);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaPointerAttributes attr;
    ret = cudaPointerGetAttributes(&attr, ary);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaFreeArray(ary);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
