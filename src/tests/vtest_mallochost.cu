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

TEST(MallocHost, MallocFree) {
    cudaError_t ret;
    int * ptr;

    ret = cudaMallocHost((void **) &ptr, sizeof(*ptr));
    ASSERT_EQ(cudaSuccess, ret);

    ASSERT_FALSE(NULL == ptr);
    *ptr = 0;

    ret = cudaFreeHost(ptr);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(MallocHost, NullArguments) {
    cudaError_t ret;

    ret = cudaMallocHost(NULL, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMallocHost(NULL, 4);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaFreeHost(NULL);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(MallocHost, FlagRetrieval) {
    cudaError_t ret;
    int * ptr;

    ret = cudaMallocHost((void **) &ptr, sizeof(*ptr));
    ASSERT_EQ(cudaSuccess, ret);

    ASSERT_FALSE(NULL == ptr);

    unsigned int flags;
    ret = cudaHostGetFlags(&flags, ptr);
    EXPECT_EQ(cudaSuccess, ret);

    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    if (prop.unifiedAddressing) {
        EXPECT_EQ(cudaHostAllocMapped, flags);
    } else {
        EXPECT_EQ(cudaHostAllocDefault, flags);
    }

    ret = cudaFreeHost(ptr);
    EXPECT_EQ(cudaSuccess, ret);

}

/** TODO:  Mismatched */
/** TODO:  leak detection */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
