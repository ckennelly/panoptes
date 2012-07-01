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

TEST(HostAlloc, MallocFree) {
    cudaError_t ret;
    int * ptr;

    ret = cudaHostAlloc((void **) &ptr, sizeof(*ptr), 0);
    ASSERT_EQ(cudaSuccess, ret);

    ASSERT_FALSE(NULL == ptr);
    *ptr = 0;

    ret = cudaFreeHost(ptr);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(HostAlloc, FlagRetrieval) {
    cudaError_t ret;
    void * ptrs[8];
    unsigned int flags[8];

    int device;

    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    for (size_t i = 0; i < (sizeof(flags) / sizeof(flags[0])); i++) {
        unsigned int flag = cudaHostAllocDefault;

        if (i & 0x1) {
            flag |= cudaHostAllocPortable;
        }

        if (i & 0x2) {
            flag |= cudaHostAllocMapped;
        }

        if (i & 0x4) {
            flag |= cudaHostAllocWriteCombined;
        }

        ret = cudaHostAlloc(&ptrs[i], 4, flag);
        ASSERT_EQ(cudaSuccess, ret);

        flags[i] = flag;
    }

    for (size_t i = 0; i < (sizeof(flags) / sizeof(flags[0])); i++) {
        unsigned int flag;
        ret = cudaHostGetFlags(&flag, ptrs[i]);
        ASSERT_EQ(cudaSuccess, ret);

        const unsigned int expected = flags[i] |
            (prop.canMapHostMemory ? cudaHostAllocMapped : 0);
        EXPECT_EQ(expected, flag);
    }

    for (size_t i = 0; i < (sizeof(flags) / sizeof(flags[0])); i++) {
        ret = cudaFreeHost(ptrs[i]);
        EXPECT_EQ(cudaSuccess, ret);
    }
}

TEST(HostAlloc, NullArguments) {
    cudaError_t ret;

    ret = cudaHostAlloc(NULL, 0, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaHostAlloc(NULL, 4, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaFreeHost(NULL);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(HostAlloc, MappedPointer) {
    cudaError_t ret;
    int device;

    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    void * ptr;
    ret = cudaHostAlloc(&ptr, 4, cudaHostAllocMapped);
    ASSERT_EQ(cudaSuccess, ret);

    /*
     * Try to retrieve the device pointer, expecting a result according to
     * prop.canMapHostMemory.
     */
    void * device_ptr;
    ret = cudaHostGetDevicePointer(&device_ptr, ptr, 0);
    if (prop.canMapHostMemory) {
        EXPECT_EQ(cudaSuccess, ret);
        EXPECT_FALSE(device_ptr == NULL);
    } else {
        EXPECT_EQ(cudaErrorMemoryAllocation, ret);
    }

    ret = cudaFreeHost(ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

/** TODO:  Mismatched */
/** TODO:  leak detection */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
