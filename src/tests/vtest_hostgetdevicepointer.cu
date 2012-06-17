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
#include <unistd.h>

TEST(HostGetDevicePointer, NullArguments) {
    cudaError_t ret;
    void * pDevice;

    ret = cudaHostGetDevicePointer(NULL, NULL, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    /**
     * The pDevice = NULL, pHost != NULL case is provided for in
     * test_hostgetdevicepointer.cu as it causes a SIGSEGV.
     */

    ret = cudaHostGetDevicePointer(&pDevice, NULL, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
}

TEST(HostGetDevicePointer, NonZeroFlags) {
    cudaError_t ret;
    void * pDevice;
    void * pHost;

    ret = cudaHostAlloc(&pHost, 4, 0);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostGetDevicePointer(&pDevice, pHost, 1);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaFreeHost(pHost);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(HostGetDevicePointer, WrongSources) {
    cudaError_t ret;

    void * device_ptr;
    void * normal_malloc    = malloc(4);
    ret = cudaHostGetDevicePointer(&device_ptr, normal_malloc, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
    free(normal_malloc);

    void * device_malloc;
    ret = cudaMalloc(&device_malloc, 4);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostGetDevicePointer(&device_ptr, device_malloc, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaFree(device_malloc);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(HostGetDevicePointer, WrongSourcesNullArguments) {
    cudaError_t ret;

    void * normal_malloc    = malloc(4);

    ret = cudaHostGetDevicePointer(NULL, normal_malloc, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    free(normal_malloc);

    void * device_malloc;
    ret = cudaMalloc(&device_malloc, 4);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostGetDevicePointer(NULL, device_malloc, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaFree(device_malloc);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(HostGetDevicePointer, NextPage) {
    /**
     * We allocate a host buffer sized at two pages.  We then try to retrieve
     * the device pointer of the second page.
     */
    const long page_size_ = sysconf(_SC_PAGESIZE);
    if (page_size_ <= 0) {
        /* This is hopeless... */
        return;
    }

    const size_t page_size = page_size_;

    cudaError_t ret;
    uint8_t * host_ptr;
    ret = cudaHostAlloc((void **) &host_ptr, 2 * page_size, 0);
    ASSERT_EQ(cudaSuccess, ret);

    int device;

    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    void * device_ptr;
    ret = cudaHostGetDevicePointer(&device_ptr, host_ptr + page_size, 0);
    if (prop.canMapHostMemory) {
        EXPECT_EQ(cudaSuccess, ret);
    } else {
        EXPECT_EQ(cudaErrorMemoryAllocation, ret);
    }

    ret = cudaFreeHost(host_ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(HostGetDevicePointer, OffPage) {
    /**
     * We allocate a host buffer sized at one page.  We then try to retrieve
     * the device pointer part way through the allocation.
     */
    const long page_size_ = sysconf(_SC_PAGESIZE);
    if (page_size_ <= 0) {
        /* This is hopeless... */
        return;
    }

    const size_t page_size = page_size_;

    cudaError_t ret;
    uint8_t * host_ptr;
    ret = cudaHostAlloc((void **) &host_ptr, 2 * page_size, 0);
    ASSERT_EQ(cudaSuccess, ret);

    int device;

    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const size_t offset = (page_size + 1) / 2;
    assert(offset > 0);

    void * device_ptr;
    ret = cudaHostGetDevicePointer(&device_ptr, host_ptr + offset, 0);
    if (prop.canMapHostMemory) {
        EXPECT_EQ(cudaSuccess, ret);
    } else {
        EXPECT_EQ(cudaErrorMemoryAllocation, ret);
    }

    ret = cudaFreeHost(host_ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
