/**
 * Panoptes - A Binary Translation Framework for CUDA
 * (c) 2011-2013 Chris Kennelly <chris@ckennelly.com>
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
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

TEST(HostRegister, NullArguments) {
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);

    const size_t page_size = page_size_;

    cudaError_t ret;

    ret = cudaHostRegister(NULL, 0, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaHostRegister(NULL, page_size, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
}

TEST(HostRegister, Simple) {
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);

    const size_t page_size = page_size_;

    void * ptr;
    int memret = posix_memalign(&ptr, page_size, page_size);
    ASSERT_EQ(0, memret);

    cudaError_t ret;
    ret = cudaHostRegister(ptr, page_size, cudaHostRegisterMapped);
    ASSERT_EQ(cudaSuccess, ret);

    /* Look into whether we can get a device pointer. */
    int device;

    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    void * device_ptr;
    ret = cudaHostGetDevicePointer(&device_ptr, ptr, 0);
    if (prop.canMapHostMemory) {
        EXPECT_EQ(cudaSuccess, ret);
    } else {
        EXPECT_EQ(cudaErrorMemoryAllocation, ret);
    }

    ret = cudaHostUnregister(ptr);
    EXPECT_EQ(cudaSuccess, ret);

    free(ptr);
}

TEST(HostRegister, Misaligned) {
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);

    const size_t page_size = page_size_;

    void * ptr;
    int memret = posix_memalign(&ptr, page_size, page_size + sizeof(int));
    ASSERT_EQ(0, memret);

    /* We misalign by sizeof(int). */

    cudaError_t ret;
    int runtime_version;
    ret = cudaRuntimeGetVersion(&runtime_version);
    ASSERT_EQ(cudaSuccess, ret);

    void * const target_ptr = static_cast<int *>(ptr) + 1;

    ret = cudaHostRegister(target_ptr, page_size, 0);
    if (runtime_version >= 4010 /* 4.1 */) {
        /**
         * CUDA 4.1 can handle nonaligned registration requests.
         */
        EXPECT_EQ(cudaSuccess, ret);

        /* Since we succeeded, we must unregister the pointer. */
        ret = cudaHostUnregister(target_ptr);
        ASSERT_EQ(cudaSuccess, ret);
    } else {
        EXPECT_EQ(cudaErrorInvalidValue, ret);
        /* No need to unregister anything */
    }

    free(ptr);
}

TEST(HostRegister, NonpageMultiple) {
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);

    const size_t page_size = page_size_;

    void * ptr;
    int memret = posix_memalign(&ptr, page_size, page_size);
    ASSERT_EQ(0, memret);

    cudaError_t ret;
    int runtime_version;
    ret = cudaRuntimeGetVersion(&runtime_version);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostRegister(ptr, page_size / 2, 0);
    if (runtime_version >= 4010 /* 4.1 */) {
        /**
         * CUDA 4.1 can handle nonpagesized registration requests.
         */
        EXPECT_EQ(cudaSuccess, ret);

        /* Since we succeeded, we must unregister the pointer. */
        ret = cudaHostUnregister(ptr);
        ASSERT_EQ(cudaSuccess, ret);
    } else {
        EXPECT_EQ(cudaErrorInvalidValue, ret);
        /* No need to unregister anything */
    }

    free(ptr);
}

TEST(HostRegister, DeviceBuffer) {
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);

    const size_t page_size = page_size_;

    cudaError_t ret;
    void * device_ptr;
    ret = cudaMalloc(&device_ptr, page_size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostRegister(device_ptr, page_size, 0);
    EXPECT_EQ(cudaErrorUnknown, ret);

    ret = cudaFree(device_ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(HostRegister, DoubleRegister) {
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);

    const size_t page_size = page_size_;

    void * ptr;
    int memret = posix_memalign(&ptr, page_size, page_size);
    ASSERT_EQ(0, memret);

    cudaError_t ret;
    ret = cudaHostRegister(ptr, page_size, cudaHostRegisterMapped);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostRegister(ptr, page_size, cudaHostRegisterMapped);
    #if CUDART_VERSION >= 4010 /* 4.1 */
    EXPECT_EQ(cudaErrorHostMemoryAlreadyRegistered, ret);
    #else
    EXPECT_EQ(cudaErrorUnknown, ret);
    #endif

    ret = cudaHostUnregister(ptr);
    EXPECT_EQ(cudaSuccess, ret);

    free(ptr);
}

TEST(HostRegister, OverlappingRegistrations) {
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);

    const size_t page_size = page_size_;

    void * ptr;
    int memret = posix_memalign(&ptr, page_size, 3u * page_size);
    ASSERT_EQ(0, memret);

    /* Overlap a registration for pages 0/1 and pages 1/2. */
    void * const lower_ptr = ptr;
    void * const upper_ptr = static_cast<uint8_t *>(ptr) + page_size;

    cudaError_t ret;
    ret = cudaHostRegister(lower_ptr, 2u * page_size, cudaHostRegisterMapped);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostRegister(upper_ptr, 2u * page_size, cudaHostRegisterMapped);
    #if CUDART_VERSION >= 4010 /* 4.1 */
    EXPECT_EQ(cudaErrorHostMemoryAlreadyRegistered, ret);
    #else
    EXPECT_EQ(cudaErrorUnknown, ret);
    #endif

    ret = cudaHostUnregister(ptr);
    EXPECT_EQ(cudaSuccess, ret);

    free(ptr);
}

TEST(HostRegister, RegisterMallocHost) {
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);

    const size_t page_size = page_size_;

    cudaError_t ret;
    void * ptr;
    ret = cudaMallocHost(&ptr, page_size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostRegister(ptr, page_size, cudaHostRegisterMapped);
    EXPECT_EQ(cudaErrorUnknown, ret);

    ret = cudaFreeHost(ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(HostRegister, RegisterHostAlloc) {
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);

    const size_t page_size = page_size_;

    cudaError_t ret;
    void * ptr;
    ret = cudaHostAlloc(&ptr, page_size, cudaHostAllocDefault);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostRegister(ptr, page_size, cudaHostRegisterMapped);
    EXPECT_EQ(cudaErrorUnknown, ret);

    ret = cudaFreeHost(ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
