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

TEST(HostGetFlags, NullArguments) {
    cudaError_t ret;
    unsigned int flags;

    ret = cudaHostGetFlags(NULL, NULL);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    /**
     * The flags = NULL, pHost != NULL case is provided for in
     * test_hostgetflags.cu as it causes a SIGSEGV.
     */

    ret = cudaHostGetFlags(&flags, NULL);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
}

TEST(HostGetFlags, WrongSources) {
    cudaError_t ret;

    unsigned int flags;
    void * normal_malloc    = malloc(4);
    ret = cudaHostGetFlags(&flags, normal_malloc);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
    free(normal_malloc);

    void * device_malloc;
    ret = cudaMalloc(&device_malloc, 4);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaHostGetFlags(&flags, device_malloc);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaFree(device_malloc);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(HostGetFlags, NextPage) {
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

    unsigned int flags;
    ret = cudaHostGetFlags(&flags, host_ptr + page_size);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(HostGetFlags, OffPage) {
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

    const size_t offset = (page_size + 1) / 2;
    assert(offset > 0);

    unsigned int flags;
    ret = cudaHostGetFlags(&flags, host_ptr + offset);
    EXPECT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
