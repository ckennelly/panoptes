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

#include <boost/static_assert.hpp>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <unistd.h>
#include <valgrind/memcheck.h>

/**
 * The CUDA documentation for cudaMemcpyAsync notes that any host pointers
 * provided to the call must be page-locked.  Pageable addresses cause the
 * call to "return an error."
 *
 * That behavior is not evidenced by this set of tests, so Panoptes treats
 * the pointers just as it would any other.
 */

TEST(MemcpyAsync, CheckReturnValues) {
    /**
     * The API documentation states that
     * cudaErrorInvalidDevicePointer is a valid return value for
     * cudaMemcpyAsync
     *
     * TODO;  This needs a test.
     */
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * Test woefully out of range directions.
     */
    int a = 0;
    ret = cudaMemcpyAsync(&a,   &a,   sizeof(a), (cudaMemcpyKind) -1, stream);
    EXPECT_EQ(cudaErrorInvalidMemcpyDirection, ret);

    ret = cudaMemcpyAsync(NULL, NULL, sizeof(a), (cudaMemcpyKind) -1, stream);
    EXPECT_EQ(cudaErrorInvalidMemcpyDirection, ret);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    EXPECT_EQ(cudaSuccess, ret);
}

/**
 * CUDA4 introduced the cudaMemcpyDefault direction to cudaMemcpy.
 */
TEST(MemcpyAsync, CheckDefaultDirection) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    int a1 = 0;
    int a2 = 0;
    int * b;
    ret = cudaMalloc((void**) &b, sizeof(*b));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync(&a1,   &a2,  sizeof(a1), cudaMemcpyDefault, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync(&a1,    b,   sizeof(a1), cudaMemcpyDefault, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync( b,    &a1,  sizeof(a1), cudaMemcpyDefault, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync( b,    b,    sizeof(a1), cudaMemcpyDefault, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(b);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    EXPECT_EQ(cudaSuccess, ret);
}

/**
 * This test only performs copies in valid directions as to avoid upsetting
 * Valgrind.  The error-causing tests are in test_memcpy.cu.
 */
TEST(MemcpyAsync, AllDirections) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    int a1 = 0;
    int a2 = 0;
    int * b;
    ret = cudaMalloc((void**) &b, sizeof(*b) * 2);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync(&a1,    &a2,    sizeof(a1),
        cudaMemcpyHostToHost, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync(&a1,     b + 0, sizeof(a1),
        cudaMemcpyDeviceToHost, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync(&a1,     b + 1, sizeof(a1),
        cudaMemcpyDeviceToHost, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync( b + 0, &a1,    sizeof(a1),
        cudaMemcpyHostToDevice, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync( b + 1, &a1,    sizeof(a1),
        cudaMemcpyHostToDevice, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync( b + 0,  b + 0, sizeof(a1),
        cudaMemcpyDeviceToDevice, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync( b + 1,  b + 1, sizeof(a1),
        cudaMemcpyDeviceToDevice, stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(b);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(MemcpyAsync, Validity) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    int * device_ptr, src = 0, vsrc, dst, vdst;

    ret = cudaMalloc((void **) &device_ptr, sizeof(*device_ptr));
    ASSERT_EQ(cudaSuccess, ret);

    /* Only src is valid; *device_ptr and dst are invalid. */

    /* Do transfer */
    ret = cudaMemcpyAsync(device_ptr, &src, sizeof(src),
        cudaMemcpyHostToDevice, stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Both src and *device_ptr are valid; dst is invalid */
    ret = cudaMemcpyAsync(&dst, device_ptr, sizeof(dst),
        cudaMemcpyDeviceToHost, stream);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(src, dst);

    int valgrind = VALGRIND_GET_VBITS(&src, &vsrc, sizeof(src));
    assert(valgrind == 0 || valgrind == 1);

    if (valgrind == 1) {
        valgrind = VALGRIND_GET_VBITS(&dst, &vdst, sizeof(dst));
        assert(valgrind == 1);

        EXPECT_EQ(vsrc, vdst);
    }

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(device_ptr);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(MemcpyAsync, Pinned) {
    /**
     * Host memory must be pinned in order to be used as an argument to
     * cudaMemcpyAsync.  Panoptes only prints a warning about this error
     * rather than actually return an error via the CUDA API.  This test is
     * written as to check for the absence of an error once the CUDA
     * implementation starts returning one for nonpinned host memory.
     */
    const long page_size_ = sysconf(_SC_PAGESIZE);
    ASSERT_LT(0, page_size_);
    const size_t page_size = page_size_;

    const size_t pages = 3;
    assert(pages > 0);

    cudaError_t ret;
    cudaStream_t stream;

    uint8_t *device_ptr, *host_ptr;
    ret = cudaMalloc((void **) &device_ptr, pages * page_size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMallocHost((void **) &host_ptr, pages * page_size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Page aligned transfers */
    for (size_t i = 0; i < pages; i++) {
        for (size_t j = i; j < pages; j++) {
            ret = cudaMemcpyAsync(device_ptr, host_ptr + i * page_size,
                (pages - j) * page_size, cudaMemcpyHostToDevice, stream);
            EXPECT_EQ(cudaSuccess, ret);

            ret = cudaMemcpyAsync(host_ptr + i * page_size, device_ptr,
                (pages - j) * page_size, cudaMemcpyDeviceToHost, stream);
            EXPECT_EQ(cudaSuccess, ret);
        }
    }

    /* Try a nonaligned transfer. */
    ret = cudaMemcpyAsync(device_ptr, host_ptr + (page_size / 2),
        page_size / 2, cudaMemcpyHostToDevice, stream);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFreeHost(host_ptr);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(device_ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
