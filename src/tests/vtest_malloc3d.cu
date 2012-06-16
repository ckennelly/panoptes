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

#include <boost/scoped_array.hpp>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

//   ::testing::FLAGS_gtest_death_test_style = "threadsafe";

TEST(Malloc3DFree, NullArguments) {
    cudaError_t ret;

    ret = cudaMalloc3D(NULL, make_cudaExtent(0, 0, 0));
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMalloc3D(NULL, make_cudaExtent(0, 0, 8));
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMalloc3D(NULL, make_cudaExtent(0, 8, 0));
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMalloc3D(NULL, make_cudaExtent(0, 8, 8));
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMalloc3D(NULL, make_cudaExtent(8, 0, 0));
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMalloc3D(NULL, make_cudaExtent(8, 0, 8));
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaMalloc3D(NULL, make_cudaExtent(8, 8, 0));
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    /**
     * This segfaults...

    ret = cudaMalloc3D(NULL, make_cudaExtent(8, 8, 8));
    EXPECT_EQ(cudaErrorInvalidValue, ret);

     */
}

TEST(Malloc3DFree, Simple) {
    /* Allocate and free a small piece of memory to see nothing explodes */
    cudaPitchedPtr ptr;
    cudaError_t ret;
    ret = cudaMalloc3D(&ptr, make_cudaExtent(1, 1, sizeof(ptr)));
    ASSERT_EQ(cudaSuccess, ret);
    ASSERT_NE((void *) NULL, ptr.ptr);

    ret = cudaFree(ptr.ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Malloc3DFree, Validity) {
    const size_t depth  = 3;
    const size_t height = 3;
    const size_t width  = 3;

    /* Allocate a piece of memory, transfer it to the host, verify it is
     * uninitialized */
    cudaPitchedPtr ptr;
    cudaError_t ret;
    ret = cudaMalloc3D(&ptr, make_cudaExtent(width, height, depth));
    ASSERT_EQ(cudaSuccess, ret);
    ASSERT_LE(depth,         ptr.pitch);
    ASSERT_NE((void *) NULL, ptr.ptr);
    ASSERT_EQ(width,         ptr.xsize);
    ASSERT_EQ(height,        ptr.ysize);

    /* This sets the validity bits, so we need the memcpy to set them as well
     * or the test fails */
    const size_t alloc_size = ptr.pitch * ptr.xsize * ptr.ysize;
    boost::scoped_array<uint8_t> host_ptr(new uint8_t[alloc_size]);
    memset(host_ptr.get(), 0, alloc_size);

    ret = cudaMemcpy(host_ptr.get(), ptr.ptr, alloc_size,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    boost::scoped_array<uint8_t> vptr(new uint8_t[alloc_size]);

    int valgrind = VALGRIND_GET_VBITS(host_ptr.get(), vptr.get(), alloc_size);
    /* valgrind == 3 indicates we didn't do something right on the test
     * side. */
    assert(valgrind == 0 || valgrind == 1);

    if (valgrind == 1) {
        boost::scoped_array<uint8_t> expected_vptr(new uint8_t[alloc_size]);
        memset(expected_vptr.get(), 0xFF, alloc_size);

        int memcmp_ret = memcmp(vptr.get(), expected_vptr.get(), alloc_size);
        EXPECT_EQ(0, memcmp_ret);
    }

    ret = cudaFree(ptr.ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
