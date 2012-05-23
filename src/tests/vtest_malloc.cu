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
#include <valgrind/memcheck.h>

//   ::testing::FLAGS_gtest_death_test_style = "threadsafe";

TEST(MallocFree, NullArguments) {
    EXPECT_EQ(cudaErrorInvalidValue, cudaMalloc(NULL, 0));
    EXPECT_EQ(cudaErrorInvalidValue, cudaMalloc(NULL, 4));
    EXPECT_EQ(cudaSuccess, cudaFree(NULL));
}

TEST(MallocFree, Simple) {
    /* Allocate and free a small piece of memory to see nothing explodes */
    void * ptr;
    cudaError_t ret;
    ret = cudaMalloc(&ptr, sizeof(ptr));
    ASSERT_EQ(cudaSuccess, ret);
    ASSERT_NE((void *) NULL, ptr);

    ret = cudaFree(ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(MallocFree, Validity) {
    /* Allocate a piece of memory, transfer it to the host, verify it is uninitialized */
    void * ptr;
    cudaError_t ret;
    ret = cudaMalloc(&ptr, sizeof(ptr));
    ASSERT_EQ(cudaSuccess, ret);
    ASSERT_NE((void *) NULL, ptr);

    /* This sets the validity bits, so we need the memcpy to set them as well or the test fails */
    uintptr_t ptr_ = 0;
    BOOST_STATIC_ASSERT(sizeof(ptr_) == sizeof(ptr));

    ret = cudaMemcpy(&ptr_, ptr, sizeof(ptr), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);
    uintptr_t ptr_validity = 0;

    BOOST_STATIC_ASSERT(sizeof(ptr_validity) == sizeof(ptr_));
    int valgrind = VALGRIND_GET_VBITS(&ptr_, &ptr_validity, sizeof(ptr_));
    /* valgrind == 3 indicates we didn't do something right on the test side. */
    assert(valgrind == 0 || valgrind == 1);

    if (valgrind == 1) {
        uintptr_t ptr_vexpected = ~((uintptr_t) 0);
        EXPECT_EQ(ptr_vexpected, ptr_validity);
    }

    ret = cudaFree(ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
