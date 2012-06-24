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

TEST(DeviceSynchronize, NoWork) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaDeviceSynchronize();
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(DeviceSynchronize, MemcpyAsync) {
    cudaError_t ret;
    cudaStream_t stream;

    const size_t count = 1 << 20;
    int * device_ptr;
    const size_t bytes = sizeof(*device_ptr) * count;
    ret = cudaMalloc((void **) &device_ptr, bytes);
    ASSERT_EQ(cudaSuccess, ret);

    const int pattern = 0xAA;
    ret = cudaMemset(device_ptr, pattern, bytes);

    int * host_ptr = new int[count];
    memset(host_ptr, 0x00, bytes);

    int * expected_ptr = new int[count];

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    BOOST_STATIC_ASSERT(sizeof(*host_ptr) == sizeof(*device_ptr));
    ret = cudaMemcpyAsync(host_ptr, device_ptr, bytes,
        cudaMemcpyDeviceToHost, stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaDeviceSynchronize();
    EXPECT_EQ(cudaSuccess, ret);

    BOOST_STATIC_ASSERT(sizeof(*expected_ptr) == sizeof(*host_ptr));

    /**
     * Sigh.  Valgrind has issues.  We can scan its entire set of vbits for
     * an address and find them to be all set (i.e., fully valid) but
     * performing memory operations (like the memcmp below) will throw out
     * warnings about uninitialized memory.  Calling
     * VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE suppresses this, but the
     * reason for why this is necessary is not clear.
     */
    int vret = VALGRIND_GET_VBITS(host_ptr, expected_ptr, bytes);
    if (vret == 1) {
        bool error = false;
        for (size_t i = 0; i < bytes; i++) {
            uint8_t v = ((uint8_t *) expected_ptr)[i];
            if (v != 0xFF) {
                error = true;
                break;
            }
        }

        if (!(error)) {
            /** Only suppress illegitimate warnings */
            VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(host_ptr, bytes);
        }
    }

    memset(expected_ptr, pattern, bytes);
    const int memcmp_ret = memcmp(expected_ptr, host_ptr, bytes);
    EXPECT_EQ(0, memcmp_ret);

    ret = cudaStreamDestroy(stream);
    EXPECT_EQ(cudaSuccess, ret);

    delete[] expected_ptr;
    delete[] host_ptr;
    ret = cudaFree(device_ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
