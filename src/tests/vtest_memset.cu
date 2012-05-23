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
#include <valgrind/memcheck.h>

TEST(Memset, NullArguments) {
    EXPECT_EQ(cudaSuccess, cudaMemset(NULL, 0, 0));

    /* Caught by Panoptes.  Disabling for now
    EXPECT_EQ(cudaErrorInvalidValue, cudaMemset(NULL, 0, 4));
    cudaGetLastError(); */
}

TEST(Memset, MallocAfterMemset) {
    cudaError_t ret;
    void *ptr1, *ptr2;
    const size_t block = 1 << 10;

    ret = cudaMalloc(&ptr1, block);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemset(ptr1, 0, block);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc(&ptr2, block);
    ASSERT_EQ(cudaSuccess, ret);

    // Download data
    void *hptr1;
    ret = cudaMallocHost(&hptr1, block);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(hptr1, ptr1, block, cudaMemcpyDeviceToHost);

    // Copy out validity bits
    uint8_t * vptr1 = new uint8_t[block];
    int valgrind = VALGRIND_GET_VBITS(hptr1, vptr1, block);
    assert(valgrind == 0 || valgrind == 1);

    // Check if Valgrind is running
    if (valgrind == 1) {
        uint8_t * eptr1 = new uint8_t[block];
        memset(eptr1, 0x0, block);

        EXPECT_EQ(0, memcmp(vptr1, eptr1, block));
        delete[] eptr1;
    }

    delete[] vptr1;

    ret = cudaFree(ptr2);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(ptr1);
    ASSERT_EQ(cudaSuccess, ret);
}

class MemsetValidity : public ::testing::TestWithParam<int> {
    // Empty Fixture
};

TEST_P(MemsetValidity, Aligned) {
    const size_t param = GetParam();
    const size_t alloc = sizeof(void *) << param;

    uint8_t * ptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc((void **) &ptr, alloc));

    uint8_t *  data  = new uint8_t[alloc];
    uint8_t * vdata  = new uint8_t[alloc];
    uint8_t * expect = new uint8_t[alloc];
    memset(expect, 0xFF, alloc);

    // Write a pattern to the actual data and an expected validity pattern
    for (size_t i = 0; i < param; i += 2) {
        // Write
        const size_t range  = sizeof(void *) *  (1 << i);
        assert(range * 2 <= alloc);

        cudaError_t ret;
        ASSERT_EQ(cudaSuccess, cudaMemset(ptr    + range, i & 0x0, range));
                                   memset(expect + range,     0x0, range);
    }

    // Download data
    ASSERT_EQ(cudaSuccess, cudaMemcpy(data, ptr, alloc, cudaMemcpyDeviceToHost));

    // Copy out validity bits
    int valgrind = VALGRIND_GET_VBITS(data, vdata, alloc);
    assert(valgrind == 0 || valgrind == 1);

    // Check if Valgrind is running
    if (valgrind == 1) {
        EXPECT_EQ(0, memcmp(vdata, expect, alloc));
    }

    delete[] expect;
    delete[] vdata;
    delete[]  data;

    ASSERT_EQ(cudaSuccess, cudaFree(ptr));
}

INSTANTIATE_TEST_CASE_P(MemsetValidityInst, MemsetValidity,
    ::testing::Range(1, 20));

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
