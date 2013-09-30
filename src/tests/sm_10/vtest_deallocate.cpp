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

#include <boost/unordered_set.hpp>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

TEST(Deallocate, Host) {
    /**
     * This test allocates N buffers.  It writes to them and then verfies
     * the data (and validity bits) on each as they are deallocated.
     *
     * For this test to be meaningful, at least one chunk must have two
     * separate allocations placed on it.
     */
    const size_t N = 16;
    const size_t block_size = 256;
    const size_t chunk_size = 1 << 16 /* 64k */;

    /**
     * Map upper bits of each allocated pointer to a bucket (by dividing by
     * chunk_size.  If any bucket already exists, a second (Nth) block has
     * been allocated sharing a chunk.
     */
    boost::unordered_set<uintptr_t> buckets;
    bool shared_chunk = false;

    char * buffers[N];
    /* Deallocate. */
    for (size_t i = 0; i < N; i++) {
        cudaError_t ret;
        ret = cudaMalloc((void **) &buffers[i], block_size);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaMemset(buffers[i], (int) i, block_size);
        ASSERT_EQ(cudaSuccess, ret);

        const uintptr_t upper_bits = ((uintptr_t) buffers[i]) / chunk_size;
        shared_chunk |= buckets.insert(upper_bits).second;
    }
    EXPECT_TRUE(shared_chunk);

    for (size_t i = 0; i < N; i++) {
        cudaError_t ret;

        for (size_t j = i; j < N; j++) {
            char tmp[block_size];
            ret = cudaMemcpy(tmp, buffers[j], block_size,
                cudaMemcpyDeviceToHost);
            ASSERT_EQ(cudaSuccess, ret);

            char expected[block_size];
            memset(expected, (int) j, sizeof(expected));
            int cmp = memcmp(tmp, expected, block_size);
            EXPECT_EQ(0, cmp);
        }

        ret = cudaFree(buffers[i]);
        EXPECT_EQ(cudaSuccess, ret);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
