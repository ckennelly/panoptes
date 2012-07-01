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
#include <cstdio>

/**
 * Per http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
 */
size_t roundup(size_t v) {
    v--;

    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;

    v++;
    return v;
}

TEST(MemGetInfo, Nulls) {
    cudaError_t ret;
    size_t free, total;

    ret = cudaMemGetInfo(&free, &total);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemGetInfo(&free, NULL);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemGetInfo(NULL, &total);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemGetInfo(NULL, NULL);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(MemGetInfo, Efficiency) {
    cudaError_t ret;
    size_t allocated = 0;

    typedef std::vector<void *> vvector_t;
    vvector_t allocs;

    size_t total;
    for (int i = 0; i < 40; i++) {
        size_t free;
        ret = cudaMemGetInfo(&free, &total);
        /* This stops working if we can't get this */
        ASSERT_EQ(cudaSuccess, ret);

        bool success = false;
        size_t trial = roundup(free);
        while (trial > 0) {
            void * ptr;
            ret = cudaMalloc(&ptr, trial);
            if (ret == cudaSuccess) {
                allocs.push_back(ptr);
                allocated   += trial;
                success     = true;

                break;
            } else {
                trial /= 2;
            }
        }

        if (!(success)) {
            break;
        }

        size_t new_free;
        ret = cudaMemGetInfo(&new_free, NULL);
        /* This stops working if we can't get this */
        ASSERT_EQ(cudaSuccess, ret);

        double eff = (double) trial / (free - new_free);
        printf("Allocated %zu byte block with %f efficiency (out of %zu "
            "bytes).\n", trial, eff, free);
    }

    double teff = 100. * (double) allocated / total;
    printf("Total efficency: %f%% of %zu bytes\n", teff, total);

    for (vvector_t::iterator it = allocs.begin(); it != allocs.end(); ++it) {
        cudaFree(*it);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
