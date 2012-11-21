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

#include <algorithm>
#include <boost/static_assert.hpp>
#include <cuda.h>
#include <gtest/gtest.h>
#include <valgrind/memcheck.h>
#include <vector>

__global__ void k_popc(bool pred, int * out) {
    *out = __syncthreads_count(pred);
}

TEST(Irregular, PopCount) {
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * Derive block dimensions by adding powers of two together.
     */
    std::vector<int> dims[3];
    for (size_t dim = 0; dim < 3; dim++) {
        const int limit = prop.maxThreadsDim[dim];
        std::vector<int> powers;
        for (int i = 1; i < limit; i *= 2) {
            powers.push_back(i);
        }
        const size_t N = powers.size();

        for (size_t i = 0; i < N; i++) {
            const int outer = powers[i];
            for (size_t j = i; j < N; j++) {
                const int inner = powers[j];
                const int sum   = outer + inner;
                if (sum <= limit) {
                    dims[dim].push_back(sum);
                }
            }
        }
        dims[dim].push_back(1);

        std::sort(dims[dim].begin(), dims[dim].end());
    }

    /**
     * Try various block dimensions.
     */
    const size_t Nx = dims[0].size();
    const size_t Ny = dims[1].size();
    const size_t Nz = dims[2].size();
    const int    block_limit = prop.maxThreadsPerBlock;

    int * out;
    ret = cudaMalloc((void **) &out, 2 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    for (size_t ix = 0; ix < Nx; ix++) {
        const int x = dims[0][ix];
        for (size_t iy = 0; iy < Ny; iy++) {
            const int y = dims[1][iy];
            for (size_t iz = 0; iz < Nz; iz++) {
                const int z = dims[2][iz];
                const int size = x * y * z;
                if (size > block_limit) {
                    /* sizes monotonically increase in iz. */
                    break;
                }

                /**
                 * First with initialized predicate, then without.
                 */
                bool pred = true;

                const dim3 block(x, y, z);
                k_popc<<<1, block, 0, stream>>>(pred, out + 0);

                const bool valgrind = RUNNING_ON_VALGRIND;
                if (valgrind) {
                    VALGRIND_MAKE_MEM_UNDEFINED(&pred, sizeof(pred));
                    k_popc<<<1, block, 0, stream>>>(pred, out + 1);
                }

                ret = cudaStreamSynchronize(stream);
                ASSERT_EQ(cudaSuccess, ret);

                int hout[2];
                ret = cudaMemcpy(hout, out, sizeof(hout),
                    cudaMemcpyDeviceToHost);
                ASSERT_EQ(cudaSuccess, ret);

                EXPECT_EQ(size, hout[0]);
                if (valgrind) {
                    unsigned vout;
                    BOOST_STATIC_ASSERT(sizeof(vout) == sizeof(hout[1]));
                    VALGRIND_GET_VBITS(&hout[1], &vout, sizeof(&hout[1]));

                    /**
                     * Round up to next power of two, then create mask of all
                     * 1's with the highest set bit and all bits to its left.
                     *
                     * Blocks of more than 2^32 threads should not be relevant.
                     */
                    unsigned roundup = size;
                    roundup--;
                    roundup |= roundup >> 1;
                    roundup |= roundup >> 2;
                    roundup |= roundup >> 4;
                    roundup |= roundup >> 8;
                    roundup |= roundup >> 16;

                    roundup |= (roundup + 1);

                    EXPECT_EQ(roundup, vout);
                }
            }
        }
    }

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
