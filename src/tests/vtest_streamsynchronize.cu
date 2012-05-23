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

TEST(StreamSynchronize, NoWork) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(StreamSynchronize, Work) {
    cudaError_t ret;
    cudaStream_t stream;

    int * src;
    int * dst;
    const size_t allocation_size = 1u << 20;
    ret = cudaMalloc((void **) &src, allocation_size * sizeof(*src));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &dst, allocation_size * sizeof(*dst));
    ASSERT_EQ(cudaSuccess, ret);

    BOOST_STATIC_ASSERT(sizeof(*src) == sizeof(*dst));

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyAsync(dst, src, allocation_size * sizeof(*src),
        cudaMemcpyDeviceToDevice, stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(dst);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(src);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
