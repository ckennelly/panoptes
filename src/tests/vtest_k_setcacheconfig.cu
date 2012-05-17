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
#include <boost/static_assert.hpp>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>
#include <cstdio>

extern "C" __global__ void k_noop() {

}

TEST(kNOOP, SetCacheConfig) {
    cudaError_t ret;

    ret = cudaFuncSetCacheConfig(k_noop, cudaFuncCachePreferNone);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFuncSetCacheConfig(k_noop, cudaFuncCachePreferShared);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFuncSetCacheConfig(k_noop, cudaFuncCachePreferL1);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFuncSetCacheConfig(k_noop, cudaFuncCachePreferEqual);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
