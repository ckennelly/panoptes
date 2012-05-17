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

typedef int symbol_t;
__constant__ symbol_t const_symbol;
__device__   symbol_t device_symbol;

TEST(GetSymbolSize, Invalid) {
    const char missing[] = "this_symbol_does_not_exist";
    cudaError_t ret;
    size_t size;

    ret = cudaGetSymbolSize(NULL, missing);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaGetSymbolSize(&size, missing);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaGetSymbolSize(NULL, NULL);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaGetSymbolSize(&size, NULL);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);
}

TEST(GetSymbolSize, ConstantSymbol) {
    cudaError_t ret;
    size_t size;

    ret = cudaGetSymbolSize(&size, const_symbol);
    ASSERT_EQ(cudaSuccess, ret);
    EXPECT_EQ(sizeof(symbol_t), size);
}

TEST(GetSymbolSize, DeviceSymbol) {
    cudaError_t ret;
    size_t size;

    ret = cudaGetSymbolSize(&size, device_symbol);
    ASSERT_EQ(cudaSuccess, ret);
    EXPECT_EQ(sizeof(symbol_t), size);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
