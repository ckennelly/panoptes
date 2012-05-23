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

typedef int symbol_t;
__constant__ symbol_t const_symbol;
__device__   symbol_t device_symbol;

TEST(GetSymbolAddress, Invalid) {
    const char missing[] = "this_symbol_does_not_exist";
    cudaError_t ret;
    void * ptr;

    ret = cudaGetSymbolAddress(NULL, missing);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaGetSymbolAddress(&ptr, missing);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaGetSymbolAddress(NULL, NULL);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaGetSymbolAddress(&ptr, NULL);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);
}

TEST(GetSymbolAddress, ConstantSymbol) {
    cudaError_t ret;
    void * ptr;

    ret = cudaGetSymbolAddress(&ptr, const_symbol);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaPointerAttributes attr;
    ret = cudaPointerGetAttributes(&attr, ptr);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
}

TEST(GetSymbolAddress, DeviceSymbol) {
    cudaError_t ret;
    void * ptr;

    ret = cudaGetSymbolAddress(&ptr, device_symbol);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaPointerAttributes attr;
    ret = cudaPointerGetAttributes(&attr, ptr);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
