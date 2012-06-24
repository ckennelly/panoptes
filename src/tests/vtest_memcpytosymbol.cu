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

typedef uint32_t symbol_t;
__device__   symbol_t device_symbol;
__device__   symbol_t device_symbol2;

TEST(MemcpyToSymbol, Invalid) {
    const char missing[] = "this_symbol_does_not_exist";
    cudaError_t ret;
    symbol_t target;

    ret = cudaMemcpyToSymbol(missing, NULL, sizeof(symbol_t), 0,
        cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaMemcpyToSymbol(missing, &target, sizeof(symbol_t), 0,
        cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaMemcpyToSymbol(NULL, NULL, sizeof(symbol_t), 0,
        cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaMemcpyToSymbol(NULL, &target, sizeof(symbol_t), 0,
        cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);
}

__device__ symbol_t device_symbol_basic;

TEST(MemcpyToSymbol, Basic) {
    cudaError_t ret;
    void * ptr;

    ret = cudaGetSymbolAddress(&ptr, device_symbol_basic);
    ASSERT_EQ(cudaSuccess, ret);

    symbol_t target, dtarget, ftarget, vtarget;
    memset(&target, 0xAA, sizeof(symbol_t));

    ret = cudaMemcpyToSymbol(device_symbol_basic, &target,
        sizeof(symbol_t), 0, cudaMemcpyHostToDevice);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&dtarget, ptr, sizeof(symbol_t),
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    int valgrind = VALGRIND_GET_VBITS(&dtarget, &vtarget, sizeof(symbol_t));
    assert(valgrind == 0 || valgrind == 1);

    EXPECT_EQ(0xAAAAAAAA, dtarget);

    ret = cudaMemcpyFromSymbol(&ftarget, device_symbol_basic,
        sizeof(symbol_t), 0, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    if (valgrind == 1) {
        const symbol_t valid = 0;

        EXPECT_EQ(valid, vtarget);
    } else {
        return;
    }

    valgrind = VALGRIND_GET_VBITS(&ftarget, &vtarget, sizeof(symbol_t));
    assert(valgrind == 0 || valgrind == 1);

    EXPECT_EQ(0xAAAAAAAA, ftarget);

    if (valgrind == 1) {
        const symbol_t valid = 0;

        EXPECT_EQ(valid, vtarget);
    } else {
        return;
    }
}

TEST(MemcpyToSymbol, ByAddress) {
    cudaError_t ret;
    void * ptr;

    ret = cudaGetSymbolAddress(&ptr, device_symbol);
    ASSERT_EQ(cudaSuccess, device_symbol);

    ret = cudaMemset(ptr, 0xAA, sizeof(symbol_t));
    ASSERT_EQ(cudaSuccess, device_symbol);

    symbol_t target;
    ret = cudaMemcpyToSymbol(ptr, &target, sizeof(symbol_t), 0,
        cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);
}

TEST(MemcpyToSymbol, DeviceToDevice) {
    cudaError_t ret;
    void * device_ptr;
    void * symbol_ptr;

    ret = cudaMalloc(&device_ptr, sizeof(symbol_t));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaGetSymbolAddress(&symbol_ptr, device_symbol);
    ASSERT_EQ(cudaSuccess, device_symbol);

    const int pattern = 0xAA;
    ret = cudaMemset(device_ptr, pattern, sizeof(symbol_t));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyToSymbol(device_symbol, device_ptr,
        sizeof(symbol_t), 0, cudaMemcpyDeviceToDevice);
    EXPECT_EQ(cudaSuccess, ret);

    if (ret == cudaSuccess) {
        symbol_t expected, target, vtarget;
        BOOST_STATIC_ASSERT(sizeof(symbol_t) == sizeof(device_symbol));
        ret = cudaMemcpy(&target, symbol_ptr, sizeof(symbol_t),
            cudaMemcpyDeviceToHost);
        ASSERT_EQ(cudaSuccess, ret);

        memset(&expected, pattern, sizeof(expected));

        int valgrind = VALGRIND_GET_VBITS(&target, &vtarget, sizeof(symbol_t));
        assert(valgrind == 0 || valgrind == 1);

        EXPECT_EQ(expected, target);

        if (valgrind == 1) {
            const uint32_t valid = 0;
            BOOST_STATIC_ASSERT(sizeof(valid) == sizeof(target));

            EXPECT_EQ(valid, vtarget);
        }
    }

    ret = cudaFree(device_ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(MemcpyToSymbol, InvalidDirections) {
    cudaError_t ret;
    symbol_t target;

    ret = cudaMemcpyToSymbol(device_symbol, &target,
        sizeof(symbol_t), 0, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaErrorInvalidMemcpyDirection, ret);

    ret = cudaMemcpyToSymbol(device_symbol, &target,
        sizeof(symbol_t), 0, cudaMemcpyHostToHost);
    EXPECT_EQ(cudaErrorInvalidMemcpyDirection, ret);

    /* cudaMemcpyDefault implemented in test_memcpytosymbol.cu */
}

TEST(MemcpyToSymbol, NonSymbol) {
    cudaError_t ret;
    symbol_t target;
    void * device_ptr;

    ret = cudaMalloc(&device_ptr, sizeof(symbol_t));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyToSymbol(device_ptr, &target,
        sizeof(symbol_t), 0, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaFree(device_ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(MemcpyToSymbol, OutOfBounds) {
    cudaError_t ret;
    symbol_t target[2];

    ret = cudaMemcpyToSymbol(device_symbol, target,
        sizeof(symbol_t), sizeof(symbol_t), cudaMemcpyHostToDevice);
    ASSERT_EQ(cudaErrorInvalidValue, ret);
}

TEST(MemcpyToSymbol, SymbolToSymbol) {
    cudaError_t ret;
    void * device_symbol2_ptr;

    ret = cudaGetSymbolAddress(&device_symbol2_ptr, device_symbol2);
    ASSERT_EQ(cudaSuccess, ret);

    BOOST_STATIC_ASSERT(sizeof(device_symbol) == sizeof(device_symbol2));
    ret = cudaMemcpyToSymbol(device_symbol, device_symbol2_ptr,
        sizeof(symbol_t), 0, cudaMemcpyDeviceToDevice);
    EXPECT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
