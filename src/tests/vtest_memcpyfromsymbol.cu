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

TEST(MemcpyFromSymbol, Invalid) {
    const char missing[] = "this_symbol_does_not_exist";
    cudaError_t ret;
    symbol_t target;

    ret = cudaMemcpyFromSymbol(NULL, missing, sizeof(symbol_t), 0,
        cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaMemcpyFromSymbol(&target, missing, sizeof(symbol_t), 0,
        cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaMemcpyFromSymbol(NULL, NULL, sizeof(symbol_t), 0,
        cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaMemcpyFromSymbol(&target, NULL, sizeof(symbol_t), 0,
        cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);
}

TEST(MemcpyFromSymbol, Basic) {
    cudaError_t ret;
    void * ptr;

    ret = cudaGetSymbolAddress(&ptr, device_symbol);
    ASSERT_EQ(cudaSuccess, device_symbol);

    ret = cudaMemset(ptr, 0xAA, sizeof(symbol_t));
    ASSERT_EQ(cudaSuccess, device_symbol);

    symbol_t atarget, target, vtarget;
    BOOST_STATIC_ASSERT(sizeof(atarget) == sizeof(device_symbol));
    ret = cudaMemcpyFromSymbol(&atarget, "device_symbol",
        sizeof(symbol_t), 0, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyFromSymbol( &target, device_symbol,
        sizeof(symbol_t), 0, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    int valgrind = VALGRIND_GET_VBITS(&atarget, &vtarget, sizeof(symbol_t));
    assert(valgrind == 0 || valgrind == 1);

    if (valgrind == 1) {
        const symbol_t valid = 0;

        EXPECT_EQ(valid, vtarget);
    } else {
        return;
    }

    valgrind = VALGRIND_GET_VBITS(&target, &vtarget, sizeof(symbol_t));
    assert(valgrind == 0 || valgrind == 1);

    if (valgrind == 1) {
        const symbol_t valid = 0;

        EXPECT_EQ(valid, vtarget);
    }
}

TEST(MemcpyFromSymbol, ByAddress) {
    cudaError_t ret;
    void * ptr;

    ret = cudaGetSymbolAddress(&ptr, device_symbol);
    ASSERT_EQ(cudaSuccess, device_symbol);

    ret = cudaMemset(ptr, 0xAA, sizeof(symbol_t));
    ASSERT_EQ(cudaSuccess, device_symbol);

    symbol_t target;
    ret = cudaMemcpyFromSymbol(&target, ptr, sizeof(symbol_t), 0,
        cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);
}

TEST(MemcpyFromSymbol, DeviceToDevice) {
    cudaError_t ret;
    void * device_ptr;
    void * symbol_ptr;

    ret = cudaMalloc(&device_ptr, sizeof(symbol_t));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaGetSymbolAddress(&symbol_ptr, device_symbol);
    ASSERT_EQ(cudaSuccess, device_symbol);

    const int pattern = 0xAA;
    ret = cudaMemset(symbol_ptr, pattern, sizeof(symbol_t));
    ASSERT_EQ(cudaSuccess, device_symbol);

    ret = cudaMemcpyFromSymbol(device_ptr, device_symbol,
        sizeof(symbol_t), 0, cudaMemcpyDeviceToDevice);
    EXPECT_EQ(cudaSuccess, ret);

    if (ret == cudaSuccess) {
        symbol_t expected, target, vtarget;
        BOOST_STATIC_ASSERT(sizeof(symbol_t) == sizeof(device_symbol));
        ret = cudaMemcpy(&target, device_ptr, sizeof(symbol_t),
            cudaMemcpyDeviceToHost);
        ASSERT_EQ(cudaSuccess, ret);

        memset(&expected, pattern, sizeof(expected));

        int valgrind = VALGRIND_GET_VBITS(&target, &vtarget, sizeof(symbol_t));
        assert(valgrind == 0 || valgrind == 1);

        /*
         * Suppress validity warnings on target.
         */
        if (valgrind == 1) {
            VALGRIND_MAKE_MEM_DEFINED(&target, sizeof(target));
        }

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

TEST(MemcpyFromSymbol, InvalidDirections) {
    cudaError_t ret;
    symbol_t target;

    ret = cudaMemcpyFromSymbol(&target, device_symbol,
        sizeof(symbol_t), 0, cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaErrorInvalidMemcpyDirection, ret);

    ret = cudaMemcpyFromSymbol(&target, device_symbol,
        sizeof(symbol_t), 0, cudaMemcpyHostToHost);
    EXPECT_EQ(cudaErrorInvalidMemcpyDirection, ret);

    /* cudaMemcpyDefault implemented in test_memcpyfromsymbol.cu */
}

TEST(MemcpyFromSymbol, NonSymbol) {
    cudaError_t ret;
    symbol_t target;
    void * device_ptr;

    ret = cudaMalloc(&device_ptr, sizeof(symbol_t));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyFromSymbol(&target, device_ptr,
        sizeof(symbol_t), 0, cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaErrorInvalidSymbol, ret);

    ret = cudaFree(device_ptr);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(MemcpyFromSymbol, OutOfBounds) {
    cudaError_t ret;
    symbol_t target[2];

    ret = cudaMemcpyFromSymbol(target, device_symbol,
        sizeof(symbol_t), sizeof(symbol_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaErrorInvalidValue, ret);
}

TEST(MemcpyFromSymbol, SymbolToSymbol) {
    cudaError_t ret;
    void * device_symbol2_ptr;

    ret = cudaGetSymbolAddress(&device_symbol2_ptr, device_symbol2);
    ASSERT_EQ(cudaSuccess, ret);

    BOOST_STATIC_ASSERT(sizeof(device_symbol) == sizeof(device_symbol2));
    ret = cudaMemcpyFromSymbol(device_symbol2_ptr, device_symbol,
        sizeof(symbol_t), 0, cudaMemcpyDeviceToDevice);
    EXPECT_EQ(cudaSuccess, ret);
}

__device__   symbol_t initialized_symbol = 0xDEADBEEF;

TEST(MemcpyFromSymbol, InitializedSymbol) {
    cudaError_t ret;
    void * ptr;

    symbol_t target;
    BOOST_STATIC_ASSERT(sizeof(target) == sizeof(initialized_symbol));
    ret = cudaMemcpyFromSymbol(&target, initialized_symbol,
        sizeof(target), 0, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    symbol_t vtarget;
    unsigned valgrind = VALGRIND_GET_VBITS(&target, &vtarget, sizeof(target));
    assert(valgrind == 0 || valgrind == 1);
    if (valgrind == 1) {
        const symbol_t vexpected = 0;
        EXPECT_EQ(vexpected, vtarget);
    }

    VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&target, sizeof(target));
    EXPECT_EQ(0xDEADBEEF, target);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
