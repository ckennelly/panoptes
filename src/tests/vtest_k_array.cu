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

/**
 * Copies a parameter array to the specified memory location.
 */
template<typename T>
static __global__ void k_copy(uint8_t * out, const T in) {
    reinterpret_cast<T *>(out)[0] = in;
}

TEST(kArray, Copy) {
    cudaError_t ret;
    cudaStream_t stream;

    uint8_t  expected[32 * 5];
    uint8_t vexpected[32 * 5];
    memset( expected, 0x0,  sizeof( expected));
    memset(vexpected, 0xFF, sizeof(vexpected));
    for (size_t i = 0; (1u << i) < 32; i++) {
        memset( expected + 32 * i, 0xFF, 1u << i);
        memset(vexpected + 32 * i, 0x0,  1u << i);
    }

    uint8_t * out;
    ret = cudaMalloc((void **) &out, sizeof(expected));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    {
        uchar1 tmp;
        tmp.x = 0xFF;
        k_copy<<<1, 1, 0, stream>>>(out + 32 * 0, tmp);
    }

    {
        uchar2 tmp;
        tmp.x = tmp.y = 0xFF;
        k_copy<<<1, 1, 0, stream>>>(out + 32 * 1, tmp);
    }

    {
        uchar4 tmp;
        tmp.x = tmp.y = tmp.z = tmp.w = 0xFF;
        k_copy<<<1, 1, 0, stream>>>(out + 32 * 2, tmp);
    }

    {
        ushort4 tmp;
        tmp.x = tmp.y = tmp.z = tmp.w = 0xFFFF;
        k_copy<<<1, 1, 0, stream>>>(out + 32 * 3, tmp);
    }

    {
        uint4 tmp;
        tmp.x = tmp.y = tmp.z = tmp.w = 0xFFFFFFFF;
        k_copy<<<1, 1, 0, stream>>>(out + 32 * 4, tmp);
    }

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint8_t hout[32 * 5];
    BOOST_STATIC_ASSERT(sizeof(expected) == sizeof(hout));
    ret = cudaMemcpy(hout, out, sizeof(expected), cudaMemcpyDeviceToHost);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    if (RUNNING_ON_VALGRIND) {
        uint8_t vout[32 * 5];
        BOOST_STATIC_ASSERT(sizeof(vout) == sizeof(hout));
        VALGRIND_GET_VBITS(hout, vout, sizeof(hout));

        const int vret = memcmp(vout, vexpected, sizeof(vout));
        EXPECT_EQ(0, vret);
        VALGRIND_MAKE_MEM_DEFINED(hout, sizeof(hout));
    }

    /* We do not want to check the values of the gaps, so mask them out. */
    for (size_t i = 0; i < sizeof(expected); i++) {
        hout[i] &= expected[i];
    }

    const int dret = memcmp(hout, expected, sizeof(expected));
    EXPECT_EQ(0, dret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
