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

extern "C" __global__ void k_memcpy(void * dst, const void * src,
        uint32_t bytes) {
    /* Very simple byte-by-byte transfer approach. */
    for (uint32_t i = 0; i < bytes; i++) {
        static_cast<char *>(dst)[i] = static_cast<const char *>(src)[i];
    }
}

/**
 * Initializes dst such that for i on [0, ints), dst[i] = i.
 */
extern "C" __global__ void k_initialize(uint * dst, uint32_t ints) {
    const uint32_t quads = ints / 4u;
    uint32_t base = 0;
    for (uint32_t i = 0; i < quads; i++, base += 4u) {
        uint4 quad;
        quad.x = base + 0;
        quad.y = base + 1;
        quad.z = base + 2;
        quad.w = base + 3;
        ((uint4 *) dst)[i] = quad;
    }

    const uint32_t singles = ints % 4u;
    for (uint32_t i = 0; i < singles; i++) {
        dst[base + i] = base + i;
    }
}

static bool check_pattern(const uint * data, uint32_t ints) {
    for (uint32_t i = 0; i < ints; i++) {
        if (data[i] != i) {
            return false;
        }
    }

    return true;
}

TEST(kMemcpy, ExplicitStream) {
    cudaError_t ret;
    cudaStream_t stream;

    const uint32_t bytes = 1u << 20;
    void *src, *dst;
    ret = cudaMalloc(&src, bytes);
    ASSERT_EQ(cudaSuccess, ret);
    ret = cudaMalloc(&dst, bytes);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_memcpy<<<1, 1, 0, stream>>>(dst, src, bytes);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(src);
    ASSERT_EQ(cudaSuccess, ret);
    ret = cudaFree(dst);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kMemcpy, Pattern) {
    cudaError_t ret;
    cudaStream_t stream;

    const uint32_t ints = 1u << 20;
    uint *src, *dst;
    ret = cudaMalloc(&src, sizeof(*src) * ints);
    ASSERT_EQ(cudaSuccess, ret);
    ret = cudaMalloc(&dst, sizeof(*dst) * ints);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_initialize<<<1, 1, 0, stream>>>(src, ints);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    { // Check src
        boost::scoped_array<uint> host(new uint[ints]);
        ret = cudaMemcpy(host.get(), src, sizeof(*src) * ints,
            cudaMemcpyDeviceToHost);
        ASSERT_EQ(cudaSuccess, ret);

        /* TODO:  Once validity transfers work, remove this */
        (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(host.get(),
            sizeof(*src) *ints);
        EXPECT_EQ(true, check_pattern(host.get(), ints));
    }

    BOOST_STATIC_ASSERT(sizeof(*dst) == sizeof(*src));
    k_memcpy<<<1, 1, 0, stream>>>(dst, src, sizeof(*src) * ints);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    { // Check dst
        boost::scoped_array<uint> host(new uint[ints]);
        ret = cudaMemcpy(host.get(), dst, sizeof(*dst) * ints,
            cudaMemcpyDeviceToHost);
        ASSERT_EQ(cudaSuccess, ret);

        /* TODO:  Once validity transfers work, remove this */
        (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(host.get(),
            sizeof(*src) *ints);
        EXPECT_EQ(true, check_pattern(host.get(), ints));
    }

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(src);
    ASSERT_EQ(cudaSuccess, ret);
    ret = cudaFree(dst);
    ASSERT_EQ(cudaSuccess, ret);
}

/**
 * This is a parallel (thread-aware) k_memcpy with a linear grid.
 */
extern "C" __global__ void k_memcpy_p(void * dst, const void * src,
        uint32_t bytes) {
    typedef uint4 transfer_t;
    const uint32_t quads = bytes / sizeof(transfer_t);
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < quads; i += blockDim.x * gridDim.x) {
        static_cast<transfer_t *>(dst)[i] =
            static_cast<const transfer_t *>(src)[i];
    }

    if (threadIdx.x == 0) {
        for (uint32_t i = bytes & ~(sizeof(transfer_t)); i < bytes; i++) {
            static_cast<char *>(dst)[i] =
                static_cast<const char *>(src)[i];
        }
    }
}

/**
 * This is a parallel (thread-aware) k_initialize with a linear grid.
 */
extern "C" __global__ void k_initialize_p(uint * dst, uint32_t ints) {
    const uint32_t quads = ints / 4u;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < quads; i += blockDim.x * gridDim.x) {
        uint4 quad;
        quad.x = 4u * i + 0u;
        quad.y = 4u * i + 1u;
        quad.z = 4u * i + 2u;
        quad.w = 4u * i + 3u;
        ((uint4 *) dst)[i] = quad;
    }

    if (threadIdx.x == 0) {
        const uint32_t singles = ints % 4u;
        const uint32_t base = ints & ~(0x3);
        for (uint32_t i = 0; i < singles; i++) {
            dst[base + i] = base + i;
        }
    }
}

TEST(kMemcpy, PatternP) {
    cudaError_t ret;
    cudaStream_t stream;

    const uint32_t ints = 1u << 20;
    uint *src, *dst;
    ret = cudaMalloc(&src, sizeof(*src) * ints);
    ASSERT_EQ(cudaSuccess, ret);
    ret = cudaMalloc(&dst, sizeof(*dst) * ints);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_initialize_p<<<256, 16, 0, stream>>>(src, ints);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    { // Check src
        boost::scoped_array<uint> host(new uint[ints]);
        ret = cudaMemcpy(host.get(), src, sizeof(*src) * ints,
            cudaMemcpyDeviceToHost);
        ASSERT_EQ(cudaSuccess, ret);

        /* TODO:  Once validity transfers work, remove this */
        (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(host.get(),
            sizeof(*src) *ints);
        EXPECT_EQ(true, check_pattern(host.get(), ints));
    }

    BOOST_STATIC_ASSERT(sizeof(*dst) == sizeof(*src));
    k_memcpy_p<<<256, 16, 0, stream>>>(dst, src, sizeof(*src) * ints);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    { // Check dst
        boost::scoped_array<uint> host(new uint[ints]);
        ret = cudaMemcpy(host.get(), dst, sizeof(*dst) * ints,
            cudaMemcpyDeviceToHost);
        ASSERT_EQ(cudaSuccess, ret);

        /* TODO:  Once validity transfers work, remove this */
        (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(host.get(),
            sizeof(*src) *ints);
        EXPECT_EQ(true, check_pattern(host.get(), ints));
    }

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(src);
    ASSERT_EQ(cudaSuccess, ret);
    ret = cudaFree(dst);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
