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
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

typedef int32_t tex_t;
texture<tex_t, 1, cudaReadModeElementType> tex_src;

class TextureValues : public ::testing::TestWithParam<int> {
    // Empty Fixture
};

static __global__ void k_set(tex_t * out, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
            i += blockDim.x * gridDim.x) {
        out[i] = static_cast<tex_t>(i);
    }
}

static __global__ void k_copy(tex_t * out, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
            i += blockDim.x * gridDim.x) {
        out[i] = tex1Dfetch(tex_src, i);
    }
}

TEST_P(TextureValues, DataCopy) {
    /**
     * Verify we can read the values from a texture.
     */
    const int param = GetParam();
    const int alloc = 1 << param;

    const int n_threads = 256;
    const int n_blocks  = (alloc + n_threads - 1) / n_threads;

    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    if (alloc > prop.maxTexture1DLinear) {
        return;
    }

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    tex_t *tex;
    ret = cudaMalloc((void **) &tex, 2 * sizeof(*tex) * alloc);
    ASSERT_EQ(cudaSuccess, ret);

    /* Only half of the values are initialized. */
    k_set<<<n_blocks, n_threads, 0, stream>>>(tex, alloc);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    const struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<tex_t>();
    tex_src.addressMode[0] = cudaAddressModeClamp;
    tex_src.filterMode = cudaFilterModePoint;
    tex_src.normalized = false;

    ret = cudaBindTexture(NULL, tex_src, tex, desc, 2 * sizeof(*tex) * alloc);
    ASSERT_EQ(cudaSuccess, ret);

    /* Allocate output. */
    tex_t *out;
    ret = cudaMalloc((void **) &out, 2 * sizeof(*out) * alloc);
    ASSERT_EQ(cudaSuccess, ret);

    /* Run kernel. */
    k_copy<<<n_blocks, n_threads, 0, stream>>>(out, 2 * alloc);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    std::vector<tex_t> hout(2 * alloc);
    ret = cudaMemcpy(&hout[0], out, 2 * sizeof(*out) * alloc,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(tex);
    ASSERT_EQ(cudaSuccess, ret);

    if (RUNNING_ON_VALGRIND) {
        /* Check validity bits. */
        std::vector<uint32_t> vout(2 * alloc);
        BOOST_STATIC_ASSERT(sizeof(vout[0]) == sizeof(hout[0]));
        int vret = VALGRIND_GET_VBITS(&hout[0], &vout[0],
            2 * sizeof(hout[0]) * alloc);
        ASSERT_EQ(1, vret);

        bool error = false;
        int i;
        /* First half is fully initialized. */
        for (i = 0; i < alloc; i++) {
            error |= vout[i] != 0;
        }
        EXPECT_FALSE(error);

        /* Second half is uninitialized. */
        for (; i < 2 * alloc; i++) {
            error |= vout[i] != 0xFFFFFFFF;
        }
        EXPECT_FALSE(error);
    }

    bool error = false;
    for (int i = 0; i < alloc; i++) {
        const tex_t expected = static_cast<tex_t>(i);
        error |= expected != hout[i];
    }
    EXPECT_FALSE(error);
}

INSTANTIATE_TEST_CASE_P(TextureInst, TextureValues, ::testing::Range(1, 22));

static __global__ void k_load(tex_t * out, int index) {
    *out = tex1Dfetch(tex_src, index);
}

TEST(Textures, WildLoad) {
    /**
     * Allocate a buffer and initialize it.  Then perform a read from the
     * texture at an uninitialized index.
     */
    if (!(RUNNING_ON_VALGRIND)) {
        return;
    }

    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const size_t n_elements = 1u << 16;
    const size_t n_threads  = 256;
    const size_t n_blocks   = (n_elements + n_threads - 1) / n_threads;

    tex_t *tex;
    ret = cudaMalloc((void **) &tex, sizeof(*tex) * n_elements);
    ASSERT_EQ(cudaSuccess, ret);

    k_set<<<n_blocks, n_threads, 0, stream>>>(tex, n_elements);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    const struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<tex_t>();
    tex_src.addressMode[0] = cudaAddressModeClamp;
    tex_src.filterMode = cudaFilterModePoint;
    tex_src.normalized = false;

    ret = cudaBindTexture(NULL, tex_src, tex, desc, sizeof(*tex) * n_elements);
    ASSERT_EQ(cudaSuccess, ret);

    /* Allocate output. */
    tex_t *out;
    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    /* Run kernel. */
    int index = 0;
    VALGRIND_MAKE_MEM_UNDEFINED(&index, sizeof(index));
    k_load<<<1, 1, 0, stream>>>(out, index);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    tex_t hout;
    ret = cudaMemcpy(&hout, out, sizeof(*out), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(tex);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t vout;
    BOOST_STATIC_ASSERT(sizeof(vout) == sizeof(hout));
    int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    EXPECT_EQ(1, vret);
    EXPECT_EQ(0xFFFFFFFF, vout);
}

static __global__ void k_txq(unsigned * out) {
    unsigned tmp;
    asm("txq.width.b32 %0, [tex_src];" : "=r"(tmp));
    out[0] = tmp;

    asm("txq.height.b32 %0, [tex_src];" : "=r"(tmp));
    out[1] = tmp;

    asm("txq.depth.b32 %0, [tex_src];" : "=r"(tmp));
    out[2] = tmp;

    asm("txq.channel_data_type.b32 %0, [tex_src];" : "=r"(tmp));
    out[3] = tmp;

    asm("txq.channel_order.b32 %0, [tex_src];" : "=r"(tmp));
    out[4] = tmp;

    asm("txq.normalized_coords.b32 %0, [tex_src];" : "=r"(tmp));
    out[5] = tmp;

    asm("txq.filter_mode.b32 %0, [tex_src];" : "=r"(tmp));
    out[6] = tmp;

    asm("txq.addr_mode_0.b32 %0, [tex_src];" : "=r"(tmp));
    out[7] = tmp;

    asm("txq.addr_mode_1.b32 %0, [tex_src];" : "=r"(tmp));
    out[8] = tmp;

    asm("txq.addr_mode_2.b32 %0, [tex_src];" : "=r"(tmp));
    out[9] = tmp;
}

TEST(Textures, Query) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const size_t n_elements = 1 << 16;

    tex_t *tex;
    ret = cudaMalloc((void **) &tex, sizeof(*tex) * n_elements);
    ASSERT_EQ(cudaSuccess, ret);

    const struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<tex_t>();
    tex_src.addressMode[0] = cudaAddressModeClamp;
    tex_src.filterMode = cudaFilterModePoint;
    tex_src.normalized = false;

    ret = cudaBindTexture(NULL, tex_src, tex, desc, sizeof(*tex) * n_elements);
    ASSERT_EQ(cudaSuccess, ret);

    /* Allocate output. */
    unsigned hout[10];
    unsigned *out;
    ret = cudaMalloc((void **) &out, sizeof(hout));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemset(out, 0xFF, sizeof(hout));
    ASSERT_EQ(cudaSuccess, ret);

    /* Run kernel. */
    k_txq<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(tex);
    ASSERT_EQ(cudaSuccess, ret);

    /*
     * The values returned by txq return 0.
     *
     * 0: width
     * 1: height
     * 2: depth
     * 3: channel_data_type
     * 4: channel_order
     * 5: normalized_coords
     * 6: filter_mode
     * 7: addr_mode_0
     * 8: addr_mode_1
     * 9: addr_mode_2
     */
    EXPECT_EQ(0, hout[0]);
    EXPECT_EQ(0, hout[1]);
    EXPECT_EQ(0, hout[2]);
    EXPECT_EQ(0, hout[3]);
    EXPECT_EQ(0, hout[4]);
    EXPECT_EQ(0, hout[5]);
    EXPECT_EQ(0, hout[6]);
    EXPECT_EQ(0, hout[7]);
    EXPECT_EQ(0, hout[8]);
    EXPECT_EQ(0, hout[9]);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
