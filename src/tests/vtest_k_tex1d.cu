/**
 * Panoptes - A Binary Translation Framework for CUDA
 * (c) 2011-2013 Chris Kennelly <chris@ckennelly.com>
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
#include <stdint.h>

texture<int32_t, 1, cudaReadModeElementType> tex_src;

extern "C" __global__ void k_readtex1d(void * dst, int32_t bytes) {
    bytes = (bytes + 3) / 4;

    for (int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            idx < bytes; idx += blockDim.x * gridDim.x) {
        static_cast<int32_t *>(dst)[idx] = tex1Dfetch(tex_src, idx);
    }
}

TEST(kReadTex1D, ExplicitStream) {
    cudaError_t ret;
    cudaStream_t stream;

    const uint32_t bytes = 1u << 20;
    int32_t *src;
    int32_t *dst;

    ret = cudaMalloc(&src, bytes);
    ASSERT_EQ(cudaSuccess, ret);
    ret = cudaMalloc(&dst, bytes);
    ASSERT_EQ(cudaSuccess, ret);

    const textureReference* texref;
    #if CUDA_VERSION < 5000
    ret = cudaGetTextureReference(&texref, "tex_src");
    #else
    ret = cudaGetTextureReference(&texref, &tex_src);
    #endif
    ASSERT_EQ(cudaSuccess, ret); 

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT * sizeof(*src);
    desc.y = desc.z = desc.w = 0;
    ret = cudaBindTexture(NULL, texref, src, &desc, bytes);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_readtex1d<<<1, 1, 0, stream>>>(dst, bytes);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
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
