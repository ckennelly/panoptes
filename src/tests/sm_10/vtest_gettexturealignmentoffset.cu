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

static bool bound = false;
texture<int32_t, 1, cudaReadModeElementType> tex_src;

TEST(GetTextureAlignmentOffset, Simple) {
    cudaError_t ret;
    const struct textureReference * texref;

    const uint32_t bytes = 1u << 20;
    signed char * data;
    ret = cudaMalloc((void **) &data, sizeof(*data) * bytes);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDA_VERSION < 5000
    ret = cudaGetTextureReference(&texref, "tex_src");
    #else
    ret = cudaGetTextureReference(&texref, &tex_src);
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT;
    desc.y = desc.z = desc.w = 0;

    { // offset 0
        size_t bind_time_offset;
        ret = cudaBindTexture(&bind_time_offset, texref, data, &desc, bytes);
        ASSERT_EQ(cudaSuccess, ret);
        bound = true;

        EXPECT_EQ(0, bind_time_offset);

        size_t offset;
        ret = cudaGetTextureAlignmentOffset(&offset, texref);
        ASSERT_EQ(cudaSuccess, ret);
        EXPECT_EQ(bind_time_offset, offset);
    }

    { // offset sizeof(*data)
        signed char * offset_data = data + 1;

        size_t bind_time_offset;
        ret = cudaBindTexture(&bind_time_offset, texref, offset_data, &desc,
            bytes);
        ASSERT_EQ(cudaSuccess, ret);
        bound = true;

        EXPECT_EQ((offset_data - data) * sizeof(*data), bind_time_offset);

        size_t offset;
        ret = cudaGetTextureAlignmentOffset(&offset, texref);
        ASSERT_EQ(cudaSuccess, ret);
        EXPECT_EQ(bind_time_offset, offset);
    }

    ret = cudaUnbindTexture(tex_src);
    bound = false;
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(GetTextureAlignmentOffset, UnboundTexture) {
    cudaError_t ret;
    const struct textureReference * texref;

    #if CUDA_VERSION < 5000
    ret = cudaGetTextureReference(&texref, "tex_src");
    #else
    ret = cudaGetTextureReference(&texref, &tex_src);
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    if (bound) {
        ret = cudaUnbindTexture(tex_src);
        ASSERT_EQ(cudaSuccess, ret);
    }

    size_t offset;
    ret = cudaGetTextureAlignmentOffset(&offset, texref);
    ASSERT_EQ(cudaErrorInvalidTextureBinding, ret);

    ret = cudaGetTextureAlignmentOffset(NULL, texref);
    ASSERT_EQ(cudaErrorInvalidValue, ret);
}

TEST(GetTextureAlignmentOffset, NullArguments) {
    cudaError_t ret;
    const struct textureReference * texref;

    const uint32_t bytes = 1u << 20;
    int32_t * data;
    ret = cudaMalloc((void **) &data, sizeof(*data) * bytes);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDA_VERSION < 5000
    ret = cudaGetTextureReference(&texref, "tex_src");
    #else
    ret = cudaGetTextureReference(&texref, &tex_src);
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT * sizeof(*data);
    desc.y = desc.z = desc.w = 0;

    ret = cudaBindTexture(NULL, texref, data, &desc, bytes);
    bound = true;
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaGetTextureAlignmentOffset(NULL, texref);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    size_t offset;
    ret = cudaGetTextureAlignmentOffset(&offset, NULL);
    ASSERT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaGetTextureAlignmentOffset(NULL, NULL);
    ASSERT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaUnbindTexture(tex_src);
    bound = false;
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
