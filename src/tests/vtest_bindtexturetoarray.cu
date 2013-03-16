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
#include <stdint.h>

texture<int32_t, 1, cudaReadModeElementType> tex_src;

TEST(BindTextureToArray, Simple) {
    cudaError_t ret;
    const struct textureReference * texref;
    cudaArray * array;

    const uint32_t ints = 1u << 16;

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = 32;
    desc.y = desc.z = desc.w = 0;

    ret = cudaMallocArray(&array, &desc, ints, 0, 0);
    ASSERT_EQ(cudaSuccess, ret);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    const void * ptr;
    if (version < 5000 /* 5.0 */) {
        ptr = "tex_src";
    } else {
        ptr = &tex_src;
    }
    ret = cudaGetTextureReference(&texref, ptr);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaBindTextureToArray(texref, array, &desc);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFreeArray(array);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(BindTextureToArray, FreeBeforeUnbind) {
    cudaError_t ret;
    const struct textureReference * texref;
    cudaArray * array;

    const uint32_t ints = 1u << 16;

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = 32;
    desc.y = desc.z = desc.w = 0;

    ret = cudaMallocArray(&array, &desc, ints, 0, 0);
    ASSERT_EQ(cudaSuccess, ret);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    const void * ptr;
    if (version < 5000 /* 5.0 */) {
        ptr = "tex_src";
    } else {
        ptr = &tex_src;
    }
    ret = cudaGetTextureReference(&texref, ptr);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaBindTextureToArray(texref, array, &desc);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFreeArray(array);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(BindTextureToArray, NullArguments) {
    cudaError_t ret;
    const struct textureReference * texref;
    cudaArray * array;

    const uint32_t ints = 1u << 16;

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = 32;
    desc.y = desc.z = desc.w = 0;

    ret = cudaMallocArray(&array, &desc, ints, 0, 0);
    ASSERT_EQ(cudaSuccess, ret);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    const void * ptr;
    if (version < 5000 /* 5.0 */) {
        ptr = "tex_src";
    } else {
        ptr = &tex_src;
    }
    ret = cudaGetTextureReference(&texref, ptr);
    ASSERT_EQ(cudaSuccess, ret);

/* SIGSEGV
    ret = cudaBindTextureToArray(texref, array, NULL);
    EXPECT_EQ(cudaSuccess, ret);
    */

    ret = cudaBindTextureToArray(texref, NULL,  &desc);
    EXPECT_EQ(cudaErrorInvalidResourceHandle, ret);
/* SIGSEGV
    ret = cudaBindTextureToArray(texref, NULL,  NULL);
    EXPECT_EQ(cudaSuccess, ret);
*/
    ret = cudaBindTextureToArray(NULL,   array, &desc);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaBindTextureToArray(NULL,   array, NULL);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaBindTextureToArray(NULL,   NULL,  &desc);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaBindTextureToArray(NULL,   NULL,  NULL);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    /* We never bound anything successfully, so no need for
       cudaUnbindTexture. */

    ret = cudaFreeArray(array);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
