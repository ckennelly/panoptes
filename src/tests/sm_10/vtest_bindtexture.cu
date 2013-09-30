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

TEST(BindTexture, Simple) {
    cudaError_t ret;
    const struct textureReference * texref;

    const uint32_t bytes = 1u << 20;
    int32_t * data;
    ret = cudaMalloc((void **) &data, sizeof(*data) * bytes);
    ASSERT_EQ(cudaSuccess, ret);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDA_VERSION >= 5000
    if (version < 5000 /* 5.0 */) {
    #endif
        ret = cudaGetTextureReference(&texref, "tex_src");
    #if CUDA_VERSION >= 5000
    } else {
        ret = cudaGetTextureReference(&texref, &tex_src);
    }
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT * sizeof(*data);
    desc.y = desc.z = desc.w = 0;
    ret = cudaBindTexture(NULL, texref, data, &desc, bytes);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(BindTexture, Adjacent) {
    cudaError_t ret;
    const struct textureReference * texref;

    uint32_t bytes = 1u << 20;
    uint8_t * data[2];

    while (bytes > 0) {
        ret = cudaMalloc((void **) &data[0], bytes);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaMalloc((void **) &data[1], bytes);
        ASSERT_EQ(cudaSuccess, ret);

        if (data[1] < data[0]) {
            std::swap(data[0], data[1]);
        }

        if ((size_t) (data[1] - data[0]) == bytes) {
            break;
        } else {
            ret = cudaFree(data[0]);
            ASSERT_EQ(cudaSuccess, ret);

            ret = cudaFree(data[1]);
            ASSERT_EQ(cudaSuccess, ret);

            bytes = bytes >> 1u;
        }
    }

    if (bytes == 0) {
        return;
    }

    ASSERT_LT(0, bytes);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDA_VERSION >= 5000
    if (version < 5000 /* 5.0 */) {
    #endif
        ret = cudaGetTextureReference(&texref, "tex_src");
    #if CUDA_VERSION >= 5000
    } else {
        ret = cudaGetTextureReference(&texref, &tex_src);
    }
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT;
    desc.y = desc.z = desc.w = 0;
    ret = cudaBindTexture(NULL, texref, data[0], &desc, 2u * bytes);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data[0]);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(data[1]);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(BindTexture, DoubleBind) {
    cudaError_t ret;
    const struct textureReference * texref;

    const uint32_t bytes = 1u << 20;
    int32_t * data;
    ret = cudaMalloc((void **) &data, sizeof(*data) * bytes);
    ASSERT_EQ(cudaSuccess, ret);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDA_VERSION >= 5000
    if (version < 5000 /* 5.0 */) {
    #endif
        ret = cudaGetTextureReference(&texref, "tex_src");
    #if CUDA_VERSION >= 5000
    } else {
        ret = cudaGetTextureReference(&texref, &tex_src);
    }
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT * sizeof(*data);
    desc.y = desc.z = desc.w = 0;

    ret = cudaBindTexture(NULL, texref, data, &desc, bytes);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaBindTexture(NULL, texref, data, &desc, bytes);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(BindTexture, FreeBeforeUnbind) {
    cudaError_t ret;
    const struct textureReference * texref;

    const uint32_t bytes = 1u << 20;
    int32_t * data;
    ret = cudaMalloc((void **) &data, sizeof(*data) * bytes);
    ASSERT_EQ(cudaSuccess, ret);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDA_VERSION >= 5000
    if (version < 5000 /* 5.0 */) {
    #endif
        ret = cudaGetTextureReference(&texref, "tex_src");
    #if CUDA_VERSION >= 5000
    } else {
        ret = cudaGetTextureReference(&texref, &tex_src);
    }
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT * sizeof(*data);
    desc.y = desc.z = desc.w = 0;
    ret = cudaBindTexture(NULL, texref, data, &desc, bytes);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(BindTexture, Overrun) {
    cudaError_t ret;
    const struct textureReference * texref;

    const uint32_t bytes = 1u << 20;
    int32_t * data;
    ret = cudaMalloc((void **) &data, sizeof(*data) * bytes);
    ASSERT_EQ(cudaSuccess, ret);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDA_VERSION >= 5000
    if (version < 5000 /* 5.0 */) {
    #endif
        ret = cudaGetTextureReference(&texref, "tex_src");
    #if CUDA_VERSION >= 5000
    } else {
        ret = cudaGetTextureReference(&texref, &tex_src);
    }
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT * sizeof(*data);
    desc.y = desc.z = desc.w = 0;
    ret = cudaBindTexture(NULL, texref, data, &desc, 2 * bytes * sizeof(*data));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(BindTexture, NullArguments) {
    cudaError_t ret;
    const struct textureReference * texref;

    const uint32_t bytes = 1u << 20;
    int32_t * data;
    ret = cudaMalloc((void **) &data, sizeof(*data) * bytes);
    ASSERT_EQ(cudaSuccess, ret);

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDA_VERSION >= 5000
    if (version < 5000 /* 5.0 */) {
    #endif
        ret = cudaGetTextureReference(&texref, "tex_src");
    #if CUDA_VERSION >= 5000
    } else {
        ret = cudaGetTextureReference(&texref, &tex_src);
    }
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT * sizeof(*data);
    desc.y = desc.z = desc.w = 0;

    ret = cudaBindTexture(NULL, texref, NULL, &desc, bytes);
    EXPECT_EQ(cudaErrorUnknown, ret);

    ret = cudaBindTexture(NULL, texref, NULL, NULL,  bytes);
    EXPECT_EQ(cudaErrorUnknown, ret);

    ret = cudaBindTexture(NULL, NULL,   data, &desc, bytes);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaBindTexture(NULL, NULL,   data, NULL,  bytes);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaBindTexture(NULL, NULL,   NULL, &desc, bytes);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaBindTexture(NULL, NULL,   NULL, NULL,  bytes);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(BindTexture, Offsets) {
    cudaError_t ret;
    const struct textureReference * texref;

    const uint32_t bytes = 1u << 20;
    int32_t * data;
    ret = cudaMalloc((void **) &data, sizeof(*data) * bytes);
    ASSERT_EQ(cudaSuccess, ret);

    int32_t * offset_data = data + 1;

    int version;
    ret = cudaRuntimeGetVersion(&version);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDA_VERSION >= 5000
    if (version < 5000 /* 5.0 */) {
    #endif
        ret = cudaGetTextureReference(&texref, "tex_src");
    #if CUDA_VERSION >= 5000
    } else {
        ret = cudaGetTextureReference(&texref, &tex_src);
    }
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaChannelFormatDesc desc;
    desc.f = cudaChannelFormatKindSigned;
    desc.x = CHAR_BIT * sizeof(*data);
    desc.y = desc.z = desc.w = 0;

    ret = cudaBindTexture(NULL, texref, offset_data, &desc, bytes);
    ASSERT_EQ(cudaErrorInvalidValue, ret);

    size_t offset;
    ret = cudaBindTexture(&offset, texref, offset_data, &desc, bytes);
    ASSERT_EQ(cudaSuccess, ret);
    EXPECT_EQ((offset_data - data) * sizeof(*data), offset);

    ret = cudaUnbindTexture(tex_src);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(data);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
