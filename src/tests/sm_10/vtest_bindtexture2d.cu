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

#include "fixture_bindtexture2d.h"
#include <gtest/gtest.h>
#include "scoped_allocations.h"

TEST_F(BindTexture2D, Simple) {
    cudaError_t ret;

    unsigned width  =     prop.textureAlignment;
    unsigned height = 2 * prop.textureAlignment;

    scoped_allocation<int32_t> data(width * height);
    unsigned pitch  = width * sizeof(*data);

    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int32_t>();
    size_t offset;
    ret = cudaBindTexture2D(&offset, texref, data, &desc, width, height, pitch);
    EXPECT_EQ(cudaSuccess, ret);
    EXPECT_EQ(0u, offset);

    ret = cudaUnbindTexture(texref);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST_F(BindTexture2D, Pitched) {
    cudaError_t ret;

    unsigned width  =     prop.textureAlignment + 1;
    unsigned height = 2 * prop.textureAlignment;

    scoped_pitch<int32_t> data(width, height);

    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int32_t>();
    size_t offset;
    ret = cudaBindTexture2D(&offset, texref, data, &desc, width, height,
        data.pitch());
    EXPECT_EQ(cudaSuccess, ret);
    EXPECT_EQ(0u, offset);

    ret = cudaUnbindTexture(texref);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST_F(BindTexture2D, Adjacent) {
    unsigned n = 1u << 10;
    while (n > 0) {
        scoped_allocation<int32_t> d0(n * prop.textureAlignment);
        scoped_allocation<int32_t> d1(n * prop.textureAlignment);
        unsigned width = prop.textureAlignment;
        unsigned pitch = width * sizeof(*d0);

        if (d1 < d0) {
            swap(d0, d1);
        }

        if (size_t(d1 - d0) != n * width) {
            n >>= 1;
            continue;
        }

        struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int32_t>();
        size_t offset;
        cudaError_t ret ;
        ret = cudaBindTexture2D(&offset, texref, d0, &desc, width, 2 * n,
            pitch);
        EXPECT_EQ(cudaSuccess, ret);
        EXPECT_EQ(0u, offset);

        ret = cudaUnbindTexture(texref);
        EXPECT_EQ(cudaSuccess, ret);
        return;
    }

    /* This should be unreachable. */
    ASSERT_TRUE(false);
}

TEST_F(BindTexture2D, DoubleBind) {
    cudaError_t ret;

    unsigned width  =     prop.textureAlignment;
    unsigned height = 2 * prop.textureAlignment;

    scoped_allocation<int32_t> data(width * height);
    unsigned pitch  = width * sizeof(*data);

    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int32_t>();
    size_t offset;
    ret = cudaBindTexture2D(&offset, texref, data, &desc, width, height, pitch);
    EXPECT_EQ(cudaSuccess, ret);
    EXPECT_EQ(0u, offset);

    ret = cudaBindTexture2D(&offset, texref, data, &desc, width, height, pitch);
    EXPECT_EQ(cudaSuccess, ret);
    EXPECT_EQ(0u, offset);

    ret = cudaUnbindTexture(texref);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST_F(BindTexture2D, FreeBeforeUnbind) {
    cudaError_t ret;

    unsigned width  =     prop.textureAlignment;
    unsigned height = 2 * prop.textureAlignment;

    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int32_t>();

    { /* scope of data */
        scoped_allocation<int32_t> data(width * height);
        unsigned pitch = width * sizeof(*data);

        size_t offset;
        ret = cudaBindTexture2D(&offset, texref, data, &desc, width, height,
            pitch);
        EXPECT_EQ(cudaSuccess, ret);
        EXPECT_EQ(0u, offset);
    }

    ret = cudaUnbindTexture(texref);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST_F(BindTexture2D, Overrun) {
    cudaError_t ret;

    unsigned width  = prop.textureAlignment;
    unsigned height = prop.textureAlignment;

    scoped_allocation<int32_t> data(width * height);
    unsigned pitch  = width * sizeof(*data);

    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int32_t>();
    size_t offset;
    ret = cudaBindTexture2D(&offset, texref, data, &desc, pitch, 2 * height,
        pitch);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
}

TEST_F(BindTexture2D, NullArguments) {
    cudaError_t ret;

    unsigned width  =     prop.textureAlignment;
    unsigned height = 2 * prop.textureAlignment;

    scoped_allocation<int32_t> data(width * height);
    unsigned pitch  = width * sizeof(*data);

    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int32_t>();
    size_t offset;
    ret = cudaBindTexture2D(&offset, texref, data, &desc, width, height, pitch);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaBindTexture2D(&offset, texref, NULL, &desc, width, height, pitch);
    EXPECT_EQ(cudaErrorUnknown, ret);

    ret = cudaBindTexture2D(&offset, NULL,   data, &desc, width, height, pitch);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaBindTexture2D(&offset, NULL,   NULL, &desc, width, height, pitch);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    EXPECT_EQ(0u, offset);

    ret = cudaBindTexture2D(NULL,    texref, data, &desc, width, height, pitch);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaBindTexture2D(NULL,    texref, NULL, &desc, width, height, pitch);
    EXPECT_EQ(cudaErrorUnknown, ret);

    ret = cudaBindTexture2D(NULL,    NULL,   data, &desc, width, height, pitch);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaBindTexture2D(NULL,    NULL,   NULL, &desc, width, height, pitch);
    EXPECT_EQ(cudaErrorInvalidTexture, ret);

    ret = cudaUnbindTexture(texref);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST_F(BindTexture2D, Offsets) {
    cudaError_t ret;

    unsigned width  = prop.textureAlignment;
    unsigned height = prop.textureAlignment;

    /* Overallocate by the textureAlignment. */
    scoped_allocation<int32_t> data(width * height + prop.textureAlignment);
    unsigned pitch  = width * sizeof(*data);

    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int32_t>();

    for (size_t stride = 0; stride <= prop.textureAlignment; stride += 4) {
        uintptr_t uptr = reinterpret_cast<uintptr_t>(data.get());
        uptr += stride;
        const size_t expected_offset = stride % prop.textureAlignment;

        size_t offset;
        ret = cudaBindTexture2D(&offset, texref,
            reinterpret_cast<const int32_t *>(uptr), &desc, width, height,
            pitch);
        EXPECT_EQ(cudaSuccess, ret);
        EXPECT_EQ(expected_offset, offset);

        ret = cudaUnbindTexture(texref);
        EXPECT_EQ(cudaSuccess, ret);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
