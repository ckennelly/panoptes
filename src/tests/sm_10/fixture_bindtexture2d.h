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

#ifndef __PANOPTES__TESTS__SM_10__FIXTURE_BINDTEXTURE2D_H__
#define __PANOPTES__TESTS__SM_10__FIXTURE_BINDTEXTURE2D_H__

#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>

texture<int32_t, 2, cudaReadModeElementType> tex_src;

class BindTexture2D : public ::testing::Test {
public:
    struct cudaDeviceProp prop;
    int version;

    const struct textureReference * texref;

    void SetUp() {
        cudaError_t ret;

        int device;
        ret = cudaGetDevice(&device);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaGetDeviceProperties(&prop, device);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaRuntimeGetVersion(&version);
        ASSERT_EQ(cudaSuccess, ret);

        #if CUDA_VERSION >= 5000
        if (version < 5000 /* 5.0 */) {
        #endif
            ASSERT_FALSE(true);
            ret = cudaGetTextureReference(&texref, "tex_src");
        #if CUDA_VERSION >= 5000
        } else {
            ret = cudaGetTextureReference(&texref, &tex_src);
        }
        #endif
        ASSERT_EQ(cudaSuccess, ret);

        tex_src.addressMode[0] = cudaAddressModeClamp;
        tex_src.addressMode[1] = cudaAddressModeClamp;
        tex_src.filterMode     = cudaFilterModeLinear;
        tex_src.normalized     = true;
    }

    void TearDown() {

    }
};

#endif // __PANOPTES__TESTS__SM_10__FIXTURE_BINDTEXTURE2D_H__
