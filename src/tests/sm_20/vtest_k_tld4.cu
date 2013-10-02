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
#include <valgrind/memcheck.h>

texture<int4, 2, cudaReadModeElementType> tex_src;

__global__ void k_tld(int4 * dst, float x, float y) {
    /**
     * In the absence a device function using a texture through a recognizable
     * channel, nvcc version 4.0 places the texref for "tex_src" below all of
     * the device functions.  This leads the associated ptxas to look for
     * (missing) forward declarations.  To remedy this, we perform a dummy
     * texture lookup to force the compiler to place the global texture
     * declaration in the right place.
     *
     * This is not an issue for nvcc 4.1 or newer, but the version is not readily
     * obtained from standard preprocessor macros (nvcc -E -dryrun /dev/null).
     */
    (void) tex2D(tex_src, 0, 0);

    int4 tmp;

    asm volatile("tld4.r.2d.v4.s32.f32 {%0, %1, %2, %3}, [tex_src, {%4, %5}];"
        : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "f"(x), "f"(y));
    dst[0] = tmp;

    asm volatile("tld4.g.2d.v4.s32.f32 {%0, %1, %2, %3}, [tex_src, {%4, %5}];"
        : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "f"(x), "f"(y));
    dst[1] = tmp;

    asm volatile("tld4.b.2d.v4.s32.f32 {%0, %1, %2, %3}, [tex_src, {%4, %5}];"
        : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "f"(x), "f"(y));
    dst[2] = tmp;

    asm volatile("tld4.a.2d.v4.s32.f32 {%0, %1, %2, %3}, [tex_src, {%4, %5}];"
        : "=r"(tmp.x), "=r"(tmp.y), "=r"(tmp.z), "=r"(tmp.w) : "f"(x), "f"(y));
    dst[3] = tmp;
}

__global__ void k_set(int4 *dst, unsigned width, unsigned height, int z,
        int w) {
    const unsigned bound = width * height;

    for (unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
            idx < bound; idx += blockDim.x * gridDim.x) {
        int4 tmp;
        tmp.x = idx % width;
        tmp.y = idx / width;
        tmp.z = z;
        tmp.w = w;
        dst[idx] = tmp;
    }
}

TEST(kTLD, ExplicitStream) {
    cudaError_t ret;
    cudaStream_t stream;

    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    #if CUDART_VERSION >= 4010 /* 4.1 */
    const uint32_t width  = prop.texturePitchAlignment;
    #else
    const uint32_t width  = 512;
    #endif
    const uint32_t height = 2048;
    int4 *src;
    int4 hdst[4];
    int4 *dst;

    ret = cudaMalloc(&src, sizeof(*src) * width * height);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemset(src, 0, sizeof(*src) * width * height);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc(&dst, sizeof(hdst));
    ASSERT_EQ(cudaSuccess, ret);

    const textureReference* texref;
    #if CUDA_VERSION < 5000
    ret = cudaGetTextureReference(&texref, "tex_src");
    #else
    ret = cudaGetTextureReference(&texref, &tex_src);
    #endif
    ASSERT_EQ(cudaSuccess, ret);

    tex_src.addressMode[0] = cudaAddressModeClamp;
    tex_src.addressMode[1] = cudaAddressModeClamp;
    tex_src.filterMode     = cudaFilterModeLinear;
    tex_src.normalized     = true;

    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc<int4>();
    size_t offset;

    /* Initial pass with valid z and w parameters. */
    const float x = 0.5f, y = 0.5f;

    const int out_x = width  * x;
    const int out_y = height * y;

    int z = 5;
    int w = 3;
    k_set<<<1, 1, 0, stream>>>(src, width, height, z, w);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaBindTexture2D(&offset, texref, src, &desc, width, height,
        width * sizeof(*src));
    ASSERT_EQ(cudaSuccess, ret);
    EXPECT_EQ(0u, offset);

    k_tld<<<1, 1, 0, stream>>>(dst, x, y);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaUnbindTexture(texref);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(hdst, dst, sizeof(hdst), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(out_x, hdst[0].x);
    EXPECT_EQ(out_y, hdst[0].y);
    EXPECT_EQ(5,     hdst[0].z);
    EXPECT_EQ(3,     hdst[0].w);
    EXPECT_EQ(out_x, hdst[1].x);
    EXPECT_EQ(out_y, hdst[1].y);
    EXPECT_EQ(5,     hdst[1].z);
    EXPECT_EQ(3,     hdst[1].w);
    EXPECT_EQ(out_x, hdst[2].x);
    EXPECT_EQ(out_y, hdst[2].y);
    EXPECT_EQ(5,     hdst[2].z);
    EXPECT_EQ(3,     hdst[2].w);
    EXPECT_EQ(out_x, hdst[3].x);
    EXPECT_EQ(out_y, hdst[3].y);
    EXPECT_EQ(5,     hdst[3].z);
    EXPECT_EQ(3,     hdst[3].w);

    if (RUNNING_ON_VALGRIND) {
        /* Mark z and w as invalid. */
        z = 7;
        w = 4;
        VALGRIND_MAKE_MEM_UNDEFINED(&z, sizeof(z));
        VALGRIND_MAKE_MEM_UNDEFINED(&w, sizeof(w));
        k_set<<<1, 1, 0, stream>>>(src, width, height, z, w);

        ret = cudaStreamSynchronize(stream);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaBindTexture2D(&offset, texref, src, &desc, width, height,
            width * sizeof(*src));
        ASSERT_EQ(cudaSuccess, ret);
        EXPECT_EQ(0u, offset);

        k_tld<<<1, 1, 0, stream>>>(dst, x, y);

        ret = cudaStreamSynchronize(stream);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaUnbindTexture(texref);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaMemcpy(hdst, dst, sizeof(hdst), cudaMemcpyDeviceToHost);
        ASSERT_EQ(cudaSuccess, ret);

        /* x and y dimensions should still be the same. */
        EXPECT_EQ(out_x, hdst[0].x);
        EXPECT_EQ(out_y, hdst[0].y);
        EXPECT_EQ(out_x, hdst[1].x);
        EXPECT_EQ(out_y, hdst[1].y);
        EXPECT_EQ(out_x, hdst[2].x);
        EXPECT_EQ(out_y, hdst[2].y);
        EXPECT_EQ(out_x, hdst[3].x);
        EXPECT_EQ(out_y, hdst[3].y);

        /* z and w should be uninitialized. */
        uint4 vdst[4];
        VALGRIND_GET_VBITS(hdst, vdst, sizeof(hdst));
        EXPECT_EQ(0xFFFFFFFF, vdst[0].z);
        EXPECT_EQ(0xFFFFFFFF, vdst[0].w);
        EXPECT_EQ(0xFFFFFFFF, vdst[1].z);
        EXPECT_EQ(0xFFFFFFFF, vdst[1].w);
        EXPECT_EQ(0xFFFFFFFF, vdst[2].z);
        EXPECT_EQ(0xFFFFFFFF, vdst[2].w);
        EXPECT_EQ(0xFFFFFFFF, vdst[3].z);
        EXPECT_EQ(0xFFFFFFFF, vdst[3].w);

        /* See that the sentinel changed. */
        VALGRIND_MAKE_MEM_DEFINED(hdst, sizeof(hdst));
        EXPECT_EQ(z, hdst[0].z);
        EXPECT_EQ(w, hdst[0].w);
        EXPECT_EQ(z, hdst[1].z);
        EXPECT_EQ(w, hdst[1].w);
        EXPECT_EQ(z, hdst[2].z);
        EXPECT_EQ(w, hdst[2].w);
        EXPECT_EQ(z, hdst[3].z);
        EXPECT_EQ(w, hdst[3].w);
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
