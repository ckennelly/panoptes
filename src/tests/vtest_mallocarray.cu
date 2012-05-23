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

TEST(MallocArray, NullArguments) {
    struct cudaArray * ary;
    struct cudaChannelFormatDesc dsc;
    dsc.x = dsc.y = dsc.z = dsc.w = 8;
    dsc.f = cudaChannelFormatKindSigned;

    // Commented out cases segfault.

    EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(NULL, NULL, 0, 0, 0));
    EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(NULL, NULL, 0, 8, 0));
    // EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(NULL, NULL, 8, 0, 0), 0));
    // EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(NULL, NULL, 8, 8, 0));

    EXPECT_EQ(cudaSuccess,           cudaMallocArray(&ary, NULL, 0, 0, 0));
    EXPECT_EQ(cudaSuccess,           cudaFreeArray(ary));

    EXPECT_EQ(cudaSuccess,           cudaMallocArray(&ary, NULL, 0, 8, 0));
    EXPECT_EQ(cudaSuccess,           cudaFreeArray(ary));

    // EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(&ary, NULL, 8, 0, 0));

    // EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(&ary, NULL, 8, 8, 0));

    EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(NULL, &dsc, 0, 0, 0));
    EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(NULL, &dsc, 0, 8, 0));
    // EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(NULL, &dsc, 8, 0, 0));
    // EXPECT_EQ(cudaErrorInvalidValue, cudaMallocArray(NULL, &dsc, 8, 8, 0));
}

TEST(MallocArray, Limits) {
    struct cudaArray * ary;
    struct cudaChannelFormatDesc dsc;
    dsc.x = dsc.y = dsc.z = dsc.w = 8;
    dsc.f = cudaChannelFormatKindSigned;

    cudaError_t ret;

    ret = cudaMallocArray(&ary, &dsc, 0, 0, 0);
    EXPECT_EQ(cudaSuccess, ret);
    if (ret == cudaSuccess) {
        EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
    }

    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    /* Adapt to what's available by a safe margin */
    size_t targetable = prop.totalGlobalMem / 8;

    if ((size_t) prop.maxTexture1D < targetable) {
        ret = cudaMallocArray(&ary, &dsc, prop.maxTexture1D, 0, 0);
        EXPECT_EQ(cudaSuccess, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }

        ret = cudaMallocArray(&ary, &dsc, prop.maxTexture1D + 1, 0, 0);
        EXPECT_EQ(cudaErrorInvalidValue, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }
    }

    if ((size_t) prop.maxTexture2D[0] < targetable) {
        ret = cudaMallocArray(&ary, &dsc, prop.maxTexture2D[0], 1, 0);
        EXPECT_EQ(cudaSuccess, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }

        ret = cudaMallocArray(&ary, &dsc, prop.maxTexture2D[0] + 1, 1, 0);
        EXPECT_EQ(cudaErrorInvalidValue, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }
    }

    if ((size_t) prop.maxTexture2D[1] < targetable) {
        ret = cudaMallocArray(&ary, &dsc, 1, prop.maxTexture2D[1], 0);
        EXPECT_EQ(cudaSuccess, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }

        ret = cudaMallocArray(&ary, &dsc, 1, prop.maxTexture2D[1] + 1, 0);
        EXPECT_EQ(cudaErrorInvalidValue, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }
    }

    if ((size_t) prop.maxTexture2D[0] * prop.maxTexture2D[1] < targetable) {
        ret = cudaMallocArray(&ary, &dsc, prop.maxTexture2D[0],
                prop.maxTexture2D[1], 0);
        EXPECT_EQ(cudaSuccess, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }

        ret = cudaMallocArray(&ary, &dsc, prop.maxTexture2D[0],
                prop.maxTexture2D[1] + 1, 0);
        EXPECT_EQ(cudaErrorInvalidValue, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }

        ret = cudaMallocArray(&ary, &dsc, prop.maxTexture2D[0] + 1,
                prop.maxTexture2D[1], 0);
        EXPECT_EQ(cudaErrorInvalidValue, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }

        ret = cudaMallocArray(&ary, &dsc, prop.maxTexture2D[0] + 1,
                prop.maxTexture2D[1] + 1, 0);
        EXPECT_EQ(cudaErrorInvalidValue, ret);
        if (ret == cudaSuccess) {
            EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
        }
    } else if ((size_t) prop.maxTexture2D[0] * prop.maxTexture2D[1] >
            prop.totalGlobalMem) {
        EXPECT_EQ(cudaErrorMemoryAllocation,
            cudaMallocArray(&ary, &dsc, prop.maxTexture2D[0],
            prop.maxTexture2D[1], 0));
    }
}

TEST(MallocArray, Attributes) {
    struct cudaArray * ary;
    struct cudaChannelFormatDesc dsc;
    dsc.x = dsc.y = dsc.z = dsc.w = 8;
    dsc.f = cudaChannelFormatKindSigned;

    cudaError_t ret;

    ret = cudaMallocArray(&ary, &dsc, 1, 1, 0);
    ASSERT_EQ(cudaSuccess, ret);

    struct cudaPointerAttributes attr;
    ret = cudaPointerGetAttributes(&attr, ary);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    EXPECT_EQ(cudaSuccess, cudaFreeArray(ary));
}

TEST(MallocArray, NegativeChannels) {
    struct cudaArray * ary;
    struct cudaChannelFormatDesc dsc;
    dsc.x = dsc.y = dsc.z = 8;
    dsc.w = -8;
    dsc.f = cudaChannelFormatKindSigned;

    cudaError_t ret;

    ret = cudaMallocArray(&ary, &dsc, 1, 1, 0);
    ASSERT_EQ(cudaErrorInvalidChannelDescriptor, ret);
}

TEST(MallocArray, Mismatch) {
    struct cudaArray * ary;
    struct cudaChannelFormatDesc dsc;
    dsc.x = dsc.y = dsc.z = dsc.w = 8;
    dsc.f = cudaChannelFormatKindSigned;

    cudaError_t ret;
    ret = cudaMallocArray(&ary, &dsc, 1, 1, 0);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(ary);
    EXPECT_EQ(cudaErrorInvalidDevicePointer, ret);

    ret = cudaFreeArray(ary);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
