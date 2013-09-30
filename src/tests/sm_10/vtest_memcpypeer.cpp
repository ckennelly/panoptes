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
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

TEST(MemcpyPeer, InvalidDevices) {
    cudaError_t ret;
    int devices;

    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyPeer(NULL, devices, NULL, devices, 0);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyPeer(NULL, devices, NULL, devices, 4);
    EXPECT_EQ(cudaErrorInvalidDevice, ret);
}

TEST(MemcpyPeer, OutOfBounds) {
    cudaError_t ret;
    int devices;

    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);

    if (devices == 0) {
        return;
    }

    ret = cudaMemcpyPeer(NULL, 0, NULL, 0, 0);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyPeer(NULL, 0, NULL, 0, 4);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaSetDevice(0);
    ASSERT_EQ(cudaSuccess, ret);

    const size_t size = 1 << 20;
    void * d0;
    ret = cudaMalloc(&d0, size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyPeer(d0, 0, d0, 0, 2 * size);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    if (devices <= 1) {
        ret = cudaFree(d0);
        EXPECT_EQ(cudaSuccess, ret);

        return;
    }

    ret = cudaMemcpyPeer(NULL, 0, NULL, 1, 0);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyPeer(NULL, 0, NULL, 1, 4);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaSetDevice(1);
    ASSERT_EQ(cudaSuccess, ret);

    void * d1;
    ret = cudaMalloc(&d1, size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyPeer(d0, 0, d1, 1, 2 * size);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaFree(d1);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaSetDevice(0);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(d0);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(MemcpyPeer, Basic) {
    cudaError_t ret;
    int devices;

    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);

    if (devices == 0) {
        return;
    }

    ret = cudaSetDevice(0);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t * d0;
    const size_t size = 2 * sizeof(*d0);
    ret = cudaMalloc((void **) &d0, 2 * size);
    ASSERT_EQ(cudaSuccess, ret);

    const uint32_t expected = 0xDEADBEEF;
    ret = cudaMemcpy(d0, &expected, sizeof(expected), cudaMemcpyHostToDevice);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyPeer(d0 + 2, 0, d0, 0, size);
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t host[2];
    ret = cudaMemcpy(host, d0 + 2, sizeof(host), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t vhost[2];
    int vret = VALGRIND_GET_VBITS(host, vhost, sizeof(host));
    if (vret == 1) {
        EXPECT_EQ(0, vhost[0]);
        VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&host[0], sizeof(host[0]));
        EXPECT_EQ(0xFFFFFFFF, vhost[1]);
    }

    EXPECT_EQ(expected, host[0]);

    if (devices <= 1) {
        ret = cudaFree(d0);
        EXPECT_EQ(cudaSuccess, ret);

        return;
    }

    ret = cudaSetDevice(1);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t * d1;
    ret = cudaMalloc((void **) &d1, size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemcpyPeer(d1, 1, d0, 0, size);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(host, d1, sizeof(host), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    vret = VALGRIND_GET_VBITS(host, vhost, sizeof(host));
    if (vret == 1) {
        EXPECT_EQ(0, vhost[0]);
        VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&host[0], sizeof(host[0]));
        EXPECT_EQ(0xFFFFFFFF, vhost[1]);
    }

    EXPECT_EQ(expected, host[0]);

    ret = cudaFree(d1);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaSetDevice(0);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(d0);
    EXPECT_EQ(cudaSuccess, ret);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
