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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>
#include <valgrind/memcheck.h>

static __global__ void k_p2p(int * out, const int * in) {
    out[0] = in[0];
    out[1] = in[1];
}

/**
 * This test verifies several aspects of peer-to-peer transfers:
 *
 *   -> A kernel can read from one device and write to another without
 *      catastrophic failure.
 *   -> During the read/write operation, the actual data and validity bits
 *      are transfered as well.
 */
TEST(PeerAccess, SimpleKernel) {
    cudaError_t ret;

    int devices;
    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);

    if (devices <= 1) {
        return;
    }

    bool found = false;
    int di, dj;

    for (int i = 0; i < devices && !(found); i++) {
        ret = cudaSetDevice(i);
        ASSERT_EQ(cudaSuccess, ret);

        for (int j = 0; j < devices; j++) {
            int peer;
            ret = cudaDeviceCanAccessPeer(&peer, i, j);
            ASSERT_EQ(cudaSuccess, ret);

            if (peer) {
                ret = cudaDeviceEnablePeerAccess(j, 0);
                ASSERT_EQ(cudaSuccess, ret);

                di = i;
                dj = j;
                found = true;
                break;
            }
        }
    }

    if (!(found)) {
        return;
    }

    int *mi, *mj;
    ret = cudaSetDevice(di);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &mi, 2 * sizeof(*mi));
    ASSERT_EQ(cudaSuccess, ret);

    const int expected = 5;
    ret = cudaMemcpy(mi, &expected, sizeof(expected), cudaMemcpyHostToDevice);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaSetDevice(dj);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &mj, 2 * sizeof(*mj));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaSetDevice(di);
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_p2p<<<1, 1, 0, stream>>>(mj, mi);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaSetDevice(dj);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceSynchronize();

    int host[2];
    ret = cudaMemcpy(host, mj, sizeof(host), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(expected, host[0]);
    if (RUNNING_ON_VALGRIND) {
        uint32_t vbits;

        BOOST_STATIC_ASSERT(sizeof(host[1]) == sizeof(vbits));
        const int vret = VALGRIND_GET_VBITS(&host[1], &vbits, sizeof(vbits));
        assert(vret == 0 || vret == 1);
        if (vret == 1) {
            EXPECT_EQ(0xFFFFFFFF, vbits);
        }
    }

    ret = cudaSetDevice(di);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(mi);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaDeviceDisablePeerAccess(dj);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaSetDevice(dj);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(mj);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
