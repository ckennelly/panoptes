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
#include <cstdio>
#include <vector>

TEST(PeerAccess, CanAccess) {
    cudaError_t ret;

    int devices;
    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);

    for (int i = 0; i < devices; i++) {
        for (int j = 0; j < devices; j++) {
            int peer;
            ret = cudaDeviceCanAccessPeer(&peer, i, j);
            EXPECT_EQ(cudaSuccess, ret);

            if (i == j) {
                EXPECT_FALSE(peer);
            }
        }
    }
}

TEST(PeerAccess, CanAccessInvalidDevice) {
    cudaError_t ret;

    int devices;
    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);

    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    int peer;
    ret = cudaDeviceCanAccessPeer(&peer, device, devices);
    EXPECT_EQ(cudaErrorInvalidDevice, ret);
}

/**
 * We need to verify that device allocations on separate devices that would
 * be eligible for peer to peer do not appear on the same allocation chunk
 * kept by Panoptes.  We test a stronger case:  Devices with unified addressing
 * (not just peer to peer accessability) should have this property.
 */
TEST(Malloc, AddressSpacing) {
    cudaError_t ret;

    int devices;
    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);
    if (devices <= 1) {
        return;
    }

    const size_t chunk_size = 1u << 16;

    typedef std::pair<int, void *> alloc_t;
    std::vector<alloc_t> allocations;
    for (int i = 0; i < devices; i++) {
        cudaDeviceProp prop;
        ret = cudaGetDeviceProperties(&prop, i);
        ASSERT_EQ(cudaSuccess, ret);

        /**
         * Skip the device if it does not support unified addressing.
         */
        if (!(prop.unifiedAddressing)) {
            continue;
        }

        ret = cudaSetDevice(i);
        ASSERT_EQ(cudaSuccess, ret);

        void * ptr;
        ret = cudaMalloc(&ptr, 1);
        ASSERT_EQ(cudaSuccess, ret);

        allocations.push_back(alloc_t(i, ptr));
    }

    const size_t n_allocations = allocations.size();

    /* Check upper bits of allocated vectors. */
    const size_t mask = ~(chunk_size - 1u);
    for (size_t i = 0; i < n_allocations; i++) {
        const uintptr_t ip =
            reinterpret_cast<uintptr_t>(allocations[i].second) & mask;

        for (size_t j = i + 1; j < n_allocations; j++) {
            const uintptr_t jp =
                reinterpret_cast<uintptr_t>(allocations[j].second) & mask;
            EXPECT_NE(ip, jp);
        }
    }

    /* Cleanup. */
    for (size_t i = 0; i < n_allocations; i++) {
        ret = cudaSetDevice(allocations[i].first);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaFree(allocations[i].second);
        ASSERT_EQ(cudaSuccess, ret);
    }
}

TEST(PeerAccess, EnableDisable) {
    cudaError_t ret;

    int devices;
    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);

    if (devices <= 1) {
        return;
    }

    typedef std::pair<int, int> peer_t;
    std::vector<peer_t> peers;

    for (int i = 0; i < devices; i++) {
        ret = cudaSetDevice(i);
        ASSERT_EQ(cudaSuccess, ret);

        for (int j = 0; j < devices; j++) {
            int peer;
            ret = cudaDeviceCanAccessPeer(&peer, i, j);
            ASSERT_EQ(cudaSuccess, ret);

            cudaError_t expected;

            if (peer) {
                expected = cudaSuccess;
                peers.push_back(peer_t(i, j));
            } else {
                expected = cudaErrorInvalidDevice;
            }

            ret = cudaDeviceEnablePeerAccess(j, 0);
            EXPECT_EQ(expected, ret);
        }
    }

    /* Cleanup. */
    const size_t n_peers = peers.size();

    for (size_t i = 0; i < n_peers; i++) {
        ret = cudaSetDevice(peers[i].first);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaDeviceDisablePeerAccess(peers[i].second);
        EXPECT_EQ(cudaSuccess, ret);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
