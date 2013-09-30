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

TEST(SetDeviceFlags, Simple) {
    cudaError_t ret;

    ret = cudaSetDeviceFlags(cudaDeviceScheduleAuto);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(0);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    EXPECT_EQ(cudaErrorSetOnActiveProcess, ret);

    ret = cudaGetLastError();
    EXPECT_EQ(cudaErrorSetOnActiveProcess, ret);

    ret = cudaGetLastError();
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaSetDeviceFlags(cudaDeviceScheduleAuto);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(0);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    EXPECT_EQ(cudaErrorSetOnActiveProcess, ret);

    ret = cudaGetLastError();
    EXPECT_EQ(cudaErrorSetOnActiveProcess, ret);

    ret = cudaGetLastError();
    EXPECT_EQ(cudaSuccess, ret);
}

/**
 * Verify that cudaDeviceEnablePeerAccess initializes the contexts of the two
 * devices in question.
 */
TEST(SetDeviceFlags, PeerToPeer) {
    cudaError_t ret;
    int devices;

    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);

    if (devices <= 1) {
        return;
    }

    for (int i = 0; i < devices; i++) {
        ret = cudaSetDevice(i);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaDeviceReset();
        ASSERT_EQ(cudaSuccess, ret);
    }

    bool found = false;
    int di, dj;

    for (int i = 0; i < devices && !(found); i++) {
        for (int j = 0; j < devices; j++) {
            if (j == i) {
                continue;
            }

            int peer;
            ret = cudaDeviceCanAccessPeer(&peer, i, j);
            ASSERT_EQ(cudaSuccess, ret);

            if (peer) {
                ret = cudaSetDevice(i);
                ASSERT_EQ(cudaSuccess, ret);

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

    ret = cudaSetDevice(dj);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    EXPECT_EQ(cudaErrorSetOnActiveProcess, ret);

    ret = cudaSetDevice(di);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaDeviceDisablePeerAccess(dj);
    EXPECT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
