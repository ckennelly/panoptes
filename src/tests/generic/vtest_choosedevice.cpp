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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

TEST(ChooseDevice, Basic) {
    cudaError_t ret;

    int devices;
    ret = cudaGetDeviceCount(&devices);
    ASSERT_EQ(cudaSuccess, ret);

    int device;
    struct cudaDeviceProp prop = cudaDevicePropDontCare;

    ret = cudaChooseDevice(&device, &prop);
    if (devices == 0) {
        EXPECT_EQ(cudaErrorInvalidValue, ret);
        return;
    }

    ASSERT_EQ(cudaSuccess, ret);
    EXPECT_GE(0, device);
    EXPECT_LT(device, devices);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
