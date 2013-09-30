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

static void not_a_device_function(void) { }

TEST(FuncSetCacheConfig, NotADeviceFunction) {
    cudaError_t ret;

    ret = cudaFuncSetCacheConfig((char *) not_a_device_function,
        cudaFuncCachePreferNone);
    ASSERT_EQ(cudaErrorInvalidDeviceFunction, ret);

    ret = cudaFuncSetCacheConfig((char *) not_a_device_function,
        cudaFuncCachePreferShared);
    ASSERT_EQ(cudaErrorInvalidDeviceFunction, ret);

    ret = cudaFuncSetCacheConfig((char *) not_a_device_function,
        cudaFuncCachePreferL1);
    ASSERT_EQ(cudaErrorInvalidDeviceFunction, ret);

    ret = cudaFuncSetCacheConfig((char *) not_a_device_function,
        cudaFuncCachePreferEqual);
    ASSERT_EQ(cudaErrorInvalidDeviceFunction, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
