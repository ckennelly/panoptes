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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

TEST(EventCreateWithFlags, NullArgument) {
    cudaError_t ret;

    ret = cudaEventCreateWithFlags(NULL, 0);
    EXPECT_EQ(cudaErrorInvalidValue, ret);
}

TEST(EventCreateWithFlags, CreateDestroy) {
    cudaError_t ret;
    cudaEvent_t event;

    ret = cudaEventCreateWithFlags(&event, 0);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventDestroy(event);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(EventCreateWithFlags, EventLeak) {
    cudaError_t ret;
    cudaEvent_t event;

    ret = cudaEventCreateWithFlags(&event, 0);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}