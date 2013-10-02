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

TEST(ChooseDevice, Nulls) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    int device;
    struct cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));

    EXPECT_EXIT(cudaChooseDevice(NULL, NULL),
        ::testing::KilledBySignal(SIGSEGV), "");

    EXPECT_EXIT(cudaChooseDevice(NULL, &prop),
        ::testing::KilledBySignal(SIGSEGV), "");

    EXPECT_EXIT(cudaChooseDevice(&device, NULL),
        ::testing::KilledBySignal(SIGSEGV), "");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
