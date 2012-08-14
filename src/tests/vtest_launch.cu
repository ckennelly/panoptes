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

/**
 * Specify C-linkage to avoid name mangling to assist with writing the
 * LaunchByName test.
 */
extern "C" void __global__ k_noop() { }

TEST(Launch, LaunchWithoutConfigure) {
    /**
     * Call cudaLaunch without a proceeding call to cudaConfigureCall and
     * watch what happens.
     */
    cudaError_t ret;
    ret = cudaLaunch(k_noop);

    /**
     * cudaErrorMissingConfiguration seems like an ideal error here, but
     * cudaErrorInvalidConfiguration is returned instead.
     */
    EXPECT_EQ(cudaErrorInvalidConfiguration, ret);
}

TEST(Launch, LaunchByName) {
    /**
     * In the documentation for cudaLaunch in cuda_runtime_api.h:
     *   "The entry paramater may also be a character string naming a device
     *    function to execute, however this usage is deprecated as of CUDA
     *    4.1."
     *
     * This test performs a launch by name to verify program behavior.
     */
    dim3 grid, block;
    grid.x  = grid.y  = grid.z  = 1;
    block.x = block.y = block.z = 1;

    cudaError_t ret;
    cudaStream_t cs;
    ret = cudaStreamCreate(&cs);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaConfigureCall(grid, block, 0u, cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaLaunch("k_noop");
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamSynchronize(cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(cs);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(Launch, LaunchNull) {
    /**
     * See how cudaLaunch reacts to NULL pointers.
     */
    dim3 grid, block;
    grid.x  = grid.y  = grid.z  = 1;
    block.x = block.y = block.z = 1;

    cudaError_t ret;
    cudaStream_t cs;
    ret = cudaStreamCreate(&cs);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaConfigureCall(grid, block, 0u, cs);
    EXPECT_EQ(cudaSuccess, ret);

    /**
     * For CUDA 4.2, cudaErrorInvalidDeviceFunction is returned.  For CUDA 4.1
     * and older, cudaErrorUnknown is returned.
     */
    int runtimeVersion;
    ret = cudaRuntimeGetVersion(&runtimeVersion);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaLaunch(NULL);
    if (runtimeVersion >= 4020) {
        EXPECT_EQ(cudaErrorInvalidDeviceFunction, ret);
    } else {
        EXPECT_EQ(cudaErrorUnknown, ret);
    }

    ret = cudaStreamSynchronize(cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(cs);
    EXPECT_EQ(cudaSuccess, ret);
}

/**
 * We would also like to test cudaLaunch with an arbitrary host pointer, but
 * when the function is not found by its device function address, CUDA (and
 * Panoptes) interprets it as an entry name and stores it into an std::string.
 * If our host pointer was not meant to be interpreted readily as a C-style
 * string, Valgrind will find errors, such as:
 *
 * ==11347== Conditional jump or move depends on uninitialised value(s)
 * ==11347==    at 0x4C2A849: strlen (mc_replace_strmem.c:390)
 * ==11347==    by 0x5DC9F6F: std::basic_string<char, std::char_traits<char>,
 *  std::allocator<char> >::basic_string(char const*,
 *  std::allocator<char> const&)
 *  (in /usr/lib64/gcc/x86_64-pc-linux-gnu/4.5.3/libstdc++.so.6.0.14)
 * ==11347==    by 0x5AEF750: ??? (in /opt/cuda/lib64/libcudart.so.4.1.28)
 * ==11347==    by 0x5AEF826: ??? (in /opt/cuda/lib64/libcudart.so.4.1.28)
 * ==11347==    by 0x5ADD522: ??? (in /opt/cuda/lib64/libcudart.so.4.1.28)
 * ==11347==    by 0x5B05899: cudaLaunch (in /opt/cuda/lib64/libcudart.so.4.1.28)
 */

TEST(Launch, LaunchSimple) {
    /**
     * Verify we can still launch kernels.
     */
    dim3 grid, block;
    grid.x  = grid.y  = grid.z  = 1;
    block.x = block.y = block.z = 1;

    cudaError_t ret;
    cudaStream_t cs;
    ret = cudaStreamCreate(&cs);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaConfigureCall(grid, block, 0u, cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaLaunch(k_noop);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamSynchronize(cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(cs);
    EXPECT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
