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
#include <sys/mman.h>
#include <valgrind/memcheck.h>

static void __global__ k_set(uintptr_t * p, uintptr_t a) {
    *p = a;
}

static void __global__ k_noop() { }

TEST(ConfigureCall, OrphanedConfigureCall) {
    dim3 grid, block;
    grid.x  = grid.y  = grid.z  = 1;
    block.x = block.y = block.z = 1;

    cudaError_t ret;

    cudaStream_t cs;
    ret = cudaStreamCreate(&cs);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaConfigureCall(grid, block, 0, cs);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(ConfigureCall, ExcessiveBlockSize) {
    /**
     * Get the device information, trying various block sizes at the limits and
     * beyond those specified in cudaDeviceProp.
     */
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t cs;
    ret = cudaStreamCreate(&cs);
    ASSERT_EQ(cudaSuccess, ret);

    dim3 grid, block;
    grid.x = grid.y = grid.z = 1;

    { // Max X
        block.x = prop.maxThreadsDim[0];
        block.y = 1;
        block.z = 1;

        const int threads = block.x * block.y * block.z;
        const cudaError_t exp = threads <= prop.maxThreadsPerBlock ?
            cudaSuccess : cudaErrorInvalidConfiguration;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(exp, ret);
        cudaGetLastError();
    }

    { // Max Y
        block.x = 1;
        block.y = prop.maxThreadsDim[1];
        block.z = 1;

        const int threads = block.x * block.y * block.z;
        const cudaError_t exp = threads <= prop.maxThreadsPerBlock ?
            cudaSuccess : cudaErrorInvalidConfiguration;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(exp, ret);
        cudaGetLastError();
    }

    { // Max X, Y
        block.x = prop.maxThreadsDim[0];
        block.y = prop.maxThreadsDim[1];
        block.z = 1;

        const int threads = block.x * block.y * block.z;
        const cudaError_t exp = threads <= prop.maxThreadsPerBlock ?
            cudaSuccess : cudaErrorInvalidConfiguration;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(exp, ret);
        cudaGetLastError();
    }

    { // Max Z
        block.x = 1;
        block.y = 1;
        block.z = prop.maxThreadsDim[2];

        const int threads = block.x * block.y * block.z;
        const cudaError_t exp = threads <= prop.maxThreadsPerBlock ?
            cudaSuccess : cudaErrorInvalidConfiguration;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(exp, ret);
        cudaGetLastError();
    }

    { // Max X, Z
        block.x = prop.maxThreadsDim[0];
        block.y = 1;
        block.z = prop.maxThreadsDim[2];

        const int threads = block.x * block.y * block.z;
        const cudaError_t exp = threads <= prop.maxThreadsPerBlock ?
            cudaSuccess : cudaErrorInvalidConfiguration;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(exp, ret);
        cudaGetLastError();
    }

    { // Max Y, Z
        block.x = 1;
        block.y = prop.maxThreadsDim[1];
        block.z = prop.maxThreadsDim[2];

        const int threads = block.x * block.y * block.z;
        const cudaError_t exp = threads <= prop.maxThreadsPerBlock ?
            cudaSuccess : cudaErrorInvalidConfiguration;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(exp, ret);
        cudaGetLastError();
    }

    { // Max X, Y, Z
        block.x = prop.maxThreadsDim[0];
        block.y = prop.maxThreadsDim[1];
        block.z = prop.maxThreadsDim[2];

        const int threads = block.x * block.y * block.z;
        const cudaError_t exp = threads <= prop.maxThreadsPerBlock ?
            cudaSuccess : cudaErrorInvalidConfiguration;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(exp, ret);
        cudaGetLastError();
    }

    { // Max X + 1
        block.x = prop.maxThreadsDim[0] + 1;
        block.y = 1;
        block.z = 1;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(cudaErrorInvalidConfiguration, ret);
        cudaGetLastError();
    }

    { // Max Y + 1
        block.x = 1;
        block.y = prop.maxThreadsDim[1] + 1;
        block.z = 1;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(cudaErrorInvalidConfiguration, ret);
        cudaGetLastError();
    }

    { // Max Z + 1
        block.x = 1;
        block.y = 1;
        block.z = prop.maxThreadsDim[2] + 1;

        ret = cudaConfigureCall(grid, block, 0, cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaLaunch(k_noop);
        EXPECT_EQ(cudaErrorInvalidConfiguration, ret);
        cudaGetLastError();
    }

    ret = cudaStreamSynchronize(cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(cs);
    EXPECT_EQ(cudaSuccess, ret);
}

TEST(ConfigureCall, ExcessiveSharedMemory) {
    dim3 grid, block;
    grid.x  = grid.y  = grid.z  = 1;
    block.x = block.y = block.z = 1;

    cudaError_t ret;

    if (sizeof(size_t) != sizeof(uint32_t)) {
        cudaStream_t cs;
        ret = cudaStreamCreate(&cs);
        ASSERT_EQ(cudaSuccess, ret);

        /* 2^32 + 2^0 */
        {
            const size_t limits = (size_t) 4294967297ULL;
            ret = cudaConfigureCall(grid, block, limits, cs);
            EXPECT_EQ(cudaSuccess, ret);

            ret = cudaLaunch(k_noop);
            EXPECT_EQ(cudaSuccess, ret);
        }

        /* 2^32 + 2^16 + 2^0 */
        {
            const size_t limits = (size_t) 4295032833ULL;
            ret = cudaConfigureCall(grid, block, limits, cs);
            EXPECT_EQ(cudaSuccess, ret);

            ret = cudaLaunch(k_noop);
            EXPECT_EQ(cudaErrorInvalidValue, ret);
        }

        /* 2^32 + 2^17 + 2^0 */
        {
            const size_t limits = (size_t) 4295098369ULL;
            ret = cudaConfigureCall(grid, block, limits, cs);
            EXPECT_EQ(cudaSuccess, ret);

            ret = cudaLaunch(k_noop);
            EXPECT_EQ(cudaErrorInvalidValue, ret);
        }

        /* 2^32 + 2^20 + 2^0 */
        {
            const size_t limits = (size_t) 4296015873ULL;
            ret = cudaConfigureCall(grid, block, limits, cs);
            EXPECT_EQ(cudaSuccess, ret);

            ret = cudaLaunch(k_noop);
            EXPECT_EQ(cudaErrorInvalidValue, ret);
        }

        /* 2^32 + 2^25 + 2^0 */
        {
            const size_t limits = (size_t) 4328521729ULL;
            ret = cudaConfigureCall(grid, block, limits, cs);
            EXPECT_EQ(cudaSuccess, ret);

            ret = cudaLaunch(k_noop);
            EXPECT_EQ(cudaErrorInvalidValue, ret);
        }

        /* 2^32 + 2^30 + 2^0 */
        {
            const size_t limits = (size_t) 5368709121ULL;
            ret = cudaConfigureCall(grid, block, limits, cs);
            EXPECT_EQ(cudaSuccess, ret);

            ret = cudaLaunch(k_noop);
            EXPECT_EQ(cudaErrorInvalidValue, ret);
        }

        /* 2^32 + 2^31 + 2^0 */
        {
            const size_t limits = (size_t) 6442450945ULL;
            ret = cudaConfigureCall(grid, block, limits, cs);
            EXPECT_EQ(cudaSuccess, ret);

            ret = cudaLaunch(k_noop);
            EXPECT_EQ(cudaErrorInvalidValue, ret);
        }

        ret = cudaStreamSynchronize(cs);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaStreamDestroy(cs);
        EXPECT_EQ(cudaSuccess, ret);
    }
}

TEST(SetupArgument, Simple) {
    dim3 grid, block;
    grid.x  = grid.y  = grid.z  = 1;
    block.x = block.y = block.z = 1;

    cudaError_t ret;
    cudaStream_t cs;
    ret = cudaStreamCreate(&cs);
    ASSERT_EQ(cudaSuccess, ret);

    uintptr_t * p;
    uintptr_t in = 5;
    ret = cudaMalloc((void **) &p, sizeof(*p));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaConfigureCall(grid, block, 0u, cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaSetupArgument(&p, sizeof(p), 0);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaSetupArgument(&in, sizeof(in), sizeof(p));
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaLaunch(k_set);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamSynchronize(cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(cs);
    EXPECT_EQ(cudaSuccess, ret);

    uintptr_t out;
    BOOST_STATIC_ASSERT(sizeof(out) == sizeof(*p));
    ret = cudaMemcpy(&out, p, sizeof(out), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(p);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(in, out);
}

static void __global__ k_consume(uint8_t data[]) {

}

TEST(SetupArgument, LargeArguments) {
    dim3 grid, block;
    grid.x  = grid.y  = grid.z  = 1;
    block.x = block.y = block.z = 1;

    cudaError_t ret;
    cudaStream_t cs;
    ret = cudaStreamCreate(&cs);
    ASSERT_EQ(cudaSuccess, ret);

    uint8_t large[1u << 13];

    ret = cudaConfigureCall(grid, block, 0u, cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaSetupArgument(large, sizeof(large), 0);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaLaunch(k_consume);
    EXPECT_EQ(cudaErrorInvalidValue, ret);

    ret = cudaStreamSynchronize(cs);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(cs);
    EXPECT_EQ(cudaSuccess, ret);
}

class SetupArgumentFixture : public ::testing::Test {
        // Empty Fixture
public:
    virtual void SetUp() {
        const long page_size_ = sysconf(_SC_PAGESIZE);
        ASSERT_LT(0, page_size_);

        page_size = page_size_;

        buffer = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
        EXPECT_FALSE(buffer == (void *) -1);
    }

    virtual void TearDown() {
        (void) munmap(buffer, page_size);
    }

    void clear() {
        memset(buffer, 0, page_size);
        (void) VALGRIND_MAKE_MEM_UNDEFINED(buffer, page_size);
    }

    void protect() {
        int ret = mprotect(buffer, page_size, PROT_NONE);
        EXPECT_EQ(0, ret);
    }

    void unprotect() {
        int ret = mprotect(buffer, page_size, PROT_READ | PROT_WRITE);
        EXPECT_EQ(0, ret);
    }

    size_t page_size;
    void * buffer;
};

TEST_F(SetupArgumentFixture, WhenCopied) {
    dim3 grid, block;
    grid.x  = grid.y  = grid.z  = 1;
    block.x = block.y = block.z = 1;

    cudaError_t ret;
    cudaStream_t cs;
    ret = cudaStreamCreate(&cs);
    ASSERT_EQ(cudaSuccess, ret);

    uintptr_t * p;
    uintptr_t in = 5;
    ret = cudaMalloc((void **) &p, sizeof(*p));
    ASSERT_EQ(cudaSuccess, ret);

    ASSERT_NE(in, (uintptr_t) p);

    ret = cudaConfigureCall(grid, block, 0u, cs);
    EXPECT_EQ(cudaSuccess, ret);

    clear();
    *reinterpret_cast<uintptr_t **>(buffer) = p;

    ret = cudaSetupArgument(buffer, sizeof(p), 0);
    EXPECT_EQ(cudaSuccess, ret);

    clear();
    *reinterpret_cast<uintptr_t *>(buffer) = in;

    ret = cudaSetupArgument(buffer, sizeof(in), sizeof(p));
    EXPECT_EQ(cudaSuccess, ret);

    clear();

    /**
     * Amongst other ways this test will produce a failure if CUDA is not
     * making full memory copies of the arguments passed in by
     * cudaSetupArgument, we'll SIGSEGV here when it tries to read the
     * pointers.
     */
    protect();

    ret = cudaLaunch(k_set);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamSynchronize(cs);
    EXPECT_EQ(cudaSuccess, ret);

    unprotect();

    ret = cudaStreamDestroy(cs);
    EXPECT_EQ(cudaSuccess, ret);

    uintptr_t out;
    BOOST_STATIC_ASSERT(sizeof(out) == sizeof(*p));
    ret = cudaMemcpy(&out, p, sizeof(out), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(p);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * Since p != in (per early test), this will fail if we used the same
     * memory region multiple times without copying its value.
     */
    EXPECT_EQ(in, out);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
