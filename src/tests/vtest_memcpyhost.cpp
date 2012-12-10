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
#include <stdint.h>

TEST(MemcpyHost, PingPong) {
    /**
     * Allocate a succession of small, adjacent host buffers and then
     * proceed to bounce data to and from the device.
     */
    const size_t base = 512;
    const size_t lg_max_size = 16;
    std::vector<uint8_t *> host_pointers;

    cudaError_t ret;
    size_t max_allocated = 0;
    for (size_t lg = 0; lg < lg_max_size; lg++) {
        const size_t size = base << lg;
        uint8_t * host_pointer;
        ret = cudaHostAlloc((void **) &host_pointer, size, cudaHostAllocMapped);
        if (ret != cudaSuccess) {
            break;
        }

        if (host_pointers.size() > 0) {
            uint8_t * last = host_pointers.back();
            assert(lg > 0);
            const size_t last_size = base << (lg - 1);
            if (last + last_size != host_pointer) {
                /* The allocations are discontinuous.  Bail. */
                ret = cudaFreeHost(host_pointer);
                ASSERT_EQ(cudaSuccess, ret);
                break;
            }
        }

        max_allocated = std::max(max_allocated, size);
        host_pointers.push_back(host_pointer);
    }

    ASSERT_LT(1u, host_pointers.size());

    /* Allocate a comparable device buffer. */
    uint8_t * device_pointer;
    ret = cudaMalloc((void **) &device_pointer, max_allocated);
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    for (size_t lg = 0; lg < lg_max_size; lg++) {
        const size_t size = base << lg;
        if (size > max_allocated) {
            break;
        }

        /* To Device. */
        ret = cudaMemcpy(device_pointer, host_pointers[lg], size,
            cudaMemcpyHostToDevice);
        EXPECT_EQ(cudaSuccess, ret);

        /* From Device. */
        ret = cudaMemcpy(host_pointers[lg], device_pointer, size,
            cudaMemcpyDeviceToHost);
        EXPECT_EQ(cudaSuccess, ret);

        /* To Device Async. */
        ret = cudaMemcpyAsync(device_pointer, host_pointers[lg], size,
            cudaMemcpyHostToDevice, stream);
        EXPECT_EQ(cudaSuccess, ret);

        /* From Device Async. */
        ret = cudaMemcpyAsync(host_pointers[lg], device_pointer, size,
            cudaMemcpyDeviceToHost, stream);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaStreamSynchronize(stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(device_pointer);
    ASSERT_EQ(cudaSuccess, ret);

    for (size_t i = 0; i < host_pointers.size(); i++) {
        ret = cudaFreeHost(host_pointers[i]);
        ASSERT_EQ(cudaSuccess, ret);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
