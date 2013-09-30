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

#include <algorithm>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>
#include <vector>

class SharedOffsets : public ::testing::Test {
public:
    SharedOffsets() { }
    ~SharedOffsets() { }

    void SetUp() {
        cudaError_t ret;
        ret = cudaStreamCreate(&stream);
        EXPECT_EQ(cudaSuccess, ret);

        threads = 256;
        ret = cudaMalloc((void **) &d, threads * sizeof(*d));
        EXPECT_EQ(cudaSuccess, ret);

        reset = false;
    }

    void TearDown() {
        cudaError_t ret;

        if (reset) {
            ret = cudaDeviceReset();
            EXPECT_EQ(cudaSuccess, ret);
        } else {
            ret = cudaStreamDestroy(stream);
            EXPECT_EQ(cudaSuccess, ret);

            ret = cudaFree(d);
            EXPECT_EQ(cudaSuccess, ret);
        }
    }

    cudaStream_t stream;
    uint32_t     threads;
    uint32_t   * d;
    bool         reset;
};

static __global__ void k_known_symbol(uint32_t * d, uint32_t base) {
    uint32_t out;
    /**
     * __shared__ uint32_t u;
     * u = a;
     * __syncthreads();
     * out = atomicInc(u, 0xFFFFFFFF);
     */
    asm volatile(
        "{ .shared .align 4 .u32 u;\n"
        "st.shared.u32 [u], %1;\n"
        "bar.sync 0;\n"
        "atom.shared.inc.u32 %0, [u], -1;\n}" : "=r"(out) : "r"(base));
    d[threadIdx.x] = out;
}

TEST_F(SharedOffsets, KnownSymbol) {
    cudaError_t ret;

    const uint32_t base = 256;
    k_known_symbol<<<1, threads, 0, stream>>>(d, base);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    std::vector<uint32_t> hd(threads);
    ret = cudaMemcpy(hd.data(), d, sizeof(*d) * threads,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    std::sort(hd.begin(), hd.end());
    for (uint32_t i = 0; i < threads; i++) {
        EXPECT_EQ(i + base, hd[i]);
    }
}

static __global__ void k_known_symbol_suffix(uint32_t * d, uint32_t base) {
    uint32_t out;
    /**
     * __shared__ uint32_t u<2>;
     * u1 = a;
     * __syncthreads();
     * out = atomicInc(u1, 0xFFFFFFFF);
     */
    asm volatile(
        "{ .shared .align 4 .u32 u<2>;\n"
        "st.shared.u32 [u1], %1;\n"
        "bar.sync 0;\n"
        "atom.shared.inc.u32 %0, [u1], -1;\n}" : "=r"(out) : "r"(base));
    d[threadIdx.x] = out;
}

TEST_F(SharedOffsets, KnownSuffixedSymbol) {
    cudaError_t ret;

    const uint32_t base = 256;
    k_known_symbol_suffix<<<1, threads, 0, stream>>>(d, base);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    std::vector<uint32_t> hd(threads);
    ret = cudaMemcpy(hd.data(), d, sizeof(*d) * threads,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    std::sort(hd.begin(), hd.end());
    for (uint32_t i = 0; i < threads; i++) {
        EXPECT_EQ(i + base, hd[i]);
    }
}

static __global__ void k_known_symbol_offsets(uint32_t * d, uint32_t base) {
    uint32_t out;
    /**
     * __shared__ uint32_t u;
     * u = a;
     * __syncthreads();
     * out = atomicInc(u, 0xFFFFFFFF);
     */

    #if CUDA_VERSION == 5000
    /*
     * Work around a bug in ptxas.  Without this, ptxas fails with:
     *  "Internal error: overlapping offsets allocated to objects"
     */
    __shared__ uint32_t uf[1];
    uf[0] = base;
    *d = uf[0];
    #endif

    asm volatile(
        "{ .shared .align 4 .u32 u[2];\n"
        "st.shared.u32 [u+4], %1;\n"
        "bar.sync 0;\n"
        "atom.shared.inc.u32 %0, [u+4], -1;\n}" : "=r"(out) : "r"(base));
    d[threadIdx.x] = out;
}

TEST_F(SharedOffsets, KnownSymbolOffsets) {
    cudaError_t ret;

    const uint32_t base = 256;
    k_known_symbol_offsets<<<1, threads, 0, stream>>>(d, base);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    std::vector<uint32_t> hd(threads);
    ret = cudaMemcpy(hd.data(), d, sizeof(*d) * threads,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    std::sort(hd.begin(), hd.end());
    for (uint32_t i = 0; i < threads; i++) {
        EXPECT_EQ(i + base, hd[i]);
    }
}

/**
 * a is the final "result" of the address.  Since we are storing a value into
 * a loaded from an invalid address [u+4], it should be uninitialized.
 */
static __global__ void k_known_symbol_overrun(uint32_t * d, uint32_t * a,
        uint32_t base) {
    uint32_t out;
    uint32_t aout;
    /**
     * __shared__ uint32_t u;
     * u = a;
     * __syncthreads();
     * out = atomicInc(u, 0xFFFFFFFF);
     */
    asm volatile(
        "{ .shared .align 4 .u32 u[1];\n"
        "st.shared.u32 [u+4], %2;\n"
        "bar.sync 0;\n"
        "atom.shared.inc.u32 %0, [u+4], -1;\n"
        "bar.sync 0;\n"
        "ld.shared.u32 %1, [u+4];\n}" : "=r"(out), "=r"(aout) : "r"(base));
    d[threadIdx.x] = out;
    *a = aout;
}

TEST_F(SharedOffsets, StaticOverrun) {
    cudaError_t ret;

    uint32_t * a;
    ret = cudaMalloc((void **) &a, sizeof(*a));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMemset(a, 0, sizeof(*a));
    ASSERT_EQ(cudaSuccess, ret);

    const uint32_t base = 256;
    k_known_symbol_overrun<<<1, threads, 0, stream>>>(d, a, base);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * The final state of d should be undefined.  Since Panoptes drops the st
     * and atom instructions against the address, the buffer should be
     * uninitialized.
     */
    std::vector<uint32_t> hd(threads);
    ret = cudaMemcpy(hd.data(), d, sizeof(*d) * threads,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    std::vector<uint32_t> vd(threads);
    int vret = VALGRIND_GET_VBITS(hd.data(), vd.data(), sizeof(*d) * threads);
    if (vret == 1) {
        for (uint32_t i = 0; i < threads; i++) {
            EXPECT_EQ(0xFFFFFFFF, vd[i]);
        }

        /**
         * Verify that we did not perform the ld.shared.u32 instruction.
         */
        uint32_t a_final;
        ret = cudaMemcpy(&a_final, a, sizeof(*a), cudaMemcpyDeviceToHost);
        uint32_t va_final;
        vret = VALGRIND_GET_VBITS(&a_final, &va_final, sizeof(a_final));
        ASSERT_EQ(1, vret);

        EXPECT_EQ(0xFFFFFFFF, va_final);
    }

    ret = cudaFree(a);
    ASSERT_EQ(cudaSuccess, ret);
}

static __global__ void k_known_symbol_flexible(uint32_t * d, uint32_t base) {
    extern __shared__ uint32_t uf[];
    uf[0] = base;
    __syncthreads();
    d[threadIdx.x] = atomicInc(uf, 0xFFFFFFFF);
}

TEST_F(SharedOffsets, FlexibleSymbol) {
    cudaError_t ret;

    const uint32_t base = 256;
    k_known_symbol_flexible
        <<<1, threads, 4, stream>>>(d, base);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    std::vector<uint32_t> hd(threads);
    ret = cudaMemcpy(hd.data(), d, sizeof(*d) * threads,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    std::sort(hd.begin(), hd.end());
    for (uint32_t i = 0; i < threads; i++) {
        EXPECT_EQ(i + base, hd[i]);
    }

    k_known_symbol_flexible<<<1, threads, 0, stream>>>(d, base);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    reset = true;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
