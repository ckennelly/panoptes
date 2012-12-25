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
#include <gtest/gtest.h>

enum {
    local_size = 128
};

static __device__ void prefetch_generic_l1(void * ptr) {
    asm volatile("prefetch.L1 [%0];" : : "l"(ptr));
}

static __global__ void prefetch_global_l1(void * ptr) {
    asm volatile("prefetch.global.L1 [%0];" : : "l"(ptr));
}

static __device__ void prefetch_local_l1(void * ptr) {
    asm volatile("prefetch.local.L1 [%0];" : : "l"(ptr));
}

static __device__ void prefetch_generic_l2(void * ptr) {
    asm volatile("prefetch.L2 [%0];" : : "l"(ptr));
}

static __global__ void prefetch_global_l2(void * ptr) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
}

static __device__ void prefetch_local_l2(void * ptr) {
    asm volatile("prefetch.local.L2 [%0];" : : "l"(ptr));
}

static __device__ void prefetchu(void * ptr) {
    asm volatile("prefetchu.L1 [%0];" : : "l"(ptr));
}

static __global__ void k_prefetchu(void * ptr) {
    prefetchu(ptr);
}

static __global__ void k_global_generic_l1(void * ptr) {
    prefetch_generic_l1(ptr);
}

static __global__ void k_global_generic_l2(void * ptr) {
    prefetch_generic_l2(ptr);
}

static __global__ void k_local_generic_l1(unsigned offset) {
    char a[local_size];
    prefetch_generic_l1(&a[offset]);
}

static __global__ void k_local_generic_l2(unsigned offset) {
    char a[local_size];
    prefetch_generic_l2(&a[offset]);
}

static __global__ void k_local_local_l1(unsigned offset) {
    char a[local_size];
    prefetch_local_l1(&a[offset]);
}

static __global__ void k_local_local_l2(unsigned offset) {
    char a[local_size];
    prefetch_local_l2(&a[offset]);
}

static __global__ void k_prefetchu_local(unsigned offset) {
    char a[local_size];
    prefetchu(&a[offset]);
}

TEST(Prefetch, GlobalInBounds) {
    cudaError_t ret;
    cudaStream_t stream;

    const size_t size = 128;
    char * tmp;
    ret = cudaMalloc((void **) &tmp, size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* In bounds. */
    prefetch_global_l1 <<<1, 1, 0, stream>>>(tmp);
    prefetch_global_l2 <<<1, 1, 0, stream>>>(tmp);
    k_global_generic_l1<<<1, 1, 0, stream>>>(tmp);
    k_global_generic_l2<<<1, 1, 0, stream>>>(tmp);
    k_prefetchu        <<<1, 1, 0, stream>>>(tmp);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(tmp);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Prefetch, GlobalEdgeOfBounds) {
    cudaError_t ret;
    cudaStream_t stream;

    const size_t size = 128;
    char * tmp;
    ret = cudaMalloc((void **) &tmp, size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Edge of bounds. */
    prefetch_global_l1 <<<1, 1, 0, stream>>>(tmp + size);
    prefetch_global_l2 <<<1, 1, 0, stream>>>(tmp + size);
    k_global_generic_l1<<<1, 1, 0, stream>>>(tmp + size);
    k_global_generic_l2<<<1, 1, 0, stream>>>(tmp + size);
    k_prefetchu        <<<1, 1, 0, stream>>>(tmp + size);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Obtain a lowerbound for the global slack. */
    prefetch_global_l1 <<<1, 1, 0, stream>>>(tmp + 1024 - sizeof(void*));
    prefetch_global_l2 <<<1, 1, 0, stream>>>(tmp + 1024 - sizeof(void*));
    k_global_generic_l1<<<1, 1, 0, stream>>>(tmp + 1024 - sizeof(void*));
    k_global_generic_l2<<<1, 1, 0, stream>>>(tmp + 1024 - sizeof(void*));
    k_prefetchu        <<<1, 1, 0, stream>>>(tmp + 1024 - sizeof(void*));

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(tmp);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Prefetch, GlobalMisaligned) {
    cudaError_t ret;
    cudaStream_t stream;

    const size_t size = 128;
    char * tmp;
    ret = cudaMalloc((void **) &tmp, size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Misaligned successes */
    prefetch_global_l1 <<<1, 1, 0, stream>>>(tmp + 1u);
    prefetch_global_l2 <<<1, 1, 0, stream>>>(tmp + 1u);
    k_prefetchu        <<<1, 1, 0, stream>>>(tmp + 1u);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Misaligned failures. */
    k_global_generic_l1<<<1, 1, 0, stream>>>(tmp + 1u);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &tmp, size);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_global_generic_l2<<<1, 1, 0, stream>>>(tmp + 1u);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Prefetch, GlobalOutOfBounds) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Out of bounds. */
    prefetch_global_l1 <<<1, 1, 0, stream>>>(NULL);
    prefetch_global_l2 <<<1, 1, 0, stream>>>(NULL);

    prefetch_global_l1 <<<1, 1, 0, stream>>>((void *) 0x100000);
    prefetch_global_l2 <<<1, 1, 0, stream>>>((void *) 0x100000);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_global_generic_l1<<<1, 1, 0, stream>>>(NULL);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_global_generic_l2<<<1, 1, 0, stream>>>(NULL);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_global_generic_l1<<<1, 1, 0, stream>>>((void *) 0xFFFFFF00);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_global_generic_l2<<<1, 1, 0, stream>>>((void *) 0x100000);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_prefetchu       <<<1, 1, 0, stream>>>(NULL);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_prefetchu       <<<1, 1, 0, stream>>>((void *) 0x100000);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Prefetch, LocalInBounds) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_local_generic_l1<<<1, 1, 0, stream>>>(0);
    k_local_generic_l2<<<1, 1, 0, stream>>>(0);
    k_local_local_l1  <<<1, 1, 0, stream>>>(0);
    k_local_local_l2  <<<1, 1, 0, stream>>>(0);
    k_prefetchu_local <<<1, 1, 0, stream>>>(0);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Prefetch, LocalEdge) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* The edge of bounds. */
    k_local_generic_l1<<<1, 1, 0, stream>>>(local_size);
    k_local_generic_l2<<<1, 1, 0, stream>>>(local_size);
    k_local_local_l1  <<<1, 1, 0, stream>>>(local_size);
    k_local_local_l2  <<<1, 1, 0, stream>>>(local_size);
    k_prefetchu_local <<<1, 1, 0, stream>>>(local_size);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Prefetch, LocalMisaligned) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Misaligned successes */
    k_local_local_l1  <<<1, 1, 0, stream>>>(1);
    k_local_local_l2  <<<1, 1, 0, stream>>>(1);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Misaligned failures. */
    k_local_generic_l1<<<1, 1, 0, stream>>>(1);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_local_generic_l2<<<1, 1, 0, stream>>>(1);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_prefetchu_local <<<1, 1, 0, stream>>>(1);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Prefetch, LocalOutOfBounds) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Test going out of bounds. */
    const unsigned oob = 1u << 11;
    BOOST_STATIC_ASSERT(oob > local_size);

    k_local_generic_l1<<<1, 1, 0, stream>>>(oob);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_local_generic_l2<<<1, 1, 0, stream>>>(oob);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_local_local_l1  <<<1, 1, 0, stream>>>(oob);
    k_local_local_l2  <<<1, 1, 0, stream>>>(oob);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_prefetchu_local <<<1, 1, 0, stream>>>(oob);
    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    ASSERT_EQ(cudaSuccess, ret);
}

static __global__ void k_predicated(void * ptr, const unsigned flag) {
    asm volatile("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %0, 0;\n"
        "@%temp prefetch.L1 [%1];\n}" : : "r"(flag), "l"(ptr));
}

static __global__ void k_predicatedu(void * ptr, const unsigned flag) {
    asm volatile("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %0, 0;\n"
        "@%temp prefetchu.L1 [%1];\n}" : : "r"(flag), "l"(ptr));
}

static __global__ void k_negated(void * ptr, const unsigned flag) {
    asm volatile("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %0, 0;\n"
        "@!%temp prefetch.L1 [%1];\n}" : : "r"(flag), "l"(ptr));
}

static __global__ void k_negatedu(void * ptr, const unsigned flag) {
    asm volatile("{ .reg .pred %temp;\n"
        "setp.ne.u32 %temp, %0, 0;\n"
        "@!%temp prefetchu.L1 [%1];\n}" : : "r"(flag), "l"(ptr));
}

TEST(Prefetch, Predicated) {
    cudaError_t ret;

    const size_t size = 128;
    void * tmp;
    ret = cudaMalloc(&tmp, size);
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    unsigned flag;
    flag = 1;
    k_predicated <<<1, 1, 0, stream>>>(tmp, flag);
    k_predicatedu<<<1, 1, 0, stream>>>(tmp, flag);

    flag = 0;
    k_negated    <<<1, 1, 0, stream>>>(tmp, flag);
    k_negatedu   <<<1, 1, 0, stream>>>(tmp, flag);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(tmp);
    ASSERT_EQ(cudaSuccess, ret);
}

static __global__ void k_local_1(void ** out) {
    void * ptr;
    asm volatile(".local .align 1 .b8 l[1]; mov.b64 %0, l;" : "=l"(ptr));
    *out = ptr;
}

static __global__ void k_local_8(void ** out) {
    void * ptr;
    asm volatile(".local .align 1 .b8 l[8]; mov.b64 %0, l;" : "=l"(ptr));
    *out = ptr;
}

static __global__ void k_local_256(void ** out) {
    void * ptr;
    asm volatile(".local .align 1 .b8 l[256]; mov.b64 %0, l;" : "=l"(ptr));
    *out = ptr;
}

static __global__ void k_local_2048(void ** out) {
    void * ptr;
    asm volatile(".local .align 1 .b8 l[2048]; mov.b64 %0, l;" : "=l"(ptr));
    *out = ptr;
}

/**
 * Validate the upper bound of local addresses used by Panoptes.
 */
TEST(Local, UpperBound) {
    cudaError_t ret;
    cudaStream_t stream;
    cudaDeviceReset();

    const size_t sizes[] = {1, 8, 256, 2048};

    void ** ptrs;
    ret = cudaMalloc(&ptrs, 4 * sizeof(ptrs));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_local_1   <<<1, 1, 0, stream>>>(ptrs + 0);
    k_local_8   <<<1, 1, 0, stream>>>(ptrs + 1);
    k_local_256 <<<1, 1, 0, stream>>>(ptrs + 2);
    k_local_2048<<<1, 1, 0, stream>>>(ptrs + 3);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    char * hptrs[4];
    ret = cudaMemcpy(hptrs, ptrs, sizeof(hptrs), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(ptrs);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * We merely check for this being an upperbound.  Due to Panoptes'
     * instrumentation of local storage, our allocations may not be the highest
     * in the address space.
     */
    const void * upper_bound = reinterpret_cast<void *>(0xFFFCA0);
    EXPECT_GE(upper_bound, hptrs[0] + sizes[0]);
    EXPECT_GE(upper_bound, hptrs[1] + sizes[1]);
    EXPECT_GE(upper_bound, hptrs[2] + sizes[2]);
    EXPECT_GE(upper_bound, hptrs[3] + sizes[3]);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
