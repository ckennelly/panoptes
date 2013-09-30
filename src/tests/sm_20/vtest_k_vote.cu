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
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

static __device__ bool vote_all(bool in_) {
    const int32_t in = in_;
    int out;

    asm volatile(
        "{\n"
        "  .reg .pred %tmp;\n"
        "  setp.ne.s32 %tmp, %1, 0;\n"
        "  vote.all.pred %tmp, %tmp;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(out) : "r"(in));

    return out;
}

static __device__ bool vote_none(bool in_) {
    const int32_t in = in_;
    int out;

    asm volatile(
        "{\n"
        "  .reg .pred %tmp;\n"
        "  setp.ne.s32 %tmp, %1, 0;\n"
        "  vote.all.pred %tmp, !%tmp;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(out) : "r"(in));

    return out;
}

static __device__ bool vote_any(bool in_) {
    const int32_t in = in_;
    int out;

    asm volatile(
        "{\n"
        "  .reg .pred %tmp;\n"
        "  setp.ne.s32 %tmp, %1, 0;\n"
        "  vote.any.pred %tmp, %tmp;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(out) : "r"(in));

    return out;
}

static __device__ bool vote_notall(bool in_) {
    const int32_t in = in_;
    int out;

    asm volatile(
        "{\n"
        "  .reg .pred %tmp;\n"
        "  setp.ne.s32 %tmp, %1, 0;\n"
        "  vote.any.pred %tmp, !%tmp;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(out) : "r"(in));

    return out;
}

__global__ void kv_all(bool * out, int threads, int test) {
    if (threadIdx.x >= threads) {
        return;
    }

    *out = vote_all(threadIdx.x < test);
}

__global__ void kv_none(bool * out, int threads, int test) {
    if (threadIdx.x >= threads) {
        return;
    }

    *out = vote_none(threadIdx.x < test);
}

__global__ void kv_any(bool * out, int threads, int test) {
    if (threadIdx.x >= threads) {
        return;
    }

    *out = vote_any(threadIdx.x < test);
}

__global__ void kv_notall(bool * out, int threads, int test) {
    if (threadIdx.x >= threads) {
        return;
    }

    *out = vote_notall(threadIdx.x < test);
}

/**
 * If threadIdx.x < test, in[0] is loaded as the input for the balloting.
 * Else, in[1] is loaded.
 */
__global__ void k_ballot(uint32_t * out, uint32_t * in, int test) {
    const uint32_t vote = in[threadIdx.x < test ? 0 : 1];
    *out = ballot(vote);
}

TEST(Vote, All) {
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const int warpSize = prop.warpSize;

    bool * out;
    ret = cudaMalloc((void **) &out, 5 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    kv_all<<<1, warpSize, 0, stream>>>(out + 0, warpSize,     warpSize);
    kv_all<<<1, warpSize, 0, stream>>>(out + 1, warpSize - 1, warpSize);
    kv_all<<<1, warpSize, 0, stream>>>(out + 2, warpSize,     warpSize - 1);
    kv_all<<<1, warpSize, 0, stream>>>(out + 3, warpSize - 1, warpSize - 1);
    kv_all<<<1, warpSize, 0, stream>>>(out + 4, warpSize - 1, warpSize - 2);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hout[5];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_TRUE(hout[0]);
    EXPECT_TRUE(hout[1]);
    EXPECT_FALSE(hout[2]);
    EXPECT_TRUE(hout[3]);
    EXPECT_FALSE(hout[4]);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Vote, None) {
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const int warpSize = prop.warpSize;

    bool * out;
    ret = cudaMalloc((void **) &out, 7 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    kv_none<<<1, warpSize, 0, stream>>>(out + 0, warpSize,     warpSize);
    kv_none<<<1, warpSize, 0, stream>>>(out + 1, warpSize - 1, warpSize);
    kv_none<<<1, warpSize, 0, stream>>>(out + 2, warpSize,     warpSize - 1);
    kv_none<<<1, warpSize, 0, stream>>>(out + 3, warpSize - 1, warpSize - 1);
    kv_none<<<1, warpSize, 0, stream>>>(out + 4, warpSize - 1, warpSize - 2);
    kv_none<<<1, warpSize, 0, stream>>>(out + 5, warpSize,     0);
    kv_none<<<1, warpSize, 0, stream>>>(out + 6, warpSize - 1, 0);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hout[7];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_FALSE(hout[0]);
    EXPECT_FALSE(hout[1]);
    EXPECT_FALSE(hout[2]);
    EXPECT_FALSE(hout[3]);
    EXPECT_FALSE(hout[4]);
    EXPECT_TRUE(hout[5]);
    EXPECT_TRUE(hout[6]);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Vote, Any) {
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const int warpSize = prop.warpSize;

    bool * out;
    ret = cudaMalloc((void **) &out, 7 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    kv_any<<<1, warpSize, 0, stream>>>(out + 0, warpSize,     warpSize);
    kv_any<<<1, warpSize, 0, stream>>>(out + 1, warpSize - 1, warpSize);
    kv_any<<<1, warpSize, 0, stream>>>(out + 2, warpSize,     warpSize - 1);
    kv_any<<<1, warpSize, 0, stream>>>(out + 3, warpSize - 1, warpSize - 1);
    kv_any<<<1, warpSize, 0, stream>>>(out + 4, warpSize - 1, warpSize - 2);
    kv_any<<<1, warpSize, 0, stream>>>(out + 5, warpSize,     0);
    kv_any<<<1, warpSize, 0, stream>>>(out + 6, warpSize - 1, 0);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hout[7];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_TRUE(hout[0]);
    EXPECT_TRUE(hout[1]);
    EXPECT_TRUE(hout[2]);
    EXPECT_TRUE(hout[3]);
    EXPECT_TRUE(hout[4]);
    EXPECT_FALSE(hout[5]);
    EXPECT_FALSE(hout[6]);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Vote, NotAll) {
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const int warpSize = prop.warpSize;

    bool * out;
    ret = cudaMalloc((void **) &out, 7 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    kv_notall<<<1, warpSize, 0, stream>>>(out + 0, warpSize,     warpSize);
    kv_notall<<<1, warpSize, 0, stream>>>(out + 1, warpSize - 1, warpSize);
    kv_notall<<<1, warpSize, 0, stream>>>(out + 2, warpSize,     warpSize - 1);
    kv_notall<<<1, warpSize, 0, stream>>>(out + 3, warpSize - 1, warpSize - 1);
    kv_notall<<<1, warpSize, 0, stream>>>(out + 4, warpSize - 1, warpSize - 2);
    kv_notall<<<1, warpSize, 0, stream>>>(out + 5, warpSize,     0);
    kv_notall<<<1, warpSize, 0, stream>>>(out + 6, warpSize - 1, 0);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hout[7];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_FALSE(hout[0]);
    EXPECT_FALSE(hout[1]);
    EXPECT_TRUE(hout[2]);
    EXPECT_FALSE(hout[3]);
    EXPECT_TRUE(hout[4]);
    EXPECT_TRUE(hout[5]);
    EXPECT_TRUE(hout[6]);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Ballot, Validity) {
    /**
     * We allocate two values but only initialize one.  Depending on our choice
     * of test, k_ballot will cause program behavior to depend on an
     * uninitialized value.
     */
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const int warpSize = prop.warpSize;

    uint32_t * out;
    ret = cudaMalloc((void **) &out, 2 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t * in;
    ret = cudaMalloc((void **) &in, 2 * sizeof(*in));
    ASSERT_EQ(cudaSuccess, ret);

    const uint32_t init = 1;
    ret = cudaMemcpy(in, &init, sizeof(init), cudaMemcpyHostToDevice);
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_ballot<<<1, warpSize, 0, stream>>>(out + 0, in, warpSize);
    k_ballot<<<1, warpSize, 0, stream>>>(out + 1, in, warpSize / 2);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hout[2];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    const uint32_t expected[2] = {0xFFFFFFFF * init, 0x0000FFFF * init};

    EXPECT_EQ(expected[0], hout[0]);
    EXPECT_EQ(expected[1], hout[1] & 0x0000FFFF);

    uint32_t vout[2];
    const int vret = VALGRIND_GET_VBITS(hout, vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x0, vout[0]);
        EXPECT_EQ(0xFFFF0000, vout[1]);
    }
}

static __global__ void vote_all_const0(bool * out) {
    int _out;
    asm("{\n"
        "  .reg .pred %tmp;\n"
        "  vote.all.pred %tmp, 0;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_all_const1(bool * out) {
    int _out;
    asm("{\n"
        "  .reg .pred %tmp;\n"
        "  vote.all.pred %tmp, 1;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_none_const0(bool * out) {
    int _out;
    asm("{\n"
        "  .reg .pred %tmp;\n"
        "  vote.all.pred %tmp, !0;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_none_const1(bool * out) {
    int _out;
    asm("{\n"
        "  .reg .pred %tmp;\n"
        "  vote.all.pred %tmp, !1;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_any_const0(bool * out) {
    int _out;
    asm("{\n"
        "  .reg .pred %tmp;\n"
        "  vote.any.pred %tmp, 0;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_any_const1(bool * out) {
    int _out;
    asm("{\n"
        "  .reg .pred %tmp;\n"
        "  vote.any.pred %tmp, 1;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_notall_const0(bool * out) {
    int _out;
    asm("{\n"
        "  .reg .pred %tmp;\n"
        "  vote.any.pred %tmp, !0;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_notall_const1(bool * out) {
    int _out;
    asm("{\n"
        "  .reg .pred %tmp;\n"
        "  vote.any.pred %tmp, !1;\n"
        "  selp.s32 %0, 1, 0, %tmp;\n"
        "}\n" : "=r"(_out));
    *out = _out;
}

TEST(Vote, ConstantArguments) {
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const int warpSize = prop.warpSize;

    bool * out;
    ret = cudaMalloc((void **) &out, 8 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    vote_all_const0   <<<1, warpSize, 0, stream>>>(out + 0);
    vote_all_const1   <<<1, warpSize, 0, stream>>>(out + 1);
    vote_none_const0  <<<1, warpSize, 0, stream>>>(out + 2);
    vote_none_const1  <<<1, warpSize, 0, stream>>>(out + 3);
    vote_any_const0   <<<1, warpSize, 0, stream>>>(out + 4);
    vote_any_const1   <<<1, warpSize, 0, stream>>>(out + 5);
    vote_notall_const0<<<1, warpSize, 0, stream>>>(out + 6);
    vote_notall_const1<<<1, warpSize, 0, stream>>>(out + 7);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hout[8];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_FALSE(hout[0]);
    EXPECT_TRUE (hout[1]);
    EXPECT_TRUE (hout[2]);
    EXPECT_FALSE(hout[3]);
    EXPECT_FALSE(hout[4]);
    EXPECT_TRUE (hout[5]);
    EXPECT_TRUE (hout[6]);
    EXPECT_FALSE(hout[7]);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

static __global__ void vote_ballot_const0(uint32_t * out) {
    uint32_t _out;
    asm("vote.ballot.b32 %0, 0;\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_ballot_const1(uint32_t * out) {
    uint32_t _out;
    asm("vote.ballot.b32 %0, 1;\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_ballot_not_const0(uint32_t * out) {
    uint32_t _out;
    asm("vote.ballot.b32 %0, !0;\n" : "=r"(_out));
    *out = _out;
}

static __global__ void vote_ballot_not_const1(uint32_t * out) {
    uint32_t _out;
    asm("vote.ballot.b32 %0, !1;\n" : "=r"(_out));
    *out = _out;
}

TEST(Ballot, ConstantArguments) {
    cudaError_t ret;
    int device;
    ret = cudaGetDevice(&device);
    ASSERT_EQ(cudaSuccess, ret);

    cudaDeviceProp prop;
    ret = cudaGetDeviceProperties(&prop, device);
    ASSERT_EQ(cudaSuccess, ret);

    const int warpSize = prop.warpSize;

    uint32_t * out;
    ret = cudaMalloc((void **) &out, 8 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    vote_ballot_const0    <<<1, warpSize,     0, stream>>>(out + 0);
    vote_ballot_const1    <<<1, warpSize,     0, stream>>>(out + 1);
    vote_ballot_not_const0<<<1, warpSize,     0, stream>>>(out + 2);
    vote_ballot_not_const1<<<1, warpSize,     0, stream>>>(out + 3);
    vote_ballot_const0    <<<1, warpSize / 2, 0, stream>>>(out + 4);
    vote_ballot_const1    <<<1, warpSize / 2, 0, stream>>>(out + 5);
    vote_ballot_not_const0<<<1, warpSize / 2, 0, stream>>>(out + 6);
    vote_ballot_not_const1<<<1, warpSize / 2, 0, stream>>>(out + 7);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hout[8];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * This is a particularly unexpected outcome that is validated by
     * the control run of the test suite (e.g., without Panoptes).  For
     * immediate arguments, vote.ballot treats the operand as a true predicate
     * value.  Consequently, the result of the ballot is an indicator of which
     * threads participated.
     */
    EXPECT_EQ(0xFFFFFFFF, hout[0]);
    EXPECT_EQ(0xFFFFFFFF, hout[1]);
    EXPECT_EQ(0xFFFFFFFF, hout[2]);
    EXPECT_EQ(0xFFFFFFFF, hout[3]);
    EXPECT_EQ(0x0000FFFF, hout[4]);
    EXPECT_EQ(0x0000FFFF, hout[5]);
    EXPECT_EQ(0x0000FFFF, hout[6]);
    EXPECT_EQ(0x0000FFFF, hout[7]);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
