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

extern "C" __global__ void k_copysignf(const float * a, const float * b,
        float * out, int n) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < n; idx += blockDim.x * gridDim.x) {
        const float _a = a[idx];
        const float _b = b[idx];
        float _out;
        /* copysignf does not do the right thing, so for now, use inline PTX */
        asm("copysign.f32 %0, %1, %2;" : "=f"(_out) : "f"(_a), "f"(_b));

        out[idx] = _out;
    }
}

extern "C" __global__ void k_copysign(const double * a, const double * b,
        double * out, int n) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < n; idx += blockDim.x * gridDim.x) {
        const double _a = a[idx];
        const double _b = b[idx];
        double _out;

        /* copysign does not do the right thing, so for now, use inline PTX */
        asm("copysign.f64 %0, %1, %2;" : "=d"(_out) : "d"(_a), "d"(_b));

        out[idx] = _out;
    }
}

TEST(kCopySign, SinglePrecision) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    float * a;
    float * b;
    float * out;

    ret = cudaMalloc((void **) &a, sizeof(*a) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &b, sizeof(*b) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_copysignf<<<256, n_blocks, 0, stream>>>(a, b, out, N);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(a);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(b);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kCopySign, DoublePrecision) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    double * a;
    double * b;
    double * out;

    ret = cudaMalloc((void **) &a, sizeof(*a) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &b, sizeof(*b) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_copysign<<<256, n_blocks, 0, stream>>>(a, b, out, N);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(a);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(b);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

static __global__ void k_copysign_constconst(float * out) {
    float _out;

    /* copysign does not do the right thing, so for now, use inline PTX */
    asm("copysign.f32 %0, 0fBF000000, 0f3E99999A; // -0.5000f, 0.3000f" :
        "=f"(_out));

    *out = _out;
}

TEST(kCopySign, ConstantConstant) {
    cudaError_t ret;
    cudaStream_t stream;

    float * out;

    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_copysign_constconst<<<1, 1, 0, stream>>>(out);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    float hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(-0.3000f, hout);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t vout;
    const int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x0, vout);
    }
}

static __global__ void k_copysign_single(float * out,
        const float a, const float b) {
    float _out;
    /* copysignf does not do the right thing, so for now, use inline PTX */
    asm("copysign.f32 %0, %1, %2;" : "=f"(_out) : "f"(a), "f"(b));
    *out = _out;
}

TEST(kCopySign, Validity) {
    cudaError_t ret;
    cudaStream_t stream;

    float a_valid, a_invalid;
    float b_valid, b_invalid;

    a_valid = a_invalid =  0.1f;
    b_valid = b_invalid = -0.3f;
    VALGRIND_MAKE_MEM_UNDEFINED(&a_invalid, sizeof(a_invalid));
    VALGRIND_MAKE_MEM_UNDEFINED(&b_invalid, sizeof(b_invalid));

    float * out;

    ret = cudaMalloc((void **) &out, 4 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_copysign_single<<<1, 1, 0, stream>>>(out + 0, a_valid,   b_valid);
    k_copysign_single<<<1, 1, 0, stream>>>(out + 1, a_valid,   b_invalid);
    k_copysign_single<<<1, 1, 0, stream>>>(out + 2, a_invalid, b_valid);
    k_copysign_single<<<1, 1, 0, stream>>>(out + 3, a_invalid, b_invalid);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    union {
        float    f[4];
        uint32_t u[4];
    } hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t vout[4];
    const int vret = VALGRIND_GET_VBITS(&hout, &vout, sizeof(hout));
    if (vret == 1) {
        EXPECT_EQ(0x00000000, vout[0]);
        EXPECT_EQ(0x7FFFFFFF, vout[1]);
        EXPECT_EQ(0x80000000, vout[2]);
        EXPECT_EQ(0xFFFFFFFF, vout[3]);
    }

    EXPECT_EQ(0.3f, hout.f[0]);
    EXPECT_EQ(0x0,  hout.u[1] & 0x80000000);

    hout.u[2] &= 0x7FFFFFFF;
    EXPECT_EQ(0.3f, hout.f[2]);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
