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

extern "C" __global__ void k_fsincosf(const float * in,
        const int N, float * sin_out, float * cos_out) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        sin_out[idx] = __sinf(in[idx]);
        cos_out[idx] = __cosf(in[idx]);
    }
}

extern "C" __global__ void k_sincosf(const float * in,
        const int N, float * sin_out, float * cos_out) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        sin_out[idx] = sinf(in[idx]);
        cos_out[idx] = cosf(in[idx]);
    }
}

extern "C" __global__ void k_sincos(const double * in,
        const int N, double * sin_out, double * cos_out) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
            idx < N; idx += blockDim.x * gridDim.x) {
        sin_out[idx] = sin(in[idx]);
        cos_out[idx] = cos(in[idx]);
    }
}

TEST(kSINCOS, FastFloat) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    float * in;
    float * sin_values;
    float * cos_values;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &sin_values, sizeof(*sin_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &cos_values, sizeof(*cos_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_fsincosf<<<256, n_blocks, 0, stream>>>(in, N, sin_values, cos_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(sin_values);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(cos_values);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kSINCOS, SinglePrecision) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    float * in;
    float * sin_values;
    float * cos_values;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &sin_values, sizeof(*sin_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &cos_values, sizeof(*cos_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_sincosf<<<256, n_blocks, 0, stream>>>(in, N, sin_values, cos_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(sin_values);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(cos_values);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(kSINCOS, DoublePrecision) {
    cudaError_t ret;
    cudaStream_t stream;

    const int N = 1 << 20;
    const int n_blocks = 32;

    double * in;
    double * sin_values;
    double * cos_values;

    ret = cudaMalloc((void **) &in, sizeof(*in) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &sin_values, sizeof(*sin_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &cos_values, sizeof(*cos_values) * N);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_sincos<<<256, n_blocks, 0, stream>>>(in, N, sin_values, cos_values);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(sin_values);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(cos_values);
    ASSERT_EQ(cudaSuccess, ret);
}

static __global__ void k_sincos_const(float * osin, float * ocos) {
    float _sin, _cos;
    asm volatile("sin.approx.f32 %0, 0f00000000;\n" : "=f"(_sin));
    asm volatile("cos.approx.f32 %0, 0f00000000;\n" : "=f"(_cos));
    *osin = _sin;
    *ocos = _cos;
}

TEST(kSINCOS, Constant) {
    cudaError_t ret;
    cudaStream_t stream;

    float * out;

    ret = cudaMalloc((void **) &out, 2 * sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_sincos_const<<<1, 1, 0, stream>>>(out, out + 1);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    struct {
        float sine;
        float cosine;
    } hout;
    ret = cudaMemcpy(&hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_EQ(0.f, hout.sine);
    EXPECT_EQ(1.f, hout.cosine);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
