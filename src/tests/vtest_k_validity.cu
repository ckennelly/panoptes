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

typedef uint32_t unit_t;
__device__ unit_t symbol;

extern "C" __global__ void k_validity_set(unit_t data) {
    symbol = data;
}

TEST(ValidityTransfer, ParamToSymbol) {
    unsigned valgrind = RUNNING_ON_VALGRIND;
    if (valgrind == 0) {
        /* We cannot perform the tests. */
        return;
    }

    cudaError_t ret;
    cudaStream_t stream;

    unit_t neg;
    ret = cudaMemcpyFromSymbol(&neg, "symbol", sizeof(neg), 0,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    /* Negative control. */
    unit_t vneg = 0;
    const unit_t vexpected_neg = ~((unit_t) 0);

    VALGRIND_GET_VBITS(&neg, &vneg, sizeof(vneg));
    EXPECT_EQ(vexpected_neg, vneg);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const unit_t expected = 0xDEADBEEF;

    k_validity_set<<<1, 1, 0, stream>>>(expected);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Check for positive. */
    unit_t pos;
    ret = cudaMemcpyFromSymbol(&pos, "symbol", sizeof(pos), 0,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    unit_t vpos;
    const unit_t vexpected_pos = 0;

    VALGRIND_GET_VBITS(&pos, &vpos, sizeof(vpos));
    EXPECT_EQ(vexpected_pos, vpos);

    VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&pos, sizeof(pos));
    EXPECT_EQ(expected, pos);
}

/**
 * A serial summation kernel
 */
__global__ void k_sum(int * out, const int * in, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += in[i];
    }

    *out = sum;
}

TEST(ValidityTransfer, Summation) {
    unsigned valgrind = RUNNING_ON_VALGRIND;
    if (valgrind == 0) {
        /* We cannot perform the tests. */
        return;
    }

    cudaError_t ret;
    cudaStream_t stream;

    const int n = 128;
    int32_t *in, *out;
    ret = cudaMalloc((void **) &in, sizeof(*in) * n);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &out, sizeof(*out));
    ASSERT_EQ(cudaSuccess, ret);

    /* Positive control, out should be uninitialized. */
    int32_t pos;
    ret = cudaMemcpy(&pos, out, sizeof(pos), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t vpos;
    const uint32_t vpos_expected = ~((uint32_t) 0);
    VALGRIND_GET_VBITS(&pos, &vpos, sizeof(vpos));
    EXPECT_EQ(vpos_expected, vpos);

    /* Initialize all but one value of in and out. */
    ret = cudaMemset(in , 0x1, sizeof(*out) * (n - 1));
    ASSERT_EQ(cudaSuccess, 0);

    ret = cudaMemset(out, 0, sizeof(*out));
    ASSERT_EQ(cudaSuccess, 0);

    /* Negative control. */
    int32_t neg;
    ret = cudaMemcpy(&neg, out, sizeof(neg), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t vneg;
    const uint32_t vneg_expected = 0;
    VALGRIND_GET_VBITS(&neg, &vneg, sizeof(vneg));
    EXPECT_EQ(vneg_expected, vneg);

    /* Run kernel. */
    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    k_sum<<<1, 1, 0, stream>>>(out, in, n);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Check for transfer. */
    int32_t final;
    ret = cudaMemcpy(&final, out, sizeof(final), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t vfinal;
    const uint32_t vfinal_expected = ~((uint32_t) 0);

    VALGRIND_GET_VBITS(&final, &vfinal, sizeof(vfinal));
    EXPECT_EQ(vfinal_expected, vfinal);

    /* We cannot have any expectations for the actual value of final. */

    /* Free resources. */
    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(in);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
