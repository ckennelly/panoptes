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

#include <boost/scoped_array.hpp>
#include <boost/static_assert.hpp>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>
#include <cstdio>

__global__ void k_addresses(void ** out, bool flag) {
    __shared__ uint8_t a[32];
    out[0] = a;
    __shared__ uint8_t b[32];
    out[1] = b;
    extern __shared__ uint8_t c[];
    out[2] = c;

    if (flag) {
        __shared__ uint8_t d[32];
        out[3] = d;
    } else {
        __shared__ uint8_t e[64];
        out[3] = e;
    }

    __shared__ uint8_t f[64];
    out[4] = f;
}

class SharedMemoryFixture : public ::testing::TestWithParam<unsigned> {
public:
    SharedMemoryFixture() { }
    ~SharedMemoryFixture() { }
};

TEST_P(SharedMemoryFixture, AddressDifferences) {
    cudaError_t ret;
    cudaStream_t stream;

    void **out;

    ret = cudaMalloc((void **) &out, sizeof(*out) * 5);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const unsigned shmem = 1u << GetParam();
    k_addresses<<<1, 1, shmem, stream>>>(out, false);

    ret = cudaStreamSynchronize(stream);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    void * hout[5];
    ret = cudaMemcpy(hout, out, sizeof(hout), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    for (int i = 0; i < 5; i++) {
        printf("%d %p\n", i, hout[i]);
    }

    ret = cudaFree(out);
    ASSERT_EQ(cudaSuccess, ret);
}

INSTANTIATE_TEST_CASE_P(MyInstance, SharedMemoryFixture,
    ::testing::Range(0u, 13u));

template<typename T>
class SharedMemoryTypedFixture : public ::testing::Test {
public:
    void SetUp() {
        cudaError_t ret;

        ret = cudaMalloc((void **) &out, sizeof(*out));
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaStreamCreate(&stream);
        ASSERT_EQ(cudaSuccess, ret);
    }

    void TearDown() {
        cudaError_t ret;

        ret = cudaStreamDestroy(stream);
        ASSERT_EQ(cudaSuccess, ret);

        ret = cudaFree(out);
        ASSERT_EQ(cudaSuccess, ret);
    }

    cudaStream_t stream;
    T * out;
};

typedef ::testing::Types<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
    uint64_t, int64_t, float, double> MyTypes;
TYPED_TEST_CASE(SharedMemoryTypedFixture, MyTypes);

/**
 * TODO:  Support stores and loads from absolute, constant indices.
 */

template<typename T>
__global__ void k_fixed_index(T * out, const T in) {
    __shared__ T sh[1];
    sh[0] = in;
    /* Force nvcc to flush in to sh rather than write in directly to out. */
    __syncthreads();
    out[0] = sh[0];
}

TYPED_TEST(SharedMemoryTypedFixture, ValidityTransferFixedIndex) {
    TypeParam in;
    memset(&in, 0xAE, sizeof(in));

    k_fixed_index<<<1, 1, 0, this->stream>>>(this->out, in);

    cudaError_t ret;
    ret = cudaStreamSynchronize(this->stream);
    assert(ret == cudaSuccess);

    TypeParam hout;
    ret = cudaMemcpy(&hout, this->out, sizeof(hout), cudaMemcpyDeviceToHost);
    assert(ret == cudaSuccess);

    assert(in == hout);
}

template<typename T>
__global__ void k_variable_index(T * out, const T in, int index) {
    __shared__ T sh[1];
    sh[index] = in;
    /* Force nvcc to flush in to sh rather than write in directly to out. */
    __syncthreads();
    out[0] = sh[index];
}

TYPED_TEST(SharedMemoryTypedFixture, ValidityTransferVariableIndex) {
    TypeParam in;
    memset(&in, 0xAE, sizeof(in));

    k_variable_index<<<1, 1, 0, this->stream>>>(this->out, in, 0);

    cudaError_t ret;
    ret = cudaStreamSynchronize(this->stream);
    assert(ret == cudaSuccess);

    TypeParam hout;
    ret = cudaMemcpy(&hout, this->out, sizeof(hout), cudaMemcpyDeviceToHost);
    assert(ret == cudaSuccess);

    assert(in == hout);
}

/**
 * If we declare this inside the kernel, nvcc convinces itself that we have
 * duplcate declarations (as they are not prefixed by the mangled kernel name).
 */
extern __shared__ uint8_t shdyn[];

template<typename T>
__global__ void k_dynamic(T * out, const T in, int index) {
    ((T *) shdyn)[index] = in;
    /* Force nvcc to flush in to sh rather than write in directly to out. */
    __syncthreads();
    out[0] = ((T *) shdyn)[index];
}

TYPED_TEST(SharedMemoryTypedFixture, ValidityTransferDynamic) {
    TypeParam in;
    memset(&in, 0xAE, sizeof(in));

    k_dynamic<<<1, 1, sizeof(in), this->stream>>>(this->out, in, 0);

    cudaError_t ret;
    ret = cudaStreamSynchronize(this->stream);
    assert(ret == cudaSuccess);

    TypeParam hout;
    ret = cudaMemcpy(&hout, this->out, sizeof(hout), cudaMemcpyDeviceToHost);
    assert(ret == cudaSuccess);

    assert(in == hout);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
