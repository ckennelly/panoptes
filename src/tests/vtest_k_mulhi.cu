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
#include <stdint.h>

template<typename T>
__global__ void k_mulhi(const T * x, const T * y, T * out, int32_t n) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_mulhi<int32_t>(const int32_t * x, const int32_t * y,
        int32_t * out, int32_t n) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        out[i] = __mulhi(x[i], y[i]);
    }
}

template<>
__global__ void k_mulhi(const uint32_t * x, const uint32_t * y,
        uint32_t * out, int32_t n) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        out[i] = __umulhi(x[i], y[i]);
    }
}

template<>
__global__ void k_mulhi(const int64_t * x, const int64_t * y,
        int64_t * out, int32_t n) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        out[i] = __mul64hi(x[i], y[i]);
    }
}

template<>
__global__ void k_mulhi(const uint64_t * x, const uint64_t * y,
        uint64_t * out, int32_t n) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n; i += blockDim.x * gridDim.x) {
        out[i] = __umul64hi(x[i], y[i]);
    }
}

/**
 * TODO:  nvcc and gtest do not seem to interact nicely on this
 * type-parameterized test.  For now, we are using normal assertions
 * rather than those provided by gtest.  This should be fixed.
 */
template<typename T>
class MulHiTest : public ::testing::Test {
public:
    MulHiTest() {
        cudaError_t ret;
        ret = cudaMalloc((void **) &x, sizeof(*x) * n);
        assert(cudaSuccess == ret);

        ret = cudaMalloc((void **) &y, sizeof(*y) * n);
        assert(cudaSuccess == ret);

        ret = cudaMalloc((void **) &out, sizeof(*out) * n);
        assert(cudaSuccess == ret);
    }

    ~MulHiTest() {
        cudaError_t ret;
        ret = cudaFree(x);
        assert(cudaSuccess == ret);

        ret = cudaFree(y);
        assert(cudaSuccess == ret);

        ret = cudaFree(out);
        assert(cudaSuccess == ret);
    }

    static const int32_t n = 1 << 24;
    T * x;
    T * y;
    T * out;
};

TYPED_TEST_CASE_P(MulHiTest);

TYPED_TEST_P(MulHiTest, Execute) {
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    assert(cudaSuccess == ret);

    k_mulhi<<<256, 16, 0, stream>>>(this->x, this->y, this->out, this->n);

    ret = cudaStreamSynchronize(stream);
    assert(cudaSuccess == ret);

    ret = cudaStreamDestroy(stream);
    assert(cudaSuccess == ret);
}

REGISTER_TYPED_TEST_CASE_P(MulHiTest, Execute);

typedef ::testing::Types<int32_t, uint32_t, int64_t, uint64_t> MyTypes;
INSTANTIATE_TYPED_TEST_CASE_P(My, MulHiTest, MyTypes);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
