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

template<typename T>
__global__ void k_isnan(bool * d, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_isnan(bool * d, const float a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.notanumber.f32 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "f"(a));
    *d = ret;
}

template<>
__global__ void k_isnan(bool * d, const double a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.notanumber.f64 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "d"(a));
    *d = ret;
}

TEST(TestPTest, SingleIsNan) {
    cudaError_t ret;
    cudaStream_t stream;

    bool * d;
    ret = cudaMalloc((void **) &d, 2 * sizeof(d));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const float  fs = 0.5f;
    const double fd = 0.2;
    k_isnan<<<1, 1, 0, stream>>>(d + 0, fs);
    k_isnan<<<1, 1, 0, stream>>>(d + 1, fd);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hd[2];
    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_FALSE(hd[0]);
    EXPECT_FALSE(hd[1]);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

/**
 * TODO:  Add a validity check to see that we propagate invalid bits.
 */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
