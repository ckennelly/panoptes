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
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

template<typename T>
__global__ void k_isfinite(bool * d, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_isfinite(bool * d, const float a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.finite.f32 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "f"(a));
    *d = ret;
}

template<>
__global__ void k_isfinite(bool * d, const double a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.finite.f64 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "d"(a));
    *d = ret;
}

template<typename T>
__global__ void k_isinfinite(bool * d, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_isinfinite(bool * d, const float a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.infinite.f32 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "f"(a));
    *d = ret;
}

template<>
__global__ void k_isinfinite(bool * d, const double a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.infinite.f64 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "d"(a));
    *d = ret;
}

template<typename T>
__global__ void k_isnumber(bool * d, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_isnumber(bool * d, const float a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.number.f32 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "f"(a));
    *d = ret;
}

template<>
__global__ void k_isnumber(bool * d, const double a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.number.f64 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "d"(a));
    *d = ret;
}

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

template<typename T>
__global__ void k_isnormal(bool * d, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_isnormal(bool * d, const float a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.normal.f32 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "f"(a));
    *d = ret;
}

template<>
__global__ void k_isnormal(bool * d, const double a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.normal.f64 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "d"(a));
    *d = ret;
}

template<typename T>
__global__ void k_issubnormal(bool * d, const T a) {
    BOOST_STATIC_ASSERT(sizeof(T) == 0);
}

template<>
__global__ void k_issubnormal(bool * d, const float a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.subnormal.f32 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "f"(a));
    *d = ret;
}

template<>
__global__ void k_issubnormal(bool * d, const double a) {
    int ret;
    asm volatile(
        "{ .reg .pred %tmp;\n"
        "  testp.subnormal.f64 %tmp, %1;\n"
        "  selp.b32 %0, 1, 0, %tmp; }\n" :
        "=r"(ret) : "d"(a));
    *d = ret;
}

TEST(TestPTest, Single) {
    cudaError_t ret;
    cudaStream_t stream;

    bool * d;
    ret = cudaMalloc((void **) &d, 12 * sizeof(d));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const float  fs = 0.5f;
    const double fd = 0.2;
    k_isfinite   <<<1, 1, 0, stream>>>(d + 0,  fs);
    k_isfinite   <<<1, 1, 0, stream>>>(d + 1,  fd);

    k_isinfinite <<<1, 1, 0, stream>>>(d + 2,  fs);
    k_isinfinite <<<1, 1, 0, stream>>>(d + 3,  fd);

    k_isnumber   <<<1, 1, 0, stream>>>(d + 4,  fs);
    k_isnumber   <<<1, 1, 0, stream>>>(d + 5,  fd);

    k_isnan      <<<1, 1, 0, stream>>>(d + 6,  fs);
    k_isnan      <<<1, 1, 0, stream>>>(d + 7,  fd);

    k_isnormal   <<<1, 1, 0, stream>>>(d + 8,  fs);
    k_isnormal   <<<1, 1, 0, stream>>>(d + 9,  fd);

    k_issubnormal<<<1, 1, 0, stream>>>(d + 10, fs);
    k_issubnormal<<<1, 1, 0, stream>>>(d + 11, fd);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hd[12];
    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    EXPECT_TRUE (hd[0]);
    EXPECT_TRUE (hd[1]);
    EXPECT_FALSE(hd[2]);
    EXPECT_FALSE(hd[3]);
    EXPECT_TRUE (hd[4]);
    EXPECT_TRUE (hd[5]);
    EXPECT_FALSE(hd[6]);
    EXPECT_FALSE(hd[7]);
    EXPECT_TRUE (hd[8]);
    EXPECT_TRUE (hd[9]);
    EXPECT_FALSE(hd[10]);
    EXPECT_FALSE(hd[11]);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(TestPTest, SingleValidity) {
    if (!(RUNNING_ON_VALGRIND)) {
        return;
    }

    cudaError_t ret;
    cudaStream_t stream;

    bool * d;
    ret = cudaMalloc((void **) &d, 12 * sizeof(d));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

          float  fs = 0.5f;
          double fd = 0.2;
    VALGRIND_MAKE_MEM_UNDEFINED(&fs, sizeof(fs));
    VALGRIND_MAKE_MEM_UNDEFINED(&fd, sizeof(fd));

    k_isfinite   <<<1, 1, 0, stream>>>(d + 0,  fs);
    k_isfinite   <<<1, 1, 0, stream>>>(d + 1,  fd);

    k_isinfinite <<<1, 1, 0, stream>>>(d + 2,  fs);
    k_isinfinite <<<1, 1, 0, stream>>>(d + 3,  fd);

    k_isnumber   <<<1, 1, 0, stream>>>(d + 4,  fs);
    k_isnumber   <<<1, 1, 0, stream>>>(d + 5,  fd);

    k_isnan      <<<1, 1, 0, stream>>>(d + 6,  fs);
    k_isnan      <<<1, 1, 0, stream>>>(d + 7,  fd);

    k_isnormal   <<<1, 1, 0, stream>>>(d + 8,  fs);
    k_isnormal   <<<1, 1, 0, stream>>>(d + 9,  fd);

    k_issubnormal<<<1, 1, 0, stream>>>(d + 10, fs);
    k_issubnormal<<<1, 1, 0, stream>>>(d + 11, fd);

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    bool hd[12];
    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);

    uint8_t vd[12];
    BOOST_STATIC_ASSERT(sizeof(hd) == sizeof(vd));
    int vret = VALGRIND_GET_VBITS(hd, vd, sizeof(hd));
    if (vret == 1) {
        EXPECT_EQ(0xFF, vd[0]);
        EXPECT_EQ(0xFF, vd[1]);
        EXPECT_EQ(0xFF, vd[2]);
        EXPECT_EQ(0xFF, vd[3]);
        EXPECT_EQ(0xFF, vd[4]);
        EXPECT_EQ(0xFF, vd[5]);
        EXPECT_EQ(0xFF, vd[6]);
        EXPECT_EQ(0xFF, vd[7]);
        EXPECT_EQ(0xFF, vd[8]);
        EXPECT_EQ(0xFF, vd[9]);
        EXPECT_EQ(0xFF, vd[10]);
        EXPECT_EQ(0xFF, vd[11]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
