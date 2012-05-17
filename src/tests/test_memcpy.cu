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
#include <valgrind/memcheck.h>
#include <cstdio>

TEST(MemcpyDeathTest, AllDirections) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    int a = 0;
    int * b;
    ASSERT_EQ(cudaSuccess, cudaMalloc((void**) &b, sizeof(*b)));

    int version;
    ASSERT_EQ(cudaSuccess, cudaRuntimeGetVersion(&version));

    /* Panoptes catches this.
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(&a,   &a,   sizeof(a), cudaMemcpyDeviceToDevice));
    cudaGetLastError(); */
    /* Panoptes catches this.
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(&a,   &a,   sizeof(a), cudaMemcpyDeviceToHost));
    cudaGetLastError(); */
    /* Panoptes catches this.
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(&a,   &a,   sizeof(a), cudaMemcpyHostToDevice));
    cudaGetLastError(); */

    /* Panoptes catches this.
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(&a,    b,   sizeof(a), cudaMemcpyDeviceToDevice));
    cudaGetLastError(); */
    /* Panoptes catches this.
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(&a,    b,   sizeof(a), cudaMemcpyHostToDevice));
    cudaGetLastError(); */
    EXPECT_EXIT(
        cudaMemcpy(&a,    b,   sizeof(a), cudaMemcpyHostToHost),
        ::testing::KilledBySignal(SIGSEGV), "");
    cudaGetLastError();

    /* Panoptes catches this.
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy( b,   &a,   sizeof(a), cudaMemcpyDeviceToDevice));
    cudaGetLastError(); */
    /* Error caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy( b,   &a,   sizeof(a), cudaMemcpyDeviceToHost)); */
    cudaGetLastError();
    EXPECT_EXIT(
        cudaMemcpy( b,   &a,   sizeof(a), cudaMemcpyHostToHost),
        ::testing::KilledBySignal(SIGSEGV), "");
    cudaGetLastError();

    /* Caught by Panoptes
    EXPECT_EXIT(
        cudaMemcpy( b,    b,   sizeof(a), cudaMemcpyDeviceToHost),
        ::testing::KilledBySignal(SIGSEGV), "");
    cudaGetLastError(); */
    /* Caught by Panoptes
    EXPECT_EXIT(
        cudaMemcpy( b,    b,   sizeof(a), cudaMemcpyHostToDevice),
        ::testing::KilledBySignal(SIGSEGV), "");
    cudaGetLastError(); */
    if (version >= 4010 /* 4.1 */) {
       EXPECT_EQ(cudaSuccess, cudaMemcpy(b, b, sizeof(a),
            cudaMemcpyHostToHost));
    } else {
        EXPECT_EXIT(
            cudaMemcpy( b,    b,   sizeof(a), cudaMemcpyHostToHost),
            ::testing::KilledBySignal(SIGSEGV), "");
    }
    cudaGetLastError();

    /* Caught by Panoptes.
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(&a,   NULL, sizeof(a), cudaMemcpyDeviceToDevice));
    cudaGetLastError(); */
    /* Error, caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(&a,   NULL, sizeof(a), cudaMemcpyDeviceToHost));
    cudaGetLastError(); */
    /* Caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(&a,   NULL, sizeof(a), cudaMemcpyHostToDevice));
    cudaGetLastError(); */
    EXPECT_EXIT(
        cudaMemcpy(&a,   NULL, sizeof(a), cudaMemcpyHostToHost),
        ::testing::KilledBySignal(SIGSEGV), "");
    cudaGetLastError();

    /* Caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(NULL, &a,   sizeof(a), cudaMemcpyDeviceToDevice));
    cudaGetLastError(); */
    /* Error, caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(NULL, &a,   sizeof(a), cudaMemcpyDeviceToHost));
    cudaGetLastError(); */
    /* Caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(NULL, &a,   sizeof(a), cudaMemcpyHostToDevice));
    cudaGetLastError(); */
    EXPECT_EXIT(
        cudaMemcpy(NULL, &a,   sizeof(a), cudaMemcpyHostToHost),
        ::testing::KilledBySignal(SIGSEGV), "");
    cudaGetLastError();

    /* Caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy( b,   NULL, sizeof(a), cudaMemcpyDeviceToDevice));
    cudaGetLastError(); */
    /* Error, caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy( b,   NULL, sizeof(a), cudaMemcpyDeviceToHost)); */
    EXPECT_EXIT(
        cudaMemcpy( b,   NULL, sizeof(a), cudaMemcpyHostToHost),
        ::testing::KilledBySignal(SIGSEGV), "");
    cudaGetLastError();

    /* Caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(NULL,  b,   sizeof(a), cudaMemcpyDeviceToDevice));
    cudaGetLastError(); */
    /* Caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(NULL,  b,   sizeof(a), cudaMemcpyHostToDevice));
    cudaGetLastError(); */
    EXPECT_EXIT(
        cudaMemcpy(NULL,  b,   sizeof(a), cudaMemcpyHostToHost),
        ::testing::KilledBySignal(SIGSEGV), "");
    cudaGetLastError();

    /* Caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(NULL, NULL, sizeof(a), cudaMemcpyDeviceToDevice));
    cudaGetLastError(); */
    /* Error, caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(NULL, NULL, sizeof(a), cudaMemcpyDeviceToHost));
    cudaGetLastError(); */
    /* Caught by Panoptes
    EXPECT_EQ(cudaErrorInvalidValue,
        cudaMemcpy(NULL, NULL, sizeof(a), cudaMemcpyHostToDevice));
    cudaGetLastError(); */
    if (version >= 4010 /* 4.1 */) {
        EXPECT_EQ(cudaSuccess, cudaMemcpy(NULL, NULL, sizeof(a),
            cudaMemcpyHostToHost));
    } else {
        EXPECT_EXIT(
            cudaMemcpy(NULL, NULL, sizeof(a), cudaMemcpyHostToHost),
            ::testing::KilledBySignal(SIGSEGV), "");
    }
    cudaGetLastError();

    ASSERT_EQ(cudaSuccess, cudaFree(b));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
