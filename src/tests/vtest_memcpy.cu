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

TEST(Memcpy, CheckReturnValues) {
    /**
     * The API documentation states that
     * cudaErrorInvalidDevicePointer is a valid return value for cudaMemcpy
     *
     * TODO;  This needs a test.
     */

    /**
     * Test woefully out of range directions.
     */
    int a = 0;
    EXPECT_EQ(cudaErrorInvalidMemcpyDirection,
        cudaMemcpy(&a,   &a,   sizeof(a), (cudaMemcpyKind) -1));
    EXPECT_EQ(cudaErrorInvalidMemcpyDirection,
        cudaMemcpy(NULL, NULL, sizeof(a), (cudaMemcpyKind) -1));
}

/**
 * CUDA4 introduced the cudaMemcpyDefault direction to cudaMemcpy.
 */
TEST(Memcpy, CheckDefaultDirection) {
    int a1 = 0;
    int a2 = 0;
    int * b;
    ASSERT_EQ(cudaSuccess, cudaMalloc((void**) &b, sizeof(*b)));

    EXPECT_EQ(cudaSuccess,
        cudaMemcpy(&a1,   &a2,  sizeof(a1), cudaMemcpyDefault));
    EXPECT_EQ(cudaSuccess,
        cudaMemcpy(&a1,    b,   sizeof(a1), cudaMemcpyDefault));
    EXPECT_EQ(cudaSuccess,
        cudaMemcpy( b,    &a1,  sizeof(a1), cudaMemcpyDefault));
    EXPECT_EQ(cudaSuccess,
        cudaMemcpy( b,    b,    sizeof(a1), cudaMemcpyDefault));

    ASSERT_EQ(cudaSuccess, cudaFree(b));
}

/**
 * This test only performs copies in valid directions as to avoid upsetting
 * Valgrind.  The error-causing tests are in test_memcpy.cu.
 */
TEST(Memcpy, AllDirections) {
    int a1 = 0;
    int a2 = 0;
    int * b;
    ASSERT_EQ(cudaSuccess, cudaMalloc((void**) &b, sizeof(*b) * 2));

    EXPECT_EQ(cudaSuccess,
        cudaMemcpy(&a1,    &a2,    sizeof(a1), cudaMemcpyHostToHost));
    EXPECT_EQ(cudaSuccess,
        cudaMemcpy(&a1,     b + 0, sizeof(a1), cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess,
        cudaMemcpy(&a1,     b + 1, sizeof(a1), cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess,
        cudaMemcpy( b + 0, &a1,    sizeof(a1), cudaMemcpyHostToDevice));
    EXPECT_EQ(cudaSuccess,
        cudaMemcpy( b + 1, &a1,    sizeof(a1), cudaMemcpyHostToDevice));
    EXPECT_EQ(cudaSuccess,
        cudaMemcpy( b + 0,  b + 0, sizeof(a1), cudaMemcpyDeviceToDevice));
    EXPECT_EQ(cudaSuccess,
        cudaMemcpy( b + 1,  b + 1, sizeof(a1), cudaMemcpyDeviceToDevice));

    ASSERT_EQ(cudaSuccess, cudaFree(b));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
