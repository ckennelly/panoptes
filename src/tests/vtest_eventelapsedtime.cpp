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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

TEST(EventElapsedTime, SimpleTest) {
    cudaError_t ret;
    cudaEvent_t event[2];
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventCreate(&event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventCreate(&event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventRecord(event[0], stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventRecord(event[1], stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventSynchronize(event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    float ms;
    ret = cudaEventElapsedTime(&ms, event[0], event[1]);
    EXPECT_EQ(cudaSuccess, ret);

    if (ret == cudaSuccess) {
        EXPECT_LE(0.f, ms);
    }

    ret = cudaEventDestroy(event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventDestroy(event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(EventElapsedTime, ReverseOrder) {
    cudaError_t ret;
    cudaEvent_t event[2];
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventCreate(&event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventCreate(&event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventRecord(event[0], stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventRecord(event[1], stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventSynchronize(event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    float ms;
    ret = cudaEventElapsedTime(&ms, event[1], event[0]);
    EXPECT_EQ(cudaSuccess, ret);

    if (ret == cudaSuccess) {
        EXPECT_GE(0.f, ms);
    }

    ret = cudaEventDestroy(event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventDestroy(event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(EventElapsedTime, NotRecordedBoth) {
    cudaError_t ret;
    cudaEvent_t event[2];

    ret = cudaEventCreate(&event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventCreate(&event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    float ms;
    ret = cudaEventElapsedTime(&ms, event[0], event[1]);
    EXPECT_EQ(cudaErrorInvalidResourceHandle, ret);

    ret = cudaEventDestroy(event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventDestroy(event[1]);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(EventElapsedTime, NotRecordedStart) {
    cudaError_t ret;
    cudaEvent_t event[2];
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventCreate(&event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventCreate(&event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventRecord(event[1], stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventSynchronize(event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    float ms;
    ret = cudaEventElapsedTime(&ms, event[0], event[1]);
    EXPECT_EQ(cudaErrorInvalidResourceHandle, ret);

    ret = cudaEventDestroy(event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventDestroy(event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(EventElapsedTime, NotRecordedEnd) {
    cudaError_t ret;
    cudaEvent_t event[2];
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventCreate(&event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventRecord(event[0], stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventCreate(&event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventSynchronize(event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    float ms;
    ret = cudaEventElapsedTime(&ms, event[0], event[1]);
    EXPECT_EQ(cudaErrorInvalidResourceHandle, ret);

    ret = cudaEventDestroy(event[0]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaEventDestroy(event[1]);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(EventElapsedTime, InvalidBoth) {
    cudaError_t ret;
    float ms;

    ret = cudaEventElapsedTime(&ms, NULL, NULL);
    EXPECT_EQ(cudaErrorInvalidResourceHandle, ret);
}

TEST(EventElapsedTime, InvalidEnd) {
    cudaError_t ret;
    cudaEvent_t event;

    ret = cudaEventCreate(&event);
    ASSERT_EQ(cudaSuccess, ret);

    float ms;
    ret = cudaEventElapsedTime(&ms, event, NULL);
    EXPECT_EQ(cudaErrorInvalidResourceHandle, ret);

    ret = cudaEventDestroy(event);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(EventElapsedTime, InvalidStart) {
    cudaError_t ret;
    cudaEvent_t event;

    ret = cudaEventCreate(&event);
    ASSERT_EQ(cudaSuccess, ret);

    float ms;
    ret = cudaEventElapsedTime(&ms, NULL, event);
    EXPECT_EQ(cudaErrorInvalidResourceHandle, ret);

    ret = cudaEventDestroy(event);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
