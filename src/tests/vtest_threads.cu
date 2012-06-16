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

#include <boost/thread/barrier.hpp>
#include <boost/thread/thread.hpp>
#include <cuda.h>
#include <gtest/gtest.h>

extern "C" __global__ void k_copy(int * out, const int * in, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
            i += blockDim.x * gridDim.x) {
        out[i] = in[i];
    }
}

typedef std::vector<void *> ptr_vector_t;

struct worker_data {
    ptr_vector_t * addresses;
    boost::barrier * barrier;
    size_t rank;
};

void worker(worker_data * data) {
    /* Initialize. */
    cudaError_t ret;
    cudaStream_t stream;

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    const int n_ints = 1 << 20;
    int * mem;
    ret = cudaMalloc((void **) &mem, sizeof(*mem) * n_ints);
    ASSERT_EQ(cudaSuccess, ret);

    const size_t others = data->addresses->size();
    (*data->addresses)[data->rank] = mem;

    /* Wait. */
    data->barrier->wait();

    for (size_t i = 0; i < 16; i++) {
        const size_t remote = (data->rank + i) % others;
        const int * remote_addr =
            static_cast<const int *>((*data->addresses)[remote]);

        k_copy<<<256, 32, 0, stream>>>(mem, remote_addr, n_ints);

        ret = cudaStreamSynchronize(stream);
        EXPECT_EQ(cudaSuccess, ret);
    }

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    /* Wait. */
    data->barrier->wait();
    ret = cudaFree(mem);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Threads, PingPong) {
    const size_t n_threads = 2;

    ptr_vector_t addresses(n_threads, NULL);
    boost::barrier barrier(n_threads);

    worker_data default_data;
    default_data.addresses = &addresses;
    default_data.barrier   = &barrier;

    std::vector<worker_data> data(n_threads, default_data);
    std::vector<boost::thread *> threads(n_threads);

    /* Start workers. */
    for (size_t i = 0; i < n_threads; i++) {
        data[i].rank = i;

        threads[i] = new boost::thread(worker, &data[i]);
    }

    /* Stop workers. */
    for (size_t i = 0; i < n_threads; i++) {
        threads[i]->join();
        delete threads[i];
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
