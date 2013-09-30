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

#ifndef __PANOPTES__HOST_GPU_VECTOR_H_
#define __PANOPTES__HOST_GPU_VECTOR_H_

#include <cassert>
#include "gpu_vector.h"
#include <vector>

namespace panoptes {

/**
 * A vector which lives on both the host and the GPU with methods for
 * facilitating transfers.
 */
template<class T>
class host_gpu_vector : public gpu_vector<T> {
public:
    /**
     * Allocates the buffers of n elements.
     *
     * @throws std::bad_alloc on failure
     */
    explicit host_gpu_vector(size_t n) : gpu_vector<T>(n) {
        cudaError_t ret = callout::cudaMallocHost((void**)&host_,
            sizeof(T) * n);
        if (ret != cudaSuccess) {
            throw std::bad_alloc();
        }
    }

    /**
     * Destructor.
     */
    ~host_gpu_vector() {
        callout::cudaFreeHost(host_);
    }

    T * host() { return host_; }
    const T * host() const { return host_; }

    void to_host() {
        cudaError_t ret = callout::cudaMemcpy(host_, gpu_vector<T>::gpu_,
            sizeof(T) * gpu_vector<T>::n_, cudaMemcpyDeviceToHost);
        // TODO:  Better error handling
        assert(ret == cudaSuccess);
    }

    void to_gpu() {
        cudaError_t ret = callout::cudaMemcpy(gpu_vector<T>::gpu_, host_,
            sizeof(T) * gpu_vector<T>::n_, cudaMemcpyHostToDevice);
        // TODO:  Better error handling
        assert(ret == cudaSuccess);
    }
private:
    T * host_;
}; // end class host_gpu_vector

} // end namespace panoptes

#endif // __PANOPTES__HOST_GPU_VECTOR_H_
