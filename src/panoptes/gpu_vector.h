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

#ifndef __PANOPTES__GPU_VECTOR_H_
#define __PANOPTES__GPU_VECTOR_H_

#include <panoptes/callout.h>
#include <new>

namespace panoptes {

/**
 * A GPU-centric vector which does not support resizing
 */
template<class T>
class gpu_vector {
public:
    /**
     * Allocates a GPU-based buffer of n, uninitialized elements.
     *
     * @throws std::bad_alloc on failure
     */
    explicit gpu_vector(size_t n) : n_(n) {
        cudaError_t ret = callout::cudaMalloc((void**)&gpu_, sizeof(T) * n_);
        if (ret != cudaSuccess) {
            throw std::bad_alloc();
        }
    }

    /**
     * Destructor.
     */
    ~gpu_vector() {
       callout::cudaFree(gpu_);
    }

    /**
     * Accessors
     */
    T * gpu() { return gpu_; }
    const T * gpu() const { return gpu_; }
    size_t size() const { return n_; }
protected:
    const size_t n_;
    T * gpu_;
}; // end class gpu_vector

} // end namespace panoptes

#endif // __PANOPTES__GPU_VECTOR_H_
