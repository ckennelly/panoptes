/**
 * Panoptes - A Binary Translation Framework for CUDA
 * (c) 2011-2013 Chris Kennelly <chris@ckennelly.com>
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

#ifndef __PANOPTES__TESTS__SCOPED_ALLOCATIONS_H__
#define __PANOPTES__TESTS__SCOPED_ALLOCATIONS_H__

#include <boost/utility.hpp>
#include <stdexcept>

/**
 * This provides a simple interface for allocating memory dynamically on the GPU
 * with deallocations done automatically.
 *
 * This class is designed to run in a test environment and all allocations must
 * succeed.
 */
template<typename T>
class scoped_allocation : boost::noncopyable {
public:
    scoped_allocation(size_t size) {
        /*
         * Swallow the last error so the return value of cudaMalloc only
         * concerns cudaMalloc.
         */
        cudaGetLastError();

        cudaError_t ret;
        ret = cudaMalloc(&ptr_, sizeof(*ptr_) * size);
        if (ret != cudaSuccess) {
            throw std::bad_alloc();
        }
    }

    ~scoped_allocation() {
        (void) cudaFree(ptr_);
    }

    operator T*() {
        return ptr_;
    }

    operator const T*() const {
        return ptr_;
    }

    const T* get() const {
        return ptr_;
    }

    void swap(scoped_allocation & b) {
        T *tmp = b.ptr_;
        b.ptr_ = ptr_;
        ptr_   = tmp;
    }
private:
    T *ptr_;

    void operator==(const scoped_allocation &) const;
    void operator!=(const scoped_allocation &) const;
};

template<typename T>
void swap(scoped_allocation<T> & a, scoped_allocation<T> & b) {
    a.swap(b);
}

/**
 * Similar to scoped_allocation, but this allocates a pitched pointer.
 */
template<typename T>
class scoped_pitch : boost::noncopyable {
public:
    scoped_pitch(size_t width, size_t height) {
        cudaError_t ret;
        ret = cudaMallocPitch(&ptr_, &pitch_, sizeof(*ptr_) * width, height);
        if (ret != cudaSuccess) {
            throw std::bad_alloc();
        }
    }

    ~scoped_pitch() {
        (void) cudaFree(ptr_);
    }

    operator T*() {
        return ptr_;
    }

    operator const T*() const {
        return ptr_;
    }

    size_t pitch() const {
        return pitch_;
    }
private:
    T *ptr_;
    size_t pitch_;

    void operator==(const scoped_pitch &) const;
    void operator!=(const scoped_pitch &) const;
};

#endif // __PANOPTES__TESTS__SCOPED_ALLOCATIONS_H__
