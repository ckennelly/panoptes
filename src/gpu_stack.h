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

#ifndef __PANOPTES__GPU_STACK_H_
#define __PANOPTES__GPU_STACK_H_

#include "callout.h"
#include <stack>

namespace panoptes {

/**
 * This class provides a stack implementation for allocating very small
 * objects which we expect to frequently allocate and deallocate.
 */
template<typename T>
class gpu_stack {
public:
    gpu_stack() : stack_size_(1u << 18) {
        initialize();
    }

    gpu_stack(size_t s) : stack_size_(s) {
        initialize();
    }

    T * pop() {
        if (ptrs_.size() > 0) {
            T * ret = ptrs_.top();
            ptrs_.pop();
            return ret;
        } else if (next_ == stack_size_) {
            /* Out of stack space. */
            throw std::bad_alloc();
        } else {
            return &data_[next_++];
        }
    }

    void push(T * t) {
        if (t) {
            ptrs_.push(t);
        }
    }

    size_t size() const { return stack_size_; }
protected:
    void initialize() {
        cudaError_t ret = callout::cudaMalloc((void **) &data_,
            sizeof(T) * stack_size_);
        assert(ret == cudaSuccess);
        next_ = 0;
    }
private:
    size_t stack_size_;
    T *    data_;

    std::stack<T *> ptrs_;
    /* Lazily place values into the stack regime. */
    size_t next_;
}; // end class gpu_stack

}

#endif // __PANOPTES__RING_BUFFER_H_
