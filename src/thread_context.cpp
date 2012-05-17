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

#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include "context_memcheck.h"
#include <cuda.h>
#include "thread_context.h"

using namespace panoptes;

namespace {
    /**
     * Nested shared pointers is a bit of a hack, but we want to have a nontrivial
     * constructor for cuda_context.
     *
     * As scoped_{ptr,array}'s, this code causes a double free.
     */
    typedef boost::shared_ptr<cuda_context> context_ptr_t;
    typedef boost::shared_array<context_ptr_t> context_array_t;
    static context_array_t  device_contexts_;
    static int              device_count = 0;
    static boost::once_flag flag;

    /**
     * Initializes the list of contexts for each device
     */
    void context_helper() {
        cudaError_t ret = callout::cudaGetDeviceCount(&device_count);
        assert(ret == cudaSuccess);

        device_contexts_.reset(new context_ptr_t[device_count]);
        for (int i = 0; i < device_count; i++) {
            device_contexts_[i].reset(new cuda_context_memcheck(i));
        }
    }
}

thread_context::thread_context() : runtime_(false), device_(0),
    last_error_(cudaSuccess) { }

thread_context::~thread_context() { }

cudaError_t thread_context::setLastError(cudaError_t error) {
    if (error != cudaSuccess) {
        last_error_ = error;
    }

    return last_error_;
}

cudaError_t thread_context::cudaGetLastError(void) {
    cudaError_t ret = last_error_;
    last_error_ = cudaSuccess;
    return ret;
}

cudaError_t thread_context::cudaPeekAtLastError(void) {
    return last_error_;
}

void thread_context::init_runtime() {
    boost::call_once(context_helper, flag);

    runtime_ = true;
    if (context_stack_.size() == 0) {
        context_stack_.push(device_contexts_[device_].get());
    }
}

static boost::thread_specific_ptr<thread_context> instance_;

thread_context & thread_context::instance() {
    thread_context * self = instance_.get();
    if (self) {
        return *self;
    } else {
        // Create a new thread context for this thread
        thread_context * r = new thread_context();
        instance_.reset(r);
        return *r;
    }
}

cudaError_t thread_context::cudaDeviceCanAccessPeer(int *canAccessPeer, int
        device, int peerDevice) {
    (void) canAccessPeer;
    (void) device;
    (void) peerDevice;
    return cudaErrorNotYetImplemented;
}

cudaError_t thread_context::cudaDeviceDisablePeerAccess(int peerDevice) {
    (void) peerDevice;
    return cudaErrorNotYetImplemented;
}

cudaError_t thread_context::cudaDeviceEnablePeerAccess(int peerDevice, unsigned
        int flags) {
    (void) peerDevice;
    (void) flags;
    return cudaErrorNotYetImplemented;
}

cudaError_t thread_context::cudaGetDevice(int *device) const {
    *device = device_;
    return cudaSuccess;
}

cudaError_t thread_context::cudaGetDeviceCount(int *count) const {
    *count = device_count;
    return cudaSuccess;
}

cudaError_t thread_context::cudaSetDevice(int device) {
    if (device < 0 || device >= device_count) {
        return cudaErrorInvalidDevice;
    }

    /**
     * Per documentation for cudaErrorSetOnActiveProcess, we should fail if
     * there is an existing context (presumably created by the driver API) or
     * made use of the runtime (memory and kernel launches, which trigger
     * a call to init_runtime).
     *
     * TODO:  Since CUDA 4.0 advertises rapid-fire switching between devices,
     *        this notice needs to be take with a grain of salt.  For the time
     *        being, we fail with the error if the current context did not come
     *        from the global device-bound context for the current device.
     */
    if (context_stack_.size() > 0) {
        if (context_stack_.top() != device_contexts_[device_].get()) {
            return cudaErrorSetOnActiveProcess;
        }

        if (context_stack_.size() > 1) {
            /**
             * The driver API has gotten involved, so fail for safety's sake.
             */
            return cudaErrorSetOnActiveProcess;
        }
    } else {
        context_stack_.push(NULL);
    }

    /**
     * Otherwise, swap in the appropriate context.
     */
    device_               = device;
    context_stack_.top() = device_contexts_[device].get();
    return cudaSuccess;
}

cudaError_t thread_context::cudaSetDeviceFlags(unsigned int flags) {
    cudaError_t ret = callout::cudaSetDeviceFlags(flags);
    init_runtime();
    return ret;
}

cuda_context * thread_context::context() {
    init_runtime();
    return context_stack_.top();
}

const cuda_context * thread_context::context() const {
    const_cast<thread_context *>(this)->init_runtime();
    return context_stack_.top();
}

cudaError_t thread_context::cudaSetValidDevices(int *device_arr, int len) {
    (void) device_arr;
    (void) len;
    return cudaErrorNotYetImplemented;
}

cudaError_t thread_context::cudaThreadExit(void) {
    return cudaErrorNotYetImplemented;
}
