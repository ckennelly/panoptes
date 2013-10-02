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

#include <boost/scoped_ptr.hpp>
#include <boost/thread/once.hpp>
#include <boost/thread/locks.hpp>
#include <cstdio>
#include <panoptes/context.h>
#include <panoptes/global_context.h>
#include <panoptes/logger.h>
#include <panoptes/registry.h>

using namespace panoptes;

typedef boost::unique_lock<boost::mutex> scoped_lock;

static boost::scoped_ptr<global_context> instance_;
static boost::once_flag instance_once;
static void instance_setup() {
    const char *c_env = getenv("PANOPTES_TOOL");
    if (c_env == NULL) {
        c_env = "MEMCHECK";
    }

    const std::string env(c_env);
    instance_.reset(registry::instance().create(env));
    if (!(instance_.get())) {
        /* The tool couldn't be found. */
        char msg[256];
        snprintf(msg, sizeof(msg), "Unknown tool '%s'.\n", env.c_str());
        logger::instance().print(msg);
        exit(1);
    }
}

global_context & global_context::instance() {
    boost::call_once(instance_setup, instance_once);
    return *instance_;
}

/**
 * The context() lookup methods need to be fast.  Ideally, we would only
 * acquire the lock to initialize the context and perform a fast memory
 * load (and a compare) only if we did not need to initialize the context.
 * This sounds suspiciously like a double-checked locking pattern.  While
 * it may be safe to implement one according to a strict reading of the
 * C/C++ specifications with astute use of volatile, implementations of
 * volatile in various common compilers are often buggy.  For simplicity,
 * this codepath always holds a lock (hoping for a fast underlying
 * implementation).
 */
context * global_context::context() {
    const unsigned device = current_device();
    scoped_lock lock(mx_);
    return context_impl(device, true);
}

context * global_context::context(unsigned device) {
    if (device >= devices_) {
        return NULL;
    }

    scoped_lock lock(mx_);
    return context_impl(device, true);
}

const context * global_context::context() const {
    const unsigned device = current_device();
    scoped_lock lock(mx_);
    return context(device);
}

const context * global_context::context(unsigned device) const {
    if (device >= devices_) {
        return NULL;
    }

    scoped_lock lock(mx_);
    return context(device);
}

global_context::~global_context() {
    assert(device_contexts_.size() == devices_);
    for (unsigned i = 0; i < devices_; i++) {
        delete device_contexts_[i];
    }
}

global_context::global_context() { }

global_context::thread_info_t::thread_info_t() : device(0),
    set_on_thread(false) { }

global_context::thread_info_t * global_context::current() {
    thread_info_t * local = threads_.get();
    if (!(local)) {
        local = new thread_info_t();
        threads_.reset(local);
    }

    return local;
}

const global_context::thread_info_t * global_context::current() const {
    thread_info_t * local = threads_.get();
    if (!(local)) {
        local = new thread_info_t();
        threads_.reset(local);
    }

    return local;
}

unsigned global_context::current_device() const {
    return current()->device;
}

cudaError_t global_context::cudaDeviceReset() {
    scoped_lock lock(mx_);

    const unsigned device = current_device();
    assert(device < devices());
    panoptes::context * ctx = device_contexts_[device];
    if (ctx) {
        ctx->clear();
        delete ctx;
        device_contexts_[device] = NULL;
    }

    return cudaSuccess;
}

cudaError_t global_context::cudaGetDevice(int *device) const {
    *device = static_cast<int>(current_device());
    return cudaSuccess;
}

cudaError_t global_context::cudaGetDeviceCount(int *count) const {
    *count = static_cast<int>(devices_);
    return cudaSuccess;
}

cudaError_t global_context::cudaSetDevice(int device) {
    if (device < 0) {
        return cudaErrorInvalidDevice;
    }
    const unsigned udevice = static_cast<unsigned>(device);

    if (udevice >= devices_) {
        return cudaErrorInvalidDevice;
    }

    thread_info_t * local = current();
    local->device = udevice;

    return cudaSuccess;
}

cudaError_t global_context::cudaSetDeviceFlags(unsigned int flags) {
    const unsigned device = current_device();
    scoped_lock lock(mx_);

    assert(device < devices_);
    panoptes::context * ctx = device_contexts_[device];
    if (ctx) {
        return ctx->setLastError(cudaErrorSetOnActiveProcess);
    }

    device_flags_[device] = flags;
    return cudaSuccess;
}

cudaError_t global_context::cudaSetValidDevices(int *device_arr, int len) {
    (void) device_arr;
    (void) len;
    return cudaErrorNotYetImplemented;
}

cudaError_t global_context::cudaThreadExit(void) {
    return cudaDeviceReset();
}

unsigned global_context::devices() const {
    return devices_;
}

void global_context::set_devices(unsigned n) {
    device_flags_.resize(n, 0u);

    for (size_t i = n; i < device_contexts_.size(); i++) {
        delete device_contexts_[i];
    }
    device_contexts_.resize(n, NULL);
    devices_ = n;
}

unsigned global_context::device_flags(unsigned device) const {
    assert(device < devices_);
    return device_flags_[device];
}

context * global_context::context_impl(unsigned device, bool instantiate) {
    assert(device < devices_);
    panoptes::context * ctx = device_contexts_[device];

    if (instantiate && !(ctx)) {
        ctx = factory(static_cast<int>(device), device_flags_[device]);
        device_contexts_[device] = ctx;
    }

    return ctx;
}

const context * global_context::context_impl(unsigned device,
        bool instantiate) const {
    assert(device < devices_);
    panoptes::context * ctx = device_contexts_[device];

    if (instantiate && !(ctx)) {
        ctx = factory(static_cast<int>(device), device_flags_[device]);
        device_contexts_[device] = ctx;
    }

    return ctx;
}
