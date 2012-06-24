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

#include <__cudaFatFormat.h>
#include <boost/lexical_cast.hpp>
#include <boost/thread/locks.hpp>
#include "context_internal.h"
#include "context_memcheck.h"
#include <cuda.h>
#include "fat_binary.h"
#include "global_context.h"
#include "logger.h"
#include "ptx_formatter.h"
#include "ptx_parser.h"
#include <signal.h>

using namespace panoptes;
using internal::create_handle;
using internal::free_handle;

typedef boost::unique_lock<boost::mutex> scoped_lock;

/**
 * TODO:  Make this selectable.
 */
static global_context_memcheck instance_;
global_context & global_context::instance() {
    return instance_;
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
cuda_context & global_context::context() {
    const unsigned device = current_device();
    scoped_lock lock(mx_);
    return context(device);
}

cuda_context & global_context::context(unsigned device) {
    assert(device < devices_);
    cuda_context * ctx = device_contexts_[device];
    thread_info_t * local = threads_.get();
    assert(local);

    if (!(ctx)) {
        ctx = factory(static_cast<int>(device), device_flags_[device]);
        device_contexts_[device] = ctx;
    } else if (!(local->set_on_thread)) {
        CUresult ret = cuCtxSetCurrent(ctx->ctx_);
        assert(ret == CUDA_SUCCESS);
        local->set_on_thread = true;
    }

    return *ctx;
}

const cuda_context & global_context::context() const {
    const unsigned device = current_device();
    scoped_lock lock(mx_);
    return context(device);
}

const cuda_context & global_context::context(unsigned device) const {
    assert(device < devices_);

    cuda_context * ctx = device_contexts_[device];
    thread_info_t * local = threads_.get();
    assert(local);

    if (!(ctx)) {
        ctx = factory(static_cast<int>(device), device_flags_[device]);
        device_contexts_[device] = ctx;
    } else if (!(local->set_on_thread)) {
        CUresult ret = cuCtxSetCurrent(ctx->ctx_);
        assert(ret == CUDA_SUCCESS);
        local->set_on_thread = true;
    }

    return *ctx;
}

global_context::~global_context() {
    for (unsigned i = 0; i < devices_; i++) {
        delete device_contexts_[i];
    }

    // Iterate over module registry to clean up
    for (module_map_t::iterator it = modules_.begin();
            it != modules_.end(); ++it) {
        if (it->second->handle_owned) {
            free_handle(it->first);
        }
        delete it->second;
    }
}

global_context::global_context() {
    cuInit(0);

    /**
     * TODO: Don't ignore the return value.
     */
    {
        int tmp;
        CUresult ret = cuDeviceGetCount(&tmp);
        assert(ret == CUDA_SUCCESS);
        assert(tmp >= 0);
        devices_ = static_cast<unsigned>(tmp);
    }
    device_contexts_.resize(devices_, NULL);
    device_flags_   .resize(devices_, 0);
}

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

cuda_context * global_context::factory(int device,
        unsigned int flags) const {
    return new cuda_context(const_cast<global_context *>(this), device, flags);
}

cudaError_t global_context::cudaDeviceCanAccessPeer(int *canAccessPeer,
        int device, int peerDevice) {
    return callout::cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
}

cudaError_t global_context::cudaDeviceDisablePeerAccess(int peerDevice) {
    return callout::cudaDeviceDisablePeerAccess(peerDevice);
}

cudaError_t global_context::cudaDeviceEnablePeerAccess(int peerDevice, unsigned
        int flags) {
    return callout::cudaDeviceEnablePeerAccess(peerDevice, flags);
}

cudaError_t global_context::cudaDeviceGetByPCIBusId(int *device,
        char *pciBusId) {
    (void) device;
    (void) pciBusId;
    return cudaErrorNotYetImplemented;
}

cudaError_t global_context::cudaDeviceGetPCIBusId(
        char *pciBusId, int len, int device) {
    (void) pciBusId;
    (void) len;
    (void) device;
    return cudaErrorNotYetImplemented;
}

cudaError_t global_context::cudaDeviceReset() {
    scoped_lock lock(mx_);

    const unsigned device = current_device();
    assert(device < devices_);
    cuda_context * ctx = device_contexts_[device];
    if (ctx) {
        ctx->clear();
        delete ctx;
        device_contexts_[device] = NULL;
    }

    return callout::cudaDeviceReset();
}

cudaError_t global_context::cudaGetDevice(int *device) const {
    *device = static_cast<int>(current_device());
    return cudaSuccess;
}

cudaError_t global_context::cudaGetDeviceCount(int *count) const {
    *count = static_cast<int>(devices_);
    return cudaSuccess;
}

void** global_context::cudaRegisterFatBinary(void *fatCubin) {
    scoped_lock lock(mx_);

    /**
     * Check if it has already been registered.
     */
    fatbin_map_t::const_iterator it = fatbins_.find(fatCubin);
    if (it != fatbins_.end()) {
        return it->second;
    }

    /**
     * Fat binaries contain an integer magic cookie.  Versions are
     * distinguished by differing values.  See also GPU Ocelot's implementation
     * of cuda::FatBinaryContext::FatBinaryContext, on which this is based.
     */
    const int    magic  = *(reinterpret_cast<int *>(fatCubin));
    const char * ptx    = NULL;

    if (magic == __cudaFatMAGIC) {
        /* This parsing strategy follows from __cudaFatFormat.h */

        const __cudaFatCudaBinary * handle =
            static_cast<const __cudaFatCudaBinary *>(fatCubin);
        /* TODO Handle gracefully */
        assert(handle->ptx);
        assert(handle->ptx[0].ptx);

        unsigned best_version = 0;

        for (unsigned i = 0; ; i++) {
            if (!(handle->ptx[i].ptx)) {
                break;
            }

            /* Grab compute capability in the form "compute_xy" */
            std::string profile_name = handle->ptx[i].gpuProfileName;
            std::string string_version(
                profile_name.begin() + sizeof("compute_") - 1,
                profile_name.end());

            if (profile_name.size() > 10) {
                char msg[128];
                int msgret = snprintf(msg, sizeof(msg),
                    "Compute mode is too long (%zu bytes).",
                    profile_name.size());
                // sizeof(msg) is small, so the cast is safe.
                assert(msgret < (int) sizeof(msg) - 1);
                logger::instance().print(msg);

                exit(1);
            }

            unsigned numeric_version;
            try {
                numeric_version = boost::lexical_cast<unsigned>(
                    string_version);
            } catch (boost::bad_lexical_cast) {
                char msg[128];
                int msgret = snprintf(msg, sizeof(msg),
                    "Unable to parse compute mode '%s'.",
                    handle->ptx[i].gpuProfileName);
                // sizeof(msg) is small, so the cast is safe.
                assert(msgret < (int) sizeof(msg) - 1);
                logger::instance().print(msg);

                exit(1);
            }

            if (numeric_version > best_version) {
                best_version    = numeric_version;
                ptx             = handle->ptx[i].ptx;
            }
        }
    } else if (magic == __cudaFatMAGIC2) {
        /* This follows from GPU Ocelot */
        const __cudaFatCudaBinary2 * handle =
            static_cast<const __cudaFatCudaBinary2 *>(fatCubin);
        const __cudaFatCudaBinary2Header * header =
            reinterpret_cast<const __cudaFatCudaBinary2Header *>(
                handle->fatbinData);

        const char * base = reinterpret_cast<const char *>(header + 1);
        unsigned long long offset = 0;

        const __cudaFatCudaBinary2Entry * entry =
            reinterpret_cast<const __cudaFatCudaBinary2Entry *>(base);
        while (!(entry->type & FATBIN_2_PTX) && offset < header->length) {
            entry   = reinterpret_cast<const __cudaFatCudaBinary2Entry *>(
                base + offset);
            offset  = entry->binary + entry->binarySize;
        }

        ptx = reinterpret_cast<const char *>(entry) + entry->binary;
    } else {
        char msg[128];
        int msgret = snprintf(msg, sizeof(msg),
            "Unknown cubin magic number '%08X'.", magic);
        // sizeof(msg) is small, so the cast is safe.
        assert(msgret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        exit(1);
    }

    assert(ptx);

    internal::module_t * module = new internal::module_t();
    module->handle_owned = true;

    ptx_parser      parser;

    std::string _ptx(ptx);
    parser.parse(_ptx, &module->ptx);

    void** handle = create_handle();
    instrument(handle, &module->ptx);

    /**
     * Register the module.
     */
    modules_.insert(module_map_t::value_type(handle, module));

    /* Note fatCubins. */
    fatbins_.insert(fatbin_map_t::value_type(fatCubin, handle));
    ifatbins_.insert(fatbin_imap_t::value_type(handle, fatCubin));

    return handle;
}

void global_context::load_ptx(internal::modules_t * target) {
    using internal::module_t;
    typedef boost::unordered_map<module_t *, module_t *> oldnew_map_t;
    oldnew_map_t oldnew_map;

    for (module_map_t::iterator it = modules_.begin();
            it != modules_.end(); ++it) {
        target->modules.push_back(new module_t());
        module_t * const module = target->modules.back();
        /* Module contains a bit of PTX that we can't copy. */
        module->functions = it->second->functions;
        module->variables = it->second->variables;

        typedef module_t::texture_map_t tm_t;
        for (tm_t::const_iterator jit = it->second->textures.begin();
                jit != it->second->textures.end(); ++jit) {
            module_t::texture_t * tex = new module_t::texture_t();
            assert(!(jit->second->has_texref));
            tex->has_texref     = false;
            tex->hostVar        = jit->second->hostVar;
            tex->deviceAddress  = jit->second->deviceAddress;
            tex->deviceName     = jit->second->deviceName;
            tex->dim            = jit->second->dim;
            tex->norm           = jit->second->norm;
            tex->ext            = jit->second->ext;
            assert(!(jit->second->bound));
            tex->bound          = false;

            module->textures.insert(tm_t::value_type(jit->first, tex));
        }

        module->handle_owned = it->second->handle_owned;

        oldnew_map.insert(oldnew_map_t::value_type(it->second, module));

        void ** const fatCubinHandle = it->first;
        std::stringstream ss;
        ss << it->second->ptx;

        CUresult ret = cuModuleLoadData(&module->module, ss.str().c_str());

        /**
         * TODO:  Handle this failure more gracefully.
         */
        assert(ret == CUDA_SUCCESS);
        if (ret != CUDA_SUCCESS) {
            continue;
        }

        /* Load functions. */
        for (module_t::function_map_t::iterator jit =
                module->functions.begin();
                jit != module->functions.end(); ++jit) {
            module_t::function_t & reg = jit->second;
            ret = cuModuleGetFunction(&reg.function,
                module->module, reg.deviceName);
            if (ret != CUDA_SUCCESS) {
                /**
                 * Panoptes should properly manage its interactions to the
                 * driver API such that this is the only failure mode, if any.
                 */
                assert(ret == CUDA_ERROR_NOT_FOUND);

                /**
                 * TODO:  Warn.
                 */
                continue;
            }
        }

        /* Load variables. */
        for (module_t::variable_map_t::iterator jit =
                module->variables.begin(); jit !=
                module->variables.end(); ++jit) {
            module_t::variable_t & reg = jit->second;

            CUdeviceptr dptr;
            size_t bytes;

            ret = cuModuleGetGlobal(&dptr, &bytes, module->module,
                reg.deviceName);
            if (ret == CUDA_ERROR_NOT_FOUND) {
                char msg[256];
                int sret = snprintf(msg, sizeof(msg), "Attempting to register "
                    "nonexistent symbol '%s' associated with fatbin handle %p "
                    "with cudaRegisterVar.", reg.deviceName, fatCubinHandle);
                assert(sret < (int) sizeof(msg));
                logger::instance().print(msg);

                continue;
            } else if (ret != CUDA_SUCCESS) {
                /* Ignore the error. */
                continue;
            }

            if (reg.user_size < 0 || bytes != (size_t) reg.user_size) {
                char msg[128];
                int msgret = snprintf(msg, sizeof(msg),
                    "cudaRegisterVar registered a size (%d) different from "
                    "the actual size (%zu) of the variable.", reg.user_size,
                    bytes);
                // sizeof(msg) is small, so the cast is safe.
                assert(msgret < (int) sizeof(msg) - 1);
                logger::instance().print(msg);
            }

            /* Note device address. */
            reg.deviceAddress = (char *) dptr;

            /* Now ignore what the user provided. */
            reg.size = bytes;
        }

        /* Load textures.*/
        for (module_t::texture_map_t::iterator jit =
                module->textures.begin(); jit != module->textures.end();
                ++jit) {
            module_t::texture_t * const reg = jit->second;
            const struct textureReference * const hostVar = reg->hostVar;
            const int dim = reg->dim;

            if (!(reg->has_texref)) {
                ret = cuTexRefCreate(&reg->texref);
                assert(ret == CUDA_SUCCESS);
                reg->has_texref = true;
            }

            for (int i = 0; i < dim; i++) {
                CUaddress_mode mode;
                switch (hostVar->addressMode[dim]) {
                    case cudaAddressModeWrap:
                        mode = CU_TR_ADDRESS_MODE_WRAP;
                        break;
                    case cudaAddressModeClamp:
                        mode = CU_TR_ADDRESS_MODE_CLAMP;
                        break;
                    case cudaAddressModeMirror:
                        mode = CU_TR_ADDRESS_MODE_MIRROR;
                        break;
                    case cudaAddressModeBorder:
                        mode = CU_TR_ADDRESS_MODE_BORDER;
                        break;
                }

                ret = cuTexRefSetAddressMode(reg->texref, dim, mode);
                if (ret != CUDA_SUCCESS) {
                    /* TODO Don't ignore failure */
                    continue;
                }
            }

            { // Filter mode
                CUfilter_mode mode;
                switch (hostVar->filterMode) {
                    case cudaFilterModePoint:
                        mode = CU_TR_FILTER_MODE_POINT;
                        break;
                    case cudaFilterModeLinear:
                        mode = CU_TR_FILTER_MODE_LINEAR;
                        break;
                }

                /* TODO Don't ignore result. */
                (void) cuTexRefSetFilterMode(reg->texref, mode);
            }

            /* TODO CU_TRSF_READ_AS_INTEGER */

            if (hostVar->normalized) {
                /* TODO Don't ignore result. */
                (void) cuTexRefSetFlags(reg->texref,
                    CU_TRSF_NORMALIZED_COORDINATES);
            }

            if (hostVar->sRGB) {
                /* TODO Don't ignore result. */
                (void) cuTexRefSetFlags(reg->texref, CU_TRSF_SRGB);
            }
        }
    }

    for (function_map_t::const_iterator it = functions_.begin();
            it != functions_.end(); ++it) {
        oldnew_map_t::const_iterator jit = oldnew_map.find(it->second);
        if (jit == oldnew_map.end()) {
            assert(0 && "The impossible happened.");
            continue;
        }

        using internal::modules_t;
        target->functions.insert(
            modules_t::function_map_t::value_type(it->first, jit->second));
    }

    for (variable_map_t::const_iterator it = variables_.begin();
            it != variables_.end(); ++it) {
        oldnew_map_t::const_iterator jit = oldnew_map.find(it->second);
        if (jit == oldnew_map.end()) {
            assert(0 && "The impossible happened.");
            continue;
        }

        using internal::modules_t;
        target->variables.insert(
            modules_t::variable_map_t::value_type(it->first, jit->second));
    }

    for (texture_map_t::const_iterator it = textures_.begin();
            it != textures_.end(); ++it) {
        oldnew_map_t::const_iterator jit = oldnew_map.find(it->second);
        if (jit == oldnew_map.end()) {
            assert(0 && "The impossible happened.");
            continue;
        }

        using internal::modules_t;
        target->textures.insert(
            modules_t::texture_map_t::value_type(it->first, jit->second));
    }
}

void global_context::cudaRegisterFunction(void **fatCubinHandle,
        const char *hostFun, char *deviceFun, const char *deviceName,
        int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
        int *wSize) {
    scoped_lock lock(mx_);

    /**
     * Check for registration.
     */
    module_map_t::const_iterator it = modules_.find(fatCubinHandle);
    if (it == modules_.end()) {
        /**
         * We cannot signal an error here, but it the kernel will fail to
         * launch when attempted.
         */
        return;
    }

    internal::module_t::function_map_t::const_iterator fit =
        it->second->functions.find(hostFun);
    if (fit != it->second->functions.end()) {
        /**
         * Already registered.
         */
        return;
    }

    internal::module_t::function_t reg;
    reg.deviceFun       = deviceFun;
    reg.deviceName      = deviceName;
    reg.thread_limit    = thread_limit;
    reg.tid             = tid;
    reg.bid             = bid;
    reg.bDim            = bDim;
    reg.gDim            = gDim;
    reg.wSize           = wSize;

    it->second->functions.insert(
        internal::module_t::function_map_t::value_type(hostFun, reg));
    functions_.insert(function_map_t::value_type(hostFun, it->second));
    function_names_.insert(
        function_name_map_t::value_type(deviceName, hostFun));
}

void global_context::cudaRegisterTexture(void **fatCubinHandle,
        const struct textureReference *hostVar, const void **deviceAddress,
        const char *deviceName, int dim, int norm, int ext) {
    scoped_lock lock(mx_);

    /**
     * Check for registration.
     */
    module_map_t::const_iterator it = modules_.find(fatCubinHandle);
    if (it == modules_.end()) {
        /**
         * CUDA SIGSEGV's here.
         */
        raise(SIGSEGV);
        return;
    }

    internal::module_t::texture_map_t::const_iterator vit =
        it->second->textures.find(hostVar);
    if (vit != it->second->textures.end()) {
        /**
         * Already registered.
         */
        return;
    }

    internal::module_t::texture_t * reg = new internal::module_t::texture_t();
    reg->hostVar        = hostVar;
    reg->deviceAddress  = deviceAddress;
    reg->deviceName     = deviceName;
    reg->dim            = dim;
    reg->norm           = norm;
    reg->ext            = ext;

    it->second->textures.insert(
        internal::module_t::texture_map_t::value_type(hostVar, reg));
    textures_.insert(texture_map_t::value_type(hostVar, it->second));
    texture_names_.insert(
        texture_name_map_t::value_type(deviceName, hostVar));
}

void global_context::cudaRegisterVar(void **fatCubinHandle, char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global) {
    (void) deviceAddress;
    scoped_lock lock(mx_);

    /**
     * Check for registration.
     */
    module_map_t::const_iterator it = modules_.find(fatCubinHandle);
    if (it == modules_.end()) {
        /**
         * CUDA SIGSEGV's here.
         */
        raise(SIGSEGV);
        return;
    }

    internal::module_t::variable_map_t::const_iterator vit =
        it->second->variables.find(hostVar);
    if (vit != it->second->variables.end()) {
        /**
         * Already registered.
         */
        return;
    }

    internal::module_t::variable_t reg;
    reg.hostVar         = hostVar;
    reg.deviceAddress   = NULL;
    reg.deviceName      = deviceName;
    reg.ext             = ext;
    reg.size            = 0;
    reg.user_size       = size;
    reg.constant        = constant;
    reg.global          = global;

    it->second->variables.insert(
        internal::module_t::variable_map_t::value_type(hostVar, reg));
    variables_.insert(variable_map_t::value_type(hostVar, it->second));
    variable_names_.insert(
        variable_name_map_t::value_type(deviceName, hostVar));
}

cudaError_t global_context::cudaSetDeviceFlags(unsigned int flags) {
    const unsigned device = current_device();
    scoped_lock lock(mx_);

    assert(device < devices_);
    cuda_context * ctx = device_contexts_[device];
    if (ctx) {
        return ctx->setLastError(cudaErrorSetOnActiveProcess);
    }

    device_flags_[device] = flags;
    return cudaSuccess;
}

void** global_context::cudaUnregisterFatBinary(void **fatCubinHandle) {
    scoped_lock lock(mx_);

    fatbin_imap_t::iterator iit = ifatbins_.find(fatCubinHandle);
    if (iit == ifatbins_.end()) {
        return NULL;
    }

    void* fatCubin = iit->second;

    fatbin_map_t::iterator it = fatbins_.find(fatCubin);
    assert(it != fatbins_.end());

    fatbins_.erase(it);
    ifatbins_.erase(iit);

    module_map_t::iterator mit = modules_.find(fatCubinHandle);
    if (mit != modules_.end()) {
        delete mit->second;
        modules_.erase(mit);
    }

    free_handle(fatCubinHandle);

    /** TODO:  It is not clear what needs to be returned here */
    return NULL;
}

cudaError_t global_context::cudaSetDevice(int device) {
    if (device < 0) {
        return cudaErrorInvalidDevice;
    }
    const unsigned udevice = static_cast<unsigned>(device);

    if (udevice >= devices_) {
        return cudaErrorInvalidDevice;
    }

    cudaError_t cret = callout::cudaSetDevice(device);
    if (cret != cudaSuccess) {
        return cret;
    }

    thread_info_t * local = current();
    local->device = udevice;

    if (local->set_on_thread) {
        cuda_context * ctx = device_contexts_[udevice];
        if (ctx) {
            CUresult ret = cuCtxSetCurrent(ctx->ctx_);
            assert(ret == CUDA_SUCCESS);
        } else {
            local->set_on_thread = false;
        }
    } // else: defer until use

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

void global_context::instrument(void ** fatCubinHandle, ptx_t * target) {
    // Do nothing
    (void) fatCubinHandle;
    (void) target;
}

const global_context::function_name_map_t &
        global_context::function_names() const {
    return function_names_;
}

const global_context::variable_name_map_t &
        global_context::variable_names() const {
    return variable_names_;
}

const global_context::texture_name_map_t &
        global_context::texture_names() const {
    return texture_names_;
}
