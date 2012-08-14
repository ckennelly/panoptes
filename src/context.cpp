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

#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread/locks.hpp>
#include <boost/utility.hpp>
#include "context.h"
#include "context_internal.h"
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include "global_context.h"
#include "logger.h"
#include <signal.h>
#include <sstream>
#include <valgrind/memcheck.h>

using namespace panoptes;

typedef boost::unique_lock<boost::mutex> scoped_lock;
typedef global_context global_t;

cuda_context::cuda_context(global_context * ctx, int device,
        unsigned int flags) : info_loaded_(false), device_(device),
        modules_(new internal::modules_t()), global_(ctx), error_(cudaSuccess) {
    cudaError_t rret = callout::cudaSetDevice(device);
    assert(rret == cudaSuccess);
    (void) callout::cudaSetDeviceFlags(flags);
    rret = callout::cudaFree(0);
    assert(rret == cudaSuccess);

    CUresult ret = cuCtxGetCurrent(&ctx_);
    assert(ret == CUDA_SUCCESS);
    ctx->load_ptx(modules_);

    /**
     * Load version.
     */
    rret = callout::cudaRuntimeGetVersion(&runtime_version_);
    assert(rret == cudaSuccess);
}

cuda_context::td::~td() {
    // Cleanup call stack.
    while (!(call_stack.empty())) {
        delete call_stack.top();
        call_stack.pop();
    }
}

cuda_context::~cuda_context() {
    delete modules_;

    // Ignore this result.  There is little we can do to change
    // things at this point
    (void) cuCtxDestroy(ctx_);
}

cudaError_t cuda_context::cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream) {
    scoped_lock lock(mx_);
    internal::call_t * call = new internal::call_t();
    call->gridDim   = gridDim;
    call->blockDim  = blockDim;

    /**
     * Tests in vtest_setupargument.cu demonstrate that CUDA appears to
     * consider only the least significant bits of sharedMem.
     */
    call->sharedMem = static_cast<uint32_t>(sharedMem & 0xFFFFFFFF);
    call->stream    = stream;

    call_stack_t & cs = thread_data().call_stack;
    cs.push(call);

    /**
     * TODO:  Cache sufficient information about the current device to be able
     *        to return cudaErrorInvalidConfiguration here rather than at
     *        launch.
     */
    return setLastError(cudaSuccess);
}

cudaError_t cuda_context::cudaLaunch(const char *entry) {
    scoped_lock lock(mx_);

    call_stack_t & cs = thread_data().call_stack;
    if (cs.size() == 0) {
        /**
         * This isn't a specified return value in the CUDA 4.x documentation
         * for cudaLaunch, but this has been experimentally validated.
         */
        return cudaErrorInvalidConfiguration;
    }

    /**
     * Find the containing module.
     */
    using internal::modules_t;
    global_t * const g = global();
    modules_t::function_map_t & functions = modules_->functions;
    modules_t::function_map_t::const_iterator fit = functions.find(entry);
    if (fit == functions.end()) {
        if (!(entry)) {
            /* Fail fast. */
            return setLastError(cudaErrorInvalidDeviceFunction);
        }

        /**
         * Try to interpret the entry as an entry name.
         */
        typedef global_t::function_name_map_t fn_t;
        const fn_t & fnames = g->function_names();
        fn_t::const_iterator nit = fnames.find(entry);
        if (nit == fnames.end()) {
            return setLastError(cudaErrorInvalidDeviceFunction);
        }

        entry = nit->second;

        /* Retry. */
        fit = functions.find(entry);
        if (fit == functions.end()) {
            return setLastError(cudaErrorInvalidDeviceFunction);
        }
    }

    /*
     * This double lookup is bit awkward, but it's either this or we maintain a
     * pointer into the containing module from every function *and* have this
     * list for cleanup purposes.
     */
    internal::module_t::function_map_t::const_iterator config_it =
        fit->second->functions.find(entry);
    if (config_it == fit->second->functions.end()) {
        return setLastError(cudaErrorInvalidDeviceFunction);
    }

    const internal::module_t::function_t & config = config_it->second;
    td & d = thread_data();
    boost::scoped_ptr<internal::call_t> call(d.call_stack.top());
    d.call_stack.pop();

    const CUfunction & func = config.function;

    size_t required = 0;
    for (size_t i = 0; i < call->args.size(); i++) {
        required = std::max(required, call->args[i]->size +
            call->args[i]->offset);
    }

    char buffer[4096];
    if (required > sizeof(buffer)) {
        return cudaErrorInvalidValue;
    }

    void * launchconfig[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, buffer,
        CU_LAUNCH_PARAM_BUFFER_SIZE,    &required,
        CU_LAUNCH_PARAM_END};

    for (size_t i = 0; i < call->args.size(); i++) {
        memcpy(buffer + call->args[i]->offset,
            call->args[i]->arg, call->args[i]->size);
    }

    if (call->blockDim.x * call->blockDim.y * call->blockDim.z == 0) {
        char msg[128];
        int msgret = snprintf(msg, sizeof(msg),
            "cudaConfigureCall specified invalid block dimensions "
            "(%d, %d, %d).\nThis launch will fail.", call->blockDim.x,
            call->blockDim.y, call->blockDim.z);
        // sizeof(msg) is small, so the cast is safe.
        assert(msgret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        /**
         * We could defer returning an error; runtime-based callers will carry
         * on with the kernel launch to see the error there.
         */
        return cudaErrorInvalidConfiguration;
    }

    if (!(load_device_info())) {
        /* TODO Handle more appropriately. */
        return cudaErrorUnknown;
    }

    if (call->sharedMem > info_.sharedMemPerBlock) {
        return cudaErrorInvalidValue;
    }

    CUresult ret = cuLaunchKernel(func, call->gridDim.x, call->gridDim.y,
        call->gridDim.z, call->blockDim.x, call->blockDim.y, call->blockDim.z,
        call->sharedMem, call->stream, NULL, launchconfig);

    cudaError_t ret_ = cudaSuccess;

    switch (ret) {
        case CUDA_SUCCESS:
            ret_ = cudaSuccess;
            break;
        case CUDA_ERROR_DEINITIALIZED:
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_INVALID_VALUE:
            /* Somehow, except for the sharedMem error above, INVALID_VALUE
             * also maps to a configuration error. */
            ret_ = cudaErrorInvalidConfiguration;
            break;
        case CUDA_ERROR_LAUNCH_FAILED:
            ret_ = cudaErrorLaunchFailure;
            break;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            ret_ = cudaErrorLaunchOutOfResources;
            break;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            ret_ = cudaErrorLaunchTimeout;
            break;
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            ret_ = cudaErrorSharedObjectInitFailed;
            break;
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            /**
             * The analogue for this error code is not clear.
             */
        default:
            ret_ = cudaErrorUnknown;
            break;
    }

    return setLastError(ret_);
}

cudaError_t cuda_context::cudaSetupArgument(const void *arg, size_t size,
        size_t offset) {
    scoped_lock lock(mx_);

    td & d = thread_data();
    if (d.call_stack.size() == 0) {
        /**
         * According to the CUDA 4.0 documentation, cudaSetupArgument always
         * returns cudaSuccess *but* requires that cudaConfigureCall is called
         * first.  We return cudaErrorMissingConfiguration now if that
         * sequencing does not occur.
         *
         * CUDA 4.1 SIGSEGV's
         */
        raise(SIGSEGV);
        return setLastError(cudaErrorMissingConfiguration);
    }

    internal::arg_t * arg_copy = new internal::arg_t(arg, size, offset);
    d.call_stack.top()->args.push_back(arg_copy);

    return setLastError(cudaSuccess);
}

cudaError_t cuda_context::cudaThreadGetCacheConfig(
        enum cudaFuncCache *pCacheConfig) {
    return callout::cudaThreadGetCacheConfig(pCacheConfig);
}

cudaError_t cuda_context::cudaThreadGetLimit(size_t *pValue,
        enum cudaLimit limit) {
    return callout::cudaThreadGetLimit(pValue, limit);
}

cudaError_t cuda_context::cudaThreadSetCacheConfig(
        enum cudaFuncCache cacheConfig) {
    return callout::cudaThreadSetCacheConfig(cacheConfig);
}

cudaError_t cuda_context::cudaThreadSetLimit(enum cudaLimit limit,
        size_t value) {
    return callout::cudaThreadSetLimit(limit, value);
}

cudaError_t cuda_context::cudaThreadSynchronize() {
    return cudaDeviceSynchronize();
}

cudaError_t cuda_context::cudaBindSurfaceToArray(
        const struct surfaceReference *surfref, const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc) {
    return callout::cudaBindSurfaceToArray(surfref, array, desc);
}

cudaError_t cuda_context::cudaBindTexture(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t size) {
    scoped_lock lock(mx_);

    typedef internal::modules_t::texture_map_t tm_t;
    tm_t::const_iterator it = modules_->textures.find(texref);
    if (it == modules_->textures.end()) {
        /* TODO */
        return cudaErrorInvalidTexture;
    }

    internal::module_t::texture_map_t::const_iterator jit =
        it->second->textures.find(texref);
    if (jit == it->second->textures.end()) {
        /* TODO */
        return cudaErrorInvalidTexture;
    }

    if (!(devPtr)) {
        return cudaErrorUnknown;
    }

    size_t internal_offset;
    CUresult ret = cuTexRefSetAddress(&internal_offset, jit->second->texref,
        (CUdeviceptr) devPtr, size);
    switch (ret) {
        case CUDA_SUCCESS:
            break;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_DEINITIALIZED:
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        default:
            /* TODO */
            return cudaErrorNotYetImplemented;
    }

    if (internal_offset != 0 && !(offset)) {
        return cudaErrorInvalidValue;
    } else if (offset) {
        *offset = internal_offset;
    }

    CUarray_format format;

    switch (desc->f) {
        case cudaChannelFormatKindSigned:
            format = CU_AD_FORMAT_SIGNED_INT32;
            break;
        case cudaChannelFormatKindUnsigned:
            format = CU_AD_FORMAT_UNSIGNED_INT32;
            break;
        case cudaChannelFormatKindFloat:
            format = CU_AD_FORMAT_FLOAT;
            break;
        case cudaChannelFormatKindNone:
            /* TODO */
            return cudaErrorInvalidValue;
    }

    ret = cuTexRefSetFormat(jit->second->texref, format,
        (desc->x + desc->y + desc->z + desc->w + 31) / 32);
    switch (ret) {
        case CUDA_SUCCESS:
            break;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_DEINITIALIZED:
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        default:
            /* TODO */
            return cudaErrorNotYetImplemented;
    }

    jit->second->bound  = true;
    jit->second->offset = internal_offset;
    return cudaSuccess;
}

cudaError_t cuda_context::cudaBindTexture2D(size_t *offset, const struct
        textureReference *texref, const void *devPtr, const struct
        cudaChannelFormatDesc *desc, size_t width, size_t height, size_t
        pitch) {
    return callout::cudaBindTexture2D(offset, texref, devPtr, desc, width,
        height, pitch);
}

cudaError_t cuda_context::cudaBindTextureToArray(
        const struct textureReference *texref, const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc) {
    scoped_lock lock(mx_);

    typedef internal::modules_t::texture_map_t tm_t;
    const tm_t::const_iterator it = modules_->textures.find(texref);
    if (it == modules_->textures.end()) {
        /* TODO */
        return cudaErrorInvalidTexture;
    }

    internal::module_t::texture_map_t::const_iterator jit =
        it->second->textures.find(texref);
    if (jit == it->second->textures.end()) {
        /* TODO */
        return cudaErrorInvalidTexture;
    }

    if (!(desc)) {
        /* CUDA SIGSEGV's here. */
        char msg[128];
        int msgret = snprintf(msg, sizeof(msg),
            "cudaBindTextureToArray called with null descriptor.");
        // sizeof(msg) is small, so the cast is safe.
        assert(msgret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        raise(SIGSEGV);
        return cudaErrorInvalidValue;
    }

    if (!(array)) {
        return cudaErrorInvalidResourceHandle;
    }

    CUresult ret = cuTexRefSetArray(jit->second->texref, (CUarray) array,
        CU_TRSA_OVERRIDE_FORMAT);
    switch (ret) {
        case CUDA_SUCCESS:
            break;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_DEINITIALIZED:
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        default:
            /* TODO */
            return cudaErrorInvalidResourceHandle;
    }

    jit->second->bound  = true;
    jit->second->offset = 0;
    return cudaSuccess;
}

cudaError_t cuda_context::cudaChooseDevice(int *device,
        const struct cudaDeviceProp *prop) {
    return callout::cudaChooseDevice(device, prop);
}

cudaChannelFormatDesc cuda_context::cudaCreateChannelDesc(int x, int y, int z,
        int w, enum cudaChannelFormatKind f) {
    cudaChannelFormatDesc ret;
    ret.x = x;
    ret.y = y;
    ret.z = z;
    ret.w = w;
    ret.f = f;
    return ret;
}

cudaError_t cuda_context::cudaDeviceGetCacheConfig(
        enum cudaFuncCache *pCacheConfig) {
    return callout::cudaDeviceGetCacheConfig(pCacheConfig);
}

cudaError_t cuda_context::cudaDeviceGetLimit(size_t *pValue,
        enum cudaLimit limit) {
    return callout::cudaDeviceGetLimit(pValue, limit);
}

cudaError_t cuda_context::cudaDeviceSetCacheConfig(
        enum cudaFuncCache cacheConfig) {
    return callout::cudaDeviceSetCacheConfig(cacheConfig);
}

cudaError_t cuda_context::cudaDeviceSetLimit(enum cudaLimit limit,
        size_t value) {
    return callout::cudaDeviceSetLimit(limit, value);
}

cudaError_t cuda_context::cudaDeviceSynchronize() {
    cudaError_t ret = callout::cudaDeviceSynchronize();
    return setLastError(ret);
}

cudaError_t cuda_context::cudaDriverGetVersion(int *driverVersion) {
    cudaError_t ret = callout::cudaDriverGetVersion(driverVersion);
    if (ret == cudaSuccess) {
        (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(
            driverVersion, sizeof(*driverVersion));
    } else {
        (void) VALGRIND_MAKE_MEM_UNDEFINED(driverVersion,
            sizeof(*driverVersion));
    }

    return ret;
}

cudaError_t cuda_context::cudaEventCreate(cudaEvent_t *event) {
    return callout::cudaEventCreate(event);
}

cudaError_t cuda_context::cudaEventCreateWithFlags(cudaEvent_t *event,
        unsigned int flags) {
    return callout::cudaEventCreateWithFlags(event, flags);
}

cudaError_t cuda_context::cudaEventDestroy(cudaEvent_t event) {
    return callout::cudaEventDestroy(event);
}

cudaError_t cuda_context::cudaEventElapsedTime(float *ms, cudaEvent_t start,
        cudaEvent_t end) {
    return callout::cudaEventElapsedTime(ms, start, end);
}

cudaError_t cuda_context::cudaEventQuery(cudaEvent_t event) {
    return callout::cudaEventQuery(event);
}

cudaError_t cuda_context::cudaEventRecord(cudaEvent_t event,
        cudaStream_t stream) {
    return callout::cudaEventRecord(event, stream);
}

cudaError_t cuda_context::cudaEventSynchronize(cudaEvent_t event) {
    return callout::cudaEventSynchronize(event);
}

cudaError_t cuda_context::cudaFree(void *devPtr) {
    return setLastError(callout::cudaFree(devPtr));
}

cudaError_t cuda_context::cudaFreeArray(struct cudaArray *array) {
    return callout::cudaFreeArray(array);
}

cudaError_t cuda_context::cudaFreeHost(void *ptr) {
    return callout::cudaFreeHost(ptr);
}

cudaError_t cuda_context::cudaFuncGetAttributes(
        struct cudaFuncAttributes *attr, const char *func) {
    if (!(attr)) {
        return cudaErrorInvalidValue;
    }

    if (!(func)) {
        /**
         * If CUDA 4.2, return cudaErrorInvalidDeviceFunction. Otherwise, for
         * CUDA 4.1 and older, return cudaErrorUnknown.
         */
        if (runtime_version_ >= 4020) {
            return cudaErrorInvalidDeviceFunction;
        } else {
            return cudaErrorUnknown;
        }
    }

    /**
     * For reasons that are not very clear, CUDA appears to access the memory
     * at func unless it is obviously not addressable (e.g., NULL).  This leads
     * to SIGSEGV's when the memory is not addressable.
     */
    char validity;
    unsigned valgrind = VALGRIND_GET_VBITS(func, &validity, sizeof(validity));
    if (valgrind == 3) {
        char msg[128];
        int msgret = snprintf(msg, sizeof(msg),
            "cudaFuncGetAttributes called with unaddressable function %p.",
            func);
        // sizeof(msg) is small, so the cast is safe.
        assert(msgret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);
    }

    /**
     * GCC happily compiles this load when we dereference a volatile pointer.
     * Clang will not emit the load if we do not use the value in some fashion,
     * even if it has no impact on the execution of the program (x does not get
     * used beyond this point).
     */
    volatile int x = 0;
    x += *static_cast<volatile const char *>(func);

    scoped_lock lock(mx_);

    typedef internal::modules_t::function_map_t fm_t;
    fm_t::const_iterator fit = modules_->functions.find(func);
    if (fit == modules_->functions.end()) {
        return setLastError(cudaErrorInvalidDeviceFunction);
    }

    internal::module_t::function_map_t::const_iterator config_it =
        fit->second->functions.find(func);
    if (config_it == fit->second->functions.end()) {
        return setLastError(cudaErrorInvalidDeviceFunction);
    }

    CUfunction dfunc = config_it->second.function;

#define LOAD_ATTRIBUTE(attribute, type, field)                      \
    do {                                                            \
        int i;                                                      \
        CUresult ret = cuFuncGetAttribute(&i, (attribute), dfunc);  \
        switch (ret) {                                              \
            case CUDA_SUCCESS: break;                               \
            case CUDA_ERROR_DEINITIALIZED:                          \
            case CUDA_ERROR_NOT_INITIALIZED:                        \
            case CUDA_ERROR_INVALID_CONTEXT:                        \
            default:                                                \
                return cudaErrorInitializationError;                \
        }                                                           \
                                                                    \
        attr->field = (type) i;                                     \
    } while (false)

    LOAD_ATTRIBUTE(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        int, maxThreadsPerBlock);
    LOAD_ATTRIBUTE(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        size_t, sharedSizeBytes);
    LOAD_ATTRIBUTE(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
        size_t, constSizeBytes);
    LOAD_ATTRIBUTE(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        size_t, localSizeBytes);
    LOAD_ATTRIBUTE(CU_FUNC_ATTRIBUTE_NUM_REGS,
        int, numRegs);
    LOAD_ATTRIBUTE(CU_FUNC_ATTRIBUTE_PTX_VERSION,
        int, ptxVersion);
    LOAD_ATTRIBUTE(CU_FUNC_ATTRIBUTE_BINARY_VERSION,
        int, binaryVersion);

#undef LOAD_ATTRIBUTE

    return cudaSuccess;
}

cudaError_t cuda_context::cudaFuncSetCacheConfig(const char *func,
        enum cudaFuncCache cacheConfig) {
    scoped_lock lock(mx_);

    typedef internal::modules_t::function_map_t fm_t;
    fm_t::const_iterator fit = modules_->functions.find(func);
    if (fit == modules_->functions.end()) {
        return setLastError(cudaErrorInvalidDeviceFunction);
    }

    internal::module_t::function_map_t::const_iterator config_it =
        fit->second->functions.find(func);
    if (config_it == fit->second->functions.end()) {
        return setLastError(cudaErrorInvalidDeviceFunction);
    }

    CUfunction dfunc = config_it->second.function;
    CUfunc_cache dconfig;
    switch (cacheConfig) {
        case cudaFuncCachePreferNone:
            dconfig = CU_FUNC_CACHE_PREFER_NONE;
            break;
        case cudaFuncCachePreferShared:
            dconfig = CU_FUNC_CACHE_PREFER_SHARED;
            break;
        case cudaFuncCachePreferL1:
            dconfig = CU_FUNC_CACHE_PREFER_L1;
            break;
        case cudaFuncCachePreferEqual:
            dconfig = CU_FUNC_CACHE_PREFER_EQUAL;
            break;
    }

    CUresult ret = cuFuncSetCacheConfig(dfunc, dconfig);
    switch (ret) {
        case CUDA_SUCCESS:
            return cudaSuccess;
        case CUDA_ERROR_DEINITIALIZED:
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        default:
            return cudaErrorInitializationError;
    }
}

cudaError_t cuda_context::cudaGetChannelDesc(
        struct cudaChannelFormatDesc *desc, const struct cudaArray *array) {
    return cudaGetChannelDesc(desc, array);
}

cudaError_t cuda_context::cudaGetDeviceProperties(struct cudaDeviceProp *prop,
        int device) {
    return callout::cudaGetDeviceProperties(prop, device);
}

cudaError_t cuda_context::cudaGetExportTable(const void **ppExportTable,
        const cudaUUID_t *pExportTableId) {
    /** See http://forums.nvidia.com/index.php?showtopic=188196 for what this
     *  might do... */
    return callout::cudaGetExportTable(ppExportTable, pExportTableId);
}

cudaError_t cuda_context::cudaGetSurfaceReference(
        const struct surfaceReference **surfRef, const char *symbol) {
    return callout::cudaGetSurfaceReference(surfRef, symbol);
}

cudaError_t cuda_context::cudaGetLastError() {
    cudaError_t ret = error_;
    error_ = cudaSuccess;
    return ret;
}

cudaError_t cuda_context::cudaGetSymbolAddress(void **devPtr,
        const char *symbol) {
    if (!(devPtr)) {
        return cudaErrorInvalidValue;
    }

    scoped_lock lock(mx_);

    using internal::modules_t;
    modules_t::variable_map_t::const_iterator it =
        modules_->variables.find(symbol);
    if (it == modules_->variables.end()) {
        typedef global_t::variable_name_map_t vn_t;
        const vn_t & vn = global_->variable_names();
        vn_t::const_iterator nit = vn.find(symbol);
        if (nit == vn.end()) {
            return cudaErrorInvalidSymbol;
        }

        /* Try again */
        symbol = static_cast<const char *>(nit->second);
        it = modules_->variables.find(symbol);
        if (it == modules_->variables.end()) {
            /* It shouldn't be in our variable names list if it's not
             * actually there. */
            assert(0 && "The impossible happened.");
            return cudaErrorInvalidSymbol;
        }
    }

    internal::module_t::variable_map_t::const_iterator vit =
        it->second->variables.find(symbol);
    assert(vit != it->second->variables.end());
    *devPtr = vit->second.deviceAddress;

    return cudaSuccess;
}

cudaError_t cuda_context::cudaGetSymbolSize(size_t *size, const char *symbol) {
    if (!(size)) {
        return cudaErrorInvalidValue;
    }

    scoped_lock lock(mx_);

    using internal::modules_t;
    modules_t::variable_map_t & variables = modules_->variables;
    modules_t::variable_map_t::const_iterator it = variables.find(symbol);

    if (it == variables.end()) {
        typedef global_t::variable_name_map_t vn_t;
        const vn_t & vnames = global_->variable_names();
        vn_t::const_iterator nit = vnames.find(symbol);
        if (nit == vnames.end()) {
            return cudaErrorInvalidSymbol;
        }

        /* Try again */
        symbol = static_cast<const char *>(nit->second);
        it = variables.find(symbol);
        if (it == variables.end()) {
            /* It shouldn't be in our variable names list if it's not
             * actually there. */
            assert(0 && "The impossible happened.");
            return cudaErrorInvalidSymbol;
        }
    }

    internal::module_t::variable_map_t::const_iterator vit =
        it->second->variables.find(symbol);
    assert(vit != it->second->variables.end());
    *size = vit->second.size;

    return cudaSuccess;
}

cudaError_t cuda_context::cudaGetTextureAlignmentOffset(size_t *offset,
        const struct textureReference *texref) {
    scoped_lock lock(mx_);

    typedef internal::modules_t::texture_map_t tm_t;
    const tm_t & textures = modules_->textures;
    tm_t::const_iterator it = textures.find(texref);
    if (it == textures.end()) {
        return cudaErrorInvalidTexture;
    }

    internal::module_t::texture_map_t::const_iterator jit =
        it->second->textures.find(texref);
    if (jit == it->second->textures.end()) {
        assert(0 && "The impossible happened.");
        return cudaErrorInvalidTexture;
    }

    if (!(offset)) {
        return cudaErrorInvalidValue;
    }

    if (!(jit->second->bound)) {
        return cudaErrorInvalidTextureBinding;
    }

    *offset = jit->second->offset;
    return cudaSuccess;
}

cudaError_t cuda_context::cudaGetTextureReference(
        const struct textureReference **texref, const char *symbol) {
    if (!(symbol)) {
        return cudaErrorUnknown;
    }

    scoped_lock lock(mx_);

    typedef global_t::texture_name_map_t tn_t;
    const tn_t & texture_names = global_->texture_names();
    tn_t::const_iterator it = texture_names.find(symbol);
    if (it == texture_names.end()) {
        /* TODO */
        return cudaErrorInvalidTexture;
    }

    *texref = it->second;
    return cudaSuccess;
}

cudaError_t cuda_context::cudaGraphicsMapResources(int count,
        cudaGraphicsResource_t *resources, cudaStream_t stream) {
    return callout::cudaGraphicsMapResources(count, resources, stream);
}

cudaError_t cuda_context::cudaGraphicsResourceGetMappedPointer(void **devPtr,
        size_t *size, cudaGraphicsResource_t resource) {
    return callout::cudaGraphicsResourceGetMappedPointer(devPtr, size,
        resource);
}

cudaError_t cuda_context::cudaGraphicsResourceSetMapFlags(
        cudaGraphicsResource_t resource, unsigned int flags) {
    return callout::cudaGraphicsResourceSetMapFlags(resource, flags);
}

cudaError_t cuda_context::cudaGraphicsSubResourceGetMappedArray(
        struct cudaArray **array, cudaGraphicsResource_t resource,
        unsigned int arrayIndex, unsigned int mipLevel) {
    return callout::cudaGraphicsSubResourceGetMappedArray(array, resource,
        arrayIndex, mipLevel);
}

cudaError_t cuda_context::cudaGraphicsUnmapResources(int count,
        cudaGraphicsResource_t *resources, cudaStream_t stream) {
    return callout::cudaGraphicsUnmapResources(count, resources, stream);
}

cudaError_t cuda_context::cudaGraphicsUnregisterResource(
        cudaGraphicsResource_t resource) {
    return callout::cudaGraphicsUnregisterResource(resource);
}

cudaError_t cuda_context::cudaHostAlloc(void **pHost, size_t size,
        unsigned int flags) {
    return callout::cudaHostAlloc(pHost, size, flags);
}

cudaError_t cuda_context::cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags) {
    return callout::cudaHostGetDevicePointer(pDevice, pHost, flags);
}

cudaError_t cuda_context::cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    return callout::cudaHostGetFlags(pFlags, pHost);
}

cudaError_t cuda_context::cudaHostRegister(void *ptr, size_t size,
        unsigned int flags) {
    return callout::cudaHostRegister(ptr, size, flags);
}

cudaError_t cuda_context::cudaHostUnregister(void *ptr) {
    return callout::cudaHostUnregister(ptr);
}

cudaError_t cuda_context::cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
        cudaEvent_t event) {
    return callout::cudaIpcGetEventHandle(handle, event);
}

cudaError_t cuda_context::cudaIpcOpenEventHandle(cudaEvent_t *event,
        cudaIpcEventHandle_t handle) {
    return callout::cudaIpcOpenEventHandle(event, handle);
}

cudaError_t cuda_context::cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,
        void *devPtr) {
    return callout::cudaIpcGetMemHandle(handle, devPtr);
}

cudaError_t cuda_context::cudaIpcOpenMemHandle(void **devPtr,
        cudaIpcMemHandle_t handle, unsigned int flags) {
    return callout::cudaIpcOpenMemHandle(devPtr, handle, flags);
}

cudaError_t cuda_context::cudaIpcCloseMemHandle(void *devPtr) {
    return callout::cudaIpcCloseMemHandle(devPtr);
}

cudaError_t cuda_context::cudaMalloc(void **devPtr, size_t size) {
    return setLastError(callout::cudaMalloc(devPtr, size));
}

cudaError_t cuda_context::cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr,
        struct cudaExtent extent) {
    return setLastError(callout::cudaMalloc3D(pitchedDevPtr, extent));
}

cudaError_t cuda_context::cudaMalloc3DArray(struct cudaArray** array,
        const struct cudaChannelFormatDesc *desc, struct cudaExtent extent,
        unsigned int flags) {
    return callout::cudaMalloc3DArray(array, desc, extent, flags);
}

cudaError_t cuda_context::cudaMallocArray(struct cudaArray **array,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags) {
    return callout::cudaMallocArray(array, desc, width, height, flags);
}

cudaError_t cuda_context::cudaMallocHost(void **ptr, size_t size) {
    return callout::cudaMallocHost(ptr, size);
}

cudaError_t cuda_context::cudaMallocPitch(void **devPtr, size_t *pitch,
        size_t width, size_t height) {
    return callout::cudaMallocPitch(devPtr, pitch, width, height);
}

cudaError_t cuda_context::cudaMemcpy(void *dst, const void *src, size_t size,
        enum cudaMemcpyKind kind) {
    return callout::cudaMemcpy(dst, src, size, kind);
}

cudaError_t cuda_context::cudaMemcpy2D(void *dst, size_t dpitch,
        const void *src, size_t pitch, size_t width, size_t height,
        enum cudaMemcpyKind kind) {
    return callout::cudaMemcpy2D(dst, dpitch, src, pitch, width, height, kind);
}

cudaError_t cuda_context::cudaMemcpy2DAsync(void *dst, size_t dpitch,
        const void *src, size_t spitch, size_t width, size_t height,
        enum cudaMemcpyKind kind, cudaStream_t stream) {
    return callout::cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
        kind, stream);
}

cudaError_t cuda_context::cudaMemcpy2DArrayToArray(struct cudaArray *dst,
        size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src,
        size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height,
        enum cudaMemcpyKind kind) {
    return callout::cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src,
        wOffsetSrc, hOffsetSrc, width, height, kind);
}

cudaError_t cuda_context::cudaMemcpy2DFromArray(void *dst, size_t dpitch,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t width, size_t height, enum cudaMemcpyKind kind) {
    return callout::cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset,
        width, height, kind);
}

cudaError_t cuda_context::cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t width, size_t height, enum cudaMemcpyKind kind,
        cudaStream_t stream) {
    return callout::cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset,
        hOffset, width, height, kind, stream);
}

cudaError_t cuda_context::cudaMemcpy2DToArray(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t spitch,
        size_t width, size_t height, enum cudaMemcpyKind kind) {
    return callout::cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch,
        width, height, kind);
}

cudaError_t cuda_context::cudaMemcpy2DToArrayAsync(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t spitch,
        size_t width, size_t height, enum cudaMemcpyKind kind,
        cudaStream_t stream) {
    return callout::cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src,
        spitch, width, height, kind, stream);
}

cudaError_t cuda_context::cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
    return callout::cudaMemcpy3D(p);
}

cudaError_t cuda_context::cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
        cudaStream_t stream) {
    return callout::cudaMemcpy3DAsync(p, stream);
}

cudaError_t cuda_context::cudaMemcpy3DPeer(
        const struct cudaMemcpy3DPeerParms *p) {
    return callout::cudaMemcpy3DPeer(p);
}

cudaError_t cuda_context::cudaMemcpy3DPeerAsync(
        const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream) {
    return callout::cudaMemcpy3DPeerAsync(p, stream);
}

cudaError_t cuda_context::cudaMemcpyArrayToArray(struct cudaArray *dst,
        size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src,
        size_t wOffsetSrc, size_t hOffsetSrc, size_t count,
        enum cudaMemcpyKind kind) {
    return callout::cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src,
        wOffsetSrc, hOffsetSrc, count, kind);
}

cudaError_t cuda_context::cudaMemcpyAsync(void *dst, const void *src,
        size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    return callout::cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t cuda_context::cudaMemcpyFromArray(void *dst,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t count, enum cudaMemcpyKind kind) {
    return callout::cudaMemcpyFromArray(dst, src, wOffset, hOffset,
        count, kind);
}

cudaError_t cuda_context::cudaMemcpyFromArrayAsync(void *dst,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    return callout::cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset,
        count, kind, stream);
}

cudaError_t cuda_context::cudaMemcpyFromSymbol(void *dst,
        const char *symbol, size_t count, size_t offset,
        enum cudaMemcpyKind kind) {
    uint8_t * symbol_ptr;
    cudaError_t ret = cuda_context::cudaGetSymbolAddress(
        (void **) &symbol_ptr, symbol);
    if (ret != cudaSuccess) {
        return ret;
    }

    size_t symbol_size;
    ret = cuda_context::cudaGetSymbolSize(&symbol_size, symbol);
    if (ret != cudaSuccess) {
        return ret;
    }

    if (symbol_size < count + offset) {
        return cudaErrorInvalidValue;
    }

    switch (kind) {
        case cudaMemcpyDefault:
            /**
             * For some reason, this SIGSEGVs when run directly on CUDA.  We
             * emulate that behavior here.
             */
            raise(SIGSEGV);

            /* Fall through */
        case cudaMemcpyHostToDevice:
        case cudaMemcpyHostToHost:
        default:
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyDeviceToDevice:
            break;
    }

    return cudaMemcpy(dst, symbol_ptr + offset, count, kind);
}

cudaError_t cuda_context::cudaMemcpyFromSymbolAsync(void *dst,
        const char *symbol, size_t count, size_t offset,
        enum cudaMemcpyKind kind, cudaStream_t stream) {
    uint8_t * symbol_ptr;
    cudaError_t ret = cuda_context::cudaGetSymbolAddress(
        (void **) &symbol_ptr, symbol);
    if (ret != cudaSuccess) {
        return ret;
    }

    size_t symbol_size;
    ret = cuda_context::cudaGetSymbolSize(&symbol_size, symbol);
    if (ret != cudaSuccess) {
        return ret;
    }

    if (symbol_size < count + offset) {
        return cudaErrorInvalidValue;
    }

    switch (kind) {
        case cudaMemcpyDefault:
            /**
             * For some reason, this SIGSEGVs when run directly on CUDA.  We
             * emulate that behavior here.
             */
            raise(SIGSEGV);

            /* Fall through */
        case cudaMemcpyHostToDevice:
        case cudaMemcpyHostToHost:
        default:
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyDeviceToDevice:
            break;
    }

    return cudaMemcpyAsync(dst, symbol_ptr + offset, count, kind, stream);
}

cudaError_t cuda_context::cudaMemcpyPeer(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count) {
    return callout::cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
}

cudaError_t cuda_context::cudaMemcpyPeerAsync(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count, cudaStream_t stream) {
    return callout::cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice,
        count, stream);
}

cudaError_t cuda_context::cudaMemcpyToArray(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t count,
        enum cudaMemcpyKind kind) {
    return callout::cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
}

cudaError_t cuda_context::cudaMemcpyToArrayAsync(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t count,
        enum cudaMemcpyKind kind, cudaStream_t stream) {
    return callout::cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count,
        kind, stream);
}

cudaError_t cuda_context::cudaMemcpyToSymbol(const char *symbol,
        const void *src, size_t count, size_t offset,
        enum cudaMemcpyKind kind) {
    uint8_t * symbol_ptr;
    cudaError_t ret = cuda_context::cudaGetSymbolAddress(
        (void **) &symbol_ptr, symbol);
    if (ret != cudaSuccess) {
        return ret;
    }

    size_t symbol_size;
    ret = cuda_context::cudaGetSymbolSize(&symbol_size, symbol);
    if (ret != cudaSuccess) {
        return ret;
    }

    if (symbol_size < count + offset) {
        return cudaErrorInvalidValue;
    }

    switch (kind) {
        case cudaMemcpyDefault:
            /**
             * For some reason, this SIGSEGVs when run directly on CUDA.  We
             * emulate that behavior here.
             */
            raise(SIGSEGV);

            /* Fall through */
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToHost:
        default:
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyDeviceToDevice:
        case cudaMemcpyHostToDevice:
            break;
    }

    return cudaMemcpy(symbol_ptr + offset, src, count, kind);
}

cudaError_t cuda_context::cudaMemcpyToSymbolAsync(const char *symbol,
        const void *src, size_t count, size_t offset,
        enum cudaMemcpyKind kind, cudaStream_t stream) {
    uint8_t * symbol_ptr;
    cudaError_t ret = cuda_context::cudaGetSymbolAddress(
        (void **) &symbol_ptr, symbol);
    if (ret != cudaSuccess) {
        return ret;
    }

    size_t symbol_size;
    ret = cuda_context::cudaGetSymbolSize(&symbol_size, symbol);
    if (ret != cudaSuccess) {
        return ret;
    }

    if (symbol_size < count + offset) {
        return cudaErrorInvalidValue;
    }

    switch (kind) {
        case cudaMemcpyDefault:
            /**
             * For some reason, this SIGSEGVs when run directly on CUDA.  We
             * emulate that behavior here.
             */
            raise(SIGSEGV);

            /* Fall through */
        case cudaMemcpyHostToDevice:
        case cudaMemcpyHostToHost:
        default:
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyDeviceToDevice:
            break;
    }

    return cudaMemcpyAsync(symbol_ptr + offset, src, count, kind, stream);
}

cudaError_t cuda_context::cudaMemGetInfo(size_t *free, size_t *total) {
    return callout::cudaMemGetInfo(free, total);
}

cudaError_t cuda_context::cudaMemset(void *devPtr, int value, size_t count) {
    return setLastError(callout::cudaMemset(devPtr, value, count));
}

cudaError_t cuda_context::cudaMemset2D(void *devPtr, size_t pitch, int value,
        size_t width, size_t height) {
    return callout::cudaMemset2D(devPtr, pitch, value, width, height);
}

cudaError_t cuda_context::cudaMemset2DAsync(void *devPtr, size_t pitch,
        int value, size_t width, size_t height, cudaStream_t stream) {
    return callout::cudaMemset2DAsync(devPtr, pitch, value, width, height,
        stream);
}

cudaError_t cuda_context::cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr,
        int value, struct cudaExtent extent) {
    return callout::cudaMemset3D(pitchedDevPtr, value, extent);
}

cudaError_t cuda_context::cudaMemset3DAsync(
        struct cudaPitchedPtr pitchedDevPtr, int value,
        struct cudaExtent extent, cudaStream_t stream) {
    return callout::cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
}

cudaError_t cuda_context::cudaMemsetAsync(void *devPtr, int value,
        size_t count, cudaStream_t stream) {
    return callout::cudaMemsetAsync(devPtr, value, count, stream);
}

cudaError_t cuda_context::cudaPeekAtLastError() {
    return error_;
}

cudaError_t cuda_context::cudaPointerGetAttributes(struct
        cudaPointerAttributes *attributes, const void *ptr) {
    /**
     * CUDA is unhappy about mixing driver and runtime API calls.
     * As a result, calling cudaPointerGetAttributes on our own fails with
     * cudaErrorIncompatibleDriverContext.
     */
    cudaPointerAttributes attr;

    CUcontext ctx;
    CUmemorytype type;
    CUdeviceptr device_ptr;
    void * host_ptr;

    CUresult ret = cuPointerGetAttribute((void *) &ctx,
        CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr) ptr);
    switch (ret) {
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_SUCCESS:
            break;
        default:
            return cudaErrorNotYetImplemented;
    }

    ret = cuCtxPushCurrent(ctx);
    if (ret != CUDA_SUCCESS) {
        /* Popping is probably hopeless. */
        return cudaErrorNotYetImplemented;
    }

    CUdevice device;
    ret = cuCtxGetDevice(&device);
    if (ret != CUDA_SUCCESS) {
        cuCtxPopCurrent(&ctx);
        return cudaErrorInvalidValue;
    }

    attr.device = (int) device;

    ret = cuPointerGetAttribute((void *) &type,
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr) ptr);
    switch (ret) {
        case CUDA_ERROR_INVALID_CONTEXT:
            /** TODO **/
            return cudaErrorNotYetImplemented;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_SUCCESS:
            break;
        default:
            return cudaErrorNotYetImplemented;
    }

    switch (type) {
        case CU_MEMORYTYPE_HOST:
            attr.memoryType = cudaMemoryTypeHost;
            break;
        case CU_MEMORYTYPE_DEVICE:
            attr.memoryType = cudaMemoryTypeDevice;
            break;
        case CU_MEMORYTYPE_ARRAY:
            return cudaErrorNotYetImplemented;
        case CU_MEMORYTYPE_UNIFIED:
            return cudaErrorNotYetImplemented;
    }

    /**
     * Panoptes does not support mapped memory.
     */
    if (attr.memoryType == cudaMemoryTypeHost) {
        attr.devicePointer = NULL;

        ret = cuPointerGetAttribute((void *) &host_ptr,
            CU_POINTER_ATTRIBUTE_HOST_POINTER, (CUdeviceptr) ptr);
        switch (ret) {
            case CUDA_ERROR_INVALID_VALUE:
                attr.hostPointer = NULL;
                break;
            case CUDA_SUCCESS:
                attr.hostPointer = host_ptr;
                break;
            default:
                return cudaErrorNotYetImplemented;
        }
    } else {
        ret = cuPointerGetAttribute((void *) &device_ptr,
            CU_POINTER_ATTRIBUTE_DEVICE_POINTER, (CUdeviceptr) ptr);
        switch (ret) {
            case CUDA_ERROR_INVALID_CONTEXT:
                /** TODO **/
                return cudaErrorNotYetImplemented;
            case CUDA_ERROR_INVALID_VALUE:
                attr.devicePointer = NULL;
                break;
            case CUDA_SUCCESS:
                attr.devicePointer = (void *) device_ptr;
                break;
            default:
                return cudaErrorNotYetImplemented;
        }

        attr.hostPointer = NULL;
    }

    *attributes = attr;
    return cudaSuccess;
}

cudaError_t cuda_context::cudaRuntimeGetVersion(int *runtimeVersion) {
    if (runtimeVersion) {
        *runtimeVersion = runtime_version_;
        return cudaSuccess;
    } else {
        return cudaErrorInvalidValue;
    }
}

cudaError_t cuda_context::cudaSetDoubleForDevice(double *d) {
    return callout::cudaSetDoubleForDevice(d);
}

cudaError_t cuda_context::cudaSetDoubleForHost(double *d) {
    return callout::cudaSetDoubleForHost(d);
}

cudaError_t cuda_context::cudaStreamCreate(cudaStream_t *pStream) {
    return callout::cudaStreamCreate(pStream);
}

cudaError_t cuda_context::cudaStreamDestroy(cudaStream_t stream) {
    return callout::cudaStreamDestroy(stream);
}

cudaError_t cuda_context::cudaStreamQuery(cudaStream_t stream) {
    return callout::cudaStreamQuery(stream);
}

cudaError_t cuda_context::cudaStreamSynchronize(cudaStream_t stream) {
    return callout::cudaStreamSynchronize(stream);
}

cudaError_t cuda_context::cudaStreamWaitEvent(cudaStream_t stream,
        cudaEvent_t event, unsigned int flags) {
    return callout::cudaStreamWaitEvent(stream, event, flags);
}

cudaError_t cuda_context::cudaUnbindTexture(
        const struct textureReference *texref) {
    scoped_lock lock(mx_);

    using internal::modules_t;
    modules_t::texture_map_t & textures = modules_->textures;
    modules_t::texture_map_t::const_iterator it = textures.find(texref);
    if (it == textures.end()) {
        /* TODO */
        return cudaErrorInvalidTexture;
    }

    internal::module_t::texture_map_t::const_iterator jit =
        it->second->textures.find(texref);
    if (jit == it->second->textures.end()) {
        /* TODO */
        return cudaErrorInvalidTexture;
    }

    /* The driver API lacks a notion of unbinding, but verify the
     * texture was bound before. */
    if (!(jit->second->bound)) {
        char msg[128];
        int msgret = snprintf(msg, sizeof(msg),
            "cudaUnbindTexture called on texture reference %p that was "
            "never bound.", texref);
        // sizeof(msg) is small, so the cast is safe.
        assert(msgret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);
    }

    jit->second->bound = false;
    return cudaSuccess;
}

const char * cuda_context::get_entry_name(const char * entry) const {
    /* Find module. */
    using internal::modules_t;
    modules_t::function_map_t & functions = modules_->functions;
    modules_t::function_map_t::const_iterator fit = functions.find(entry);
    if (fit == functions.end()) {
        return NULL;
    }

    /* Find function within module. */
    internal::module_t::function_map_t::const_iterator config_it =
        fit->second->functions.find(entry);
    if (config_it == fit->second->functions.end()) {
        return NULL;
    }

    return config_it->second.deviceName;
}

bool cuda_context::load_device_info() {
    if (!(info_loaded_)) {
        cudaError_t ret =
            cuda_context::cudaGetDeviceProperties(&info_, device_);
        if (ret != cudaSuccess) {
            return false;
        }

        info_loaded_ = true;
    }

    return true;
}

cudaError_t cuda_context::setLastError(cudaError_t error) const {
    if (error != cudaSuccess) {
        error_ = error;
    }

    return error_;
}

void cuda_context::clear() { }

cuda_context::td & cuda_context::thread_data() {
    td * t = thread_data_.get();
    if (!(t)) {
        t = new td();
        thread_data_.reset(t);
    }
    return *t;
}

cuda_context::call_stack_t & cuda_context::call_stack() {
    return thread_data().call_stack;
}
