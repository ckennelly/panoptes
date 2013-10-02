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

#include <cassert>
#include <cstdio>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <panoptes/callout.h>

using namespace panoptes;

callout::callout() : libcudart(RTLD_NEXT) { }

callout::~callout() { }

callout & callout::instance() {
    static callout instance_;
    return instance_;
}

typedef cudaError_t (*cudaBindTexture_t)(size_t *,
    const struct textureReference *, const void *,
    const struct cudaChannelFormatDesc *, size_t);
typedef cudaError_t (*cudaChooseDevice_t)(int *, const struct cudaDeviceProp *);
typedef cudaError_t (*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
typedef cudaError_t (*cudaDeviceCanAccessPeer_t)(int *, int, int);
typedef cudaError_t (*cudaDeviceDisablePeerAccess_t)(int);
typedef cudaError_t (*cudaDeviceEnablePeerAccess_t)(int, unsigned int);
typedef cudaError_t (*cudaDeviceGetCacheConfig_t)(enum cudaFuncCache *);
typedef cudaError_t (*cudaDeviceReset_t)();
typedef cudaError_t (*cudaDeviceSetCacheConfig_t)(enum cudaFuncCache);
typedef cudaError_t (*cudaDeviceSynchronize_t)();
typedef cudaError_t (*cudaDriverGetVersion_t)(int *);
typedef cudaError_t (*cudaEventCreate_t)(cudaEvent_t *);
typedef cudaError_t (*cudaEventCreateWithFlags_t)(cudaEvent_t *, unsigned int);
typedef cudaError_t (*cudaEventDestroy_t)(cudaEvent_t);
typedef cudaError_t (*cudaEventElapsedTime_t)(float *, cudaEvent_t,
    cudaEvent_t);
typedef cudaError_t (*cudaEventQuery_t)(cudaEvent_t);
typedef cudaError_t (*cudaEventRecord_t)(cudaEvent_t, cudaStream_t);
typedef cudaError_t (*cudaEventSynchronize_t)(cudaEvent_t);
typedef cudaError_t (*cudaFree_t)(void *);
typedef cudaError_t (*cudaFreeArray_t)(struct cudaArray *);
typedef cudaError_t (*cudaFreeHost_t)(void *);
typedef cudaError_t (*cudaFuncSetCacheConfig_t)(const char *,
    enum cudaFuncCache);
typedef cudaError_t (*cudaGetDeviceCount_t)(int *);
typedef cudaError_t (*cudaGetDeviceProperties_t)(struct cudaDeviceProp *, int);
typedef cudaError_t (*cudaGetLastError_t)(void);
typedef cudaError_t (*cudaGetTextureReference_t)(
    const struct textureReference **, const char *);
typedef cudaError_t (*cudaHostAlloc_t)(void **, size_t , unsigned int);
typedef cudaError_t (*cudaHostGetDevicePointer_t)(void **, void *,
    unsigned int);
typedef cudaError_t (*cudaHostGetFlags_t)(unsigned int *, void *);
typedef cudaError_t (*cudaHostRegister_t)(void *, size_t, unsigned int);
typedef cudaError_t (*cudaHostUnregister_t)(void *);
typedef cudaError_t (*cudaLaunch_t)(const char *);
typedef cudaError_t (*cudaMalloc_t)(void **, size_t);
typedef cudaError_t (*cudaMalloc3D_t)(struct cudaPitchedPtr *,
    struct cudaExtent);
typedef cudaError_t (*cudaMalloc3DArray_t)(struct cudaArray**,
    const struct cudaChannelFormatDesc*, struct cudaExtent, unsigned int);
typedef cudaError_t (*cudaMallocArray_t)(struct cudaArray**,
    const struct cudaChannelFormatDesc*, size_t, size_t, unsigned int);
typedef cudaError_t (*cudaMallocHost_t)(void **, size_t);
typedef cudaError_t (*cudaMallocPitch_t)(void **, size_t *, size_t, size_t);
typedef cudaError_t (*cudaMemcpy_t)(void *, const void *, size_t,
    cudaMemcpyKind);
typedef cudaError_t (*cudaMemcpyAsync_t)(void *, const void *, size_t,
    cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*cudaMemcpyFromSymbol_t)(void *, const char *,
    size_t, size_t, enum cudaMemcpyKind);
typedef cudaError_t (*cudaMemcpyFromSymbolAsync_t)(void *, const char *,
    size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*cudaMemcpyPeer_t)(void *, int, const void *, int,
    size_t);
typedef cudaError_t (*cudaMemcpyPeerAsync_t)(void *, int, const void *, int,
    size_t, cudaStream_t);
typedef cudaError_t (*cudaMemGetInfo_t)(size_t *, size_t *);
typedef cudaError_t (*cudaMemset_t)(void *, int, size_t);
typedef cudaError_t (*cudaMemsetAsync_t)(void *, int, size_t, cudaStream_t);
typedef cudaError_t (*cudaPointerGetAttributes_t)(
    struct cudaPointerAttributes *, void *);
typedef cudaError_t (*cudaRuntimeGetVersion_t)(int *);
typedef cudaError_t (*cudaSetDevice_t)(int);
typedef cudaError_t (*cudaSetDeviceFlags_t)(unsigned int);
typedef cudaError_t (*cudaSetDoubleForDevice_t)(double *);
typedef cudaError_t (*cudaSetDoubleForHost_t)(double *);
typedef cudaError_t (*cudaSetupArgument_t)(const void *, size_t, size_t);
typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t *);
typedef cudaError_t (*cudaStreamDestroy_t)(cudaStream_t);
typedef cudaError_t (*cudaStreamQuery_t)(cudaStream_t);
typedef cudaError_t (*cudaStreamSynchronize_t)(cudaStream_t);
typedef cudaError_t (*cudaStreamWaitEvent_t)(cudaStream_t, cudaEvent_t,
    unsigned int);
typedef cudaError_t (*cudaThreadSynchronize_t)();
typedef cudaError_t (*cudaUnbindTexture_t)(const struct textureReference *);

typedef void   (*registerFunction_t)(void **, const char *,
    char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
typedef void** (*registerFatBinary_t)(void *);
typedef void   (*unregisterFatBinary_t)(void **);

cudaError_t callout::cudaBindTexture(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t size) {
    cudaBindTexture_t method = (cudaBindTexture_t)
        dlsym(callout::instance().libcudart, "cudaBindTexture");
    return method(offset, texref, devPtr, desc, size);
}

cudaError_t callout::cudaChooseDevice(int *device,
        const struct cudaDeviceProp *prop) {
    cudaChooseDevice_t method = (cudaChooseDevice_t)
        dlsym(callout::instance().libcudart, "cudaChooseDevice");
    return method(device, prop);
}

cudaError_t callout::cudaConfigureCall(dim3 gridDim, dim3
        blockDim, size_t sharedMem, cudaStream_t stream) {
    cudaConfigureCall_t method = (cudaConfigureCall_t)
        dlsym(callout::instance().libcudart, "cudaConfigureCall");
    return method(gridDim, blockDim, sharedMem, stream);
}

cudaError_t callout::cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
        int peerDevice) {
    cudaDeviceCanAccessPeer_t method = (cudaDeviceCanAccessPeer_t)
        dlsym(callout::instance().libcudart, "cudaDeviceCanAccessPeer");
    return method(canAccessPeer, device, peerDevice);
}

cudaError_t callout::cudaDeviceDisablePeerAccess(int peerDevice) {
    cudaDeviceDisablePeerAccess_t method = (cudaDeviceDisablePeerAccess_t)
        dlsym(callout::instance().libcudart, "cudaDeviceDisablePeerAccess");
    return method(peerDevice);
}

cudaError_t callout::cudaDeviceEnablePeerAccess(int peerDevice,
        unsigned int flags) {
    cudaDeviceEnablePeerAccess_t method = (cudaDeviceEnablePeerAccess_t)
        dlsym(callout::instance().libcudart, "cudaDeviceEnablePeerAccess");
    return method(peerDevice, flags);
}

cudaError_t callout::cudaDeviceGetCacheConfig(enum cudaFuncCache *
        pCacheConfig) {
    cudaDeviceGetCacheConfig_t method = (cudaDeviceGetCacheConfig_t) dlsym(
        callout::instance().libcudart, "cudaDeviceGetCacheConfig");
    return method(pCacheConfig);
}

cudaError_t callout::cudaDeviceReset() {
    cudaDeviceReset_t method = (cudaDeviceReset_t) dlsym(
        callout::instance().libcudart, "cudaDeviceReset");
    return method();
}

cudaError_t callout::cudaDeviceSetCacheConfig(enum cudaFuncCache
        cacheConfig) {
    cudaDeviceSetCacheConfig_t method = (cudaDeviceSetCacheConfig_t) dlsym(
        callout::instance().libcudart, "cudaDeviceSetCacheConfig");
    return method(cacheConfig);
}

cudaError_t callout::cudaDeviceSynchronize() {
    cudaDeviceSynchronize_t method = (cudaDeviceSynchronize_t) dlsym(
        callout::instance().libcudart, "cudaDeviceSynchronize");
    return method();
}

cudaError_t callout::cudaDriverGetVersion(int *driverVersion) {
    cudaDriverGetVersion_t method = (cudaDriverGetVersion_t) dlsym(
        callout::instance().libcudart, "cudaDriverGetVersion");
    return method(driverVersion);
}

cudaError_t callout::cudaEventCreate(cudaEvent_t *event) {
    cudaEventCreate_t method = (cudaEventCreate_t)
        dlsym(callout::instance().libcudart, "cudaEventCreate");
    return method(event);
}

cudaError_t callout::cudaEventCreateWithFlags(cudaEvent_t *event,
        unsigned int flags) {
    cudaEventCreateWithFlags_t method = (cudaEventCreateWithFlags_t)
        dlsym(callout::instance().libcudart, "cudaEventCreateWithFlags");
    return method(event, flags);
}

cudaError_t callout::cudaEventDestroy(cudaEvent_t event) {
    cudaEventDestroy_t method = (cudaEventDestroy_t)
        dlsym(callout::instance().libcudart, "cudaEventDestroy");
    return method(event);
}

cudaError_t callout::cudaEventElapsedTime(float *ms, cudaEvent_t start,
        cudaEvent_t end) {
    cudaEventElapsedTime_t method = (cudaEventElapsedTime_t)
        dlsym(callout::instance().libcudart, "cudaEventElapsedTime");
    return method(ms, start, end);
}

cudaError_t callout::cudaEventQuery(cudaEvent_t event) {
    cudaEventQuery_t method = (cudaEventQuery_t)
        dlsym(callout::instance().libcudart, "cudaEventQuery");
    return method(event);
}

cudaError_t callout::cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    cudaEventRecord_t method = (cudaEventRecord_t)
        dlsym(callout::instance().libcudart, "cudaEventRecord");
    return method(event, stream);
}

cudaError_t callout::cudaEventSynchronize(cudaEvent_t event) {
    cudaEventSynchronize_t method = (cudaEventSynchronize_t)
        dlsym(callout::instance().libcudart, "cudaEventSynchronize");
    return method(event);
}

cudaError_t callout::cudaFree(void *devPtr) {
    cudaFree_t method = (cudaFree_t)
        dlsym(callout::instance().libcudart, "cudaFree");
    return method(devPtr);
}

cudaError_t callout::cudaFreeArray(struct cudaArray * array) {
    cudaFreeArray_t method = (cudaFreeArray_t)
        dlsym(callout::instance().libcudart, "cudaFreeArray");
    return method(array);
}

cudaError_t callout::cudaFreeHost(void *ptr) {
    cudaFreeHost_t method = (cudaFreeHost_t)
        dlsym(callout::instance().libcudart,
        "cudaFreeHost");
    return method(ptr);
}

cudaError_t callout::cudaFuncSetCacheConfig(const char *func,
        enum cudaFuncCache cacheConfig) {
    cudaFuncSetCacheConfig_t method = (cudaFuncSetCacheConfig_t)
        dlsym(callout::instance().libcudart, "cudaFuncSetCacheConfig");
    return method(func, cacheConfig);
}

cudaError_t callout::cudaGetDeviceCount(int * count) {
    cudaGetDeviceCount_t method = (cudaGetDeviceCount_t)
        dlsym(callout::instance().libcudart, "cudaGetDeviceCount");
    return method(count);
}

cudaError_t callout::cudaGetDeviceProperties(struct cudaDeviceProp *prop,
        int device) {
    cudaGetDeviceProperties_t method = (cudaGetDeviceProperties_t)
        dlsym(callout::instance().libcudart, "cudaGetDeviceProperties");
    return method(prop, device);
}

cudaError_t callout::cudaGetLastError() {
    cudaGetLastError_t method = (cudaGetLastError_t)
        dlsym(callout::instance().libcudart, "cudaGetLastError");
    return method();
}

cudaError_t callout::cudaGetTextureReference(
        const struct textureReference **texref, const char *symbol) {
    cudaGetTextureReference_t method = (cudaGetTextureReference_t)
        dlsym(callout::instance().libcudart, "cudaGetTextureReference");
    return method(texref, symbol);
}

cudaError_t callout::cudaHostAlloc(void **pHost, size_t size,
        unsigned int flags) {
    cudaHostAlloc_t method = (cudaHostAlloc_t)
         dlsym(callout::instance().libcudart, "cudaHostAlloc");
    return method(pHost, size, flags);
}

cudaError_t callout::cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags) {
    cudaHostGetDevicePointer_t method = (cudaHostGetDevicePointer_t)
         dlsym(callout::instance().libcudart, "cudaHostGetDevicePointer");
    return method(pDevice, pHost, flags);
}

cudaError_t callout::cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    cudaHostGetFlags_t method = (cudaHostGetFlags_t)
         dlsym(callout::instance().libcudart, "cudaHostGetFlags");
    return method(pFlags, pHost);
}

cudaError_t callout::cudaHostRegister(void *ptr, size_t size,
        unsigned int flags) {
    cudaHostRegister_t method = (cudaHostRegister_t)
         dlsym(callout::instance().libcudart, "cudaHostRegister");
    return method(ptr, size, flags);
}

cudaError_t callout::cudaHostUnregister(void *ptr) {
    cudaHostUnregister_t method = (cudaHostUnregister_t)
         dlsym(callout::instance().libcudart, "cudaHostUnregister");
    return method(ptr);
}

cudaError_t callout::cudaLaunch(const char *entry) {
    cudaLaunch_t method = (cudaLaunch_t)
        dlsym(callout::instance().libcudart, "cudaLaunch");
    return method(entry);
}

cudaError_t callout::cudaMalloc(void **devPtr, size_t size) {
    cudaMalloc_t method = (cudaMalloc_t)
        dlsym(callout::instance().libcudart, "cudaMalloc");
    return method(devPtr, size);
}

cudaError_t callout::cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct
        cudaExtent extent) {
    cudaMalloc3D_t method = (cudaMalloc3D_t)
        dlsym(callout::instance().libcudart, "cudaMalloc3D");
    return method(pitchedDevPtr, extent);
}

cudaError_t callout::cudaMalloc3DArray(struct cudaArray** array, const struct
        cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int
        flags) {
    cudaMalloc3DArray_t method = (cudaMalloc3DArray_t)
        dlsym(callout::instance().libcudart, "cudaMalloc3DArray");
    return method(array, desc, extent, flags);
}

cudaError_t callout::cudaMallocArray(struct cudaArray **array,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags) {
    cudaMallocArray_t method = (cudaMallocArray_t)
        dlsym(callout::instance().libcudart, "cudaMallocArray");
    return method(array, desc, width, height, flags);
}

cudaError_t callout::cudaMallocHost(void **ptr, size_t size) {
    cudaMallocHost_t method = (cudaMallocHost_t)
        dlsym(callout::instance().libcudart, "cudaMallocHost");
    return method(ptr, size);
}

cudaError_t callout::cudaMallocPitch(void **devPtr, size_t *pitch,
        size_t width, size_t height) {
    cudaMallocPitch_t method = (cudaMallocPitch_t)
        dlsym(callout::instance().libcudart, "cudaMallocPitch");
    return method(devPtr, pitch, width, height);
}

cudaError_t callout::cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
    cudaMemcpy_t method = (cudaMemcpy_t)
        dlsym(callout::instance().libcudart, "cudaMemcpy");
    return method(dst, src, count, kind);
}

cudaError_t callout::cudaMemcpyAsync(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind, cudaStream_t stream) {
    cudaMemcpyAsync_t method = (cudaMemcpyAsync_t)
        dlsym(callout::instance().libcudart, "cudaMemcpyAsync");
    return method(dst, src, count, kind, stream);
}

cudaError_t callout::cudaMemcpyFromSymbol(void *dst, const char *symbol,
        size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaMemcpyFromSymbol_t method = (cudaMemcpyFromSymbol_t)
        dlsym(callout::instance().libcudart, "cudaMemcpyFromSymbol");
    return method(dst, symbol, count, offset, kind);
}

cudaError_t callout::cudaMemcpyFromSymbolAsync(void *dst, const char *symbol,
        size_t count, size_t offset, enum cudaMemcpyKind kind,
        cudaStream_t stream) {
    cudaMemcpyFromSymbolAsync_t method = (cudaMemcpyFromSymbolAsync_t)
        dlsym(callout::instance().libcudart, "cudaMemcpyFromSymbolAsync");
    return method(dst, symbol, count, offset, kind, stream);
}

cudaError_t callout::cudaMemcpyPeer(void *dst, int dstDevice, const void *src,
        int srcDevice, size_t count) {
    cudaMemcpyPeer_t method = (cudaMemcpyPeer_t)
        dlsym(callout::instance().libcudart, "cudaMemcpyPeer");
    return method(dst, dstDevice, src, srcDevice, count);
}

cudaError_t callout::cudaMemcpyPeerAsync(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count, cudaStream_t stream) {
    cudaMemcpyPeerAsync_t method = (cudaMemcpyPeerAsync_t)
        dlsym(callout::instance().libcudart, "cudaMemcpyPeerAsync");
    return method(dst, dstDevice, src, srcDevice, count, stream);
}

cudaError_t callout::cudaMemGetInfo(size_t *free, size_t *total) {
    cudaMemGetInfo_t method = (cudaMemGetInfo_t)
        dlsym(callout::instance().libcudart, "cudaMemGetInfo");
    return method(free, total);
}

cudaError_t callout::cudaMemset(void *devPtr, int value, size_t count) {
    cudaMemset_t method = (cudaMemset_t)
        dlsym(callout::instance().libcudart, "cudaMemset");
    return method(devPtr, value, count);
}

cudaError_t callout::cudaMemsetAsync(void *devPtr, int value, size_t count,
        cudaStream_t stream) {
    cudaMemsetAsync_t method = (cudaMemsetAsync_t)
        dlsym(callout::instance().libcudart, "cudaMemsetAsync");
    return method(devPtr, value, count, stream);
}

cudaError_t callout::cudaPointerGetAttributes(
        struct cudaPointerAttributes *attributes, void *ptr) {
    cudaPointerGetAttributes_t method = (cudaPointerGetAttributes_t) dlsym(
        callout::instance().libcudart, "cudaPointerGetAttributes");
    return method(attributes, ptr);
}

cudaError_t callout::cudaRuntimeGetVersion(int *runtimeVersion) {
    cudaRuntimeGetVersion_t method = (cudaRuntimeGetVersion_t)
        dlsym(callout::instance().libcudart, "cudaRuntimeGetVersion");
    return method(runtimeVersion);
}

cudaError_t callout::cudaSetDevice(int device) {
    cudaSetDevice_t method = (cudaSetDevice_t)
        dlsym(callout::instance().libcudart, "cudaSetDevice");
    return method(device);
}

cudaError_t callout::cudaSetDeviceFlags(unsigned int flags) {
    cudaSetDeviceFlags_t method = (cudaSetDeviceFlags_t)
        dlsym(callout::instance().libcudart, "cudaSetDeviceFlags");
    return method(flags);
}

cudaError_t callout::cudaSetDoubleForDevice(double *d) {
    cudaSetDoubleForDevice_t method = (cudaSetDoubleForDevice_t)
        dlsym(callout::instance().libcudart, "cudaSetDoubleForDevice");
    return method(d);
}

cudaError_t callout::cudaSetDoubleForHost(double *d) {
    cudaSetDoubleForHost_t method = (cudaSetDoubleForHost_t)
        dlsym(callout::instance().libcudart, "cudaSetDoubleForHost");
    return method(d);
}

cudaError_t callout::cudaSetupArgument(const void *arg, size_t size, size_t
        offset) {
    cudaSetupArgument_t method = (cudaSetupArgument_t)
        dlsym(callout::instance().libcudart, "cudaSetupArgument");
    return method(arg, size, offset);
}

cudaError_t callout::cudaStreamCreate(cudaStream_t *pStream) {
    cudaStreamCreate_t method = (cudaStreamCreate_t)
        dlsym(callout::instance().libcudart, "cudaStreamCreate");
    return method(pStream);
}

cudaError_t callout::cudaStreamDestroy(cudaStream_t stream) {
    cudaStreamDestroy_t method = (cudaStreamDestroy_t)
        dlsym(callout::instance().libcudart, "cudaStreamDestroy");
    return method(stream);
}

cudaError_t callout::cudaStreamQuery(cudaStream_t stream) {
    cudaStreamQuery_t method = (cudaStreamQuery_t)
        dlsym(callout::instance().libcudart, "cudaStreamQuery");
    return method(stream);
}

cudaError_t callout::cudaStreamSynchronize(cudaStream_t stream) {
    cudaStreamSynchronize_t method = (cudaStreamSynchronize_t)
        dlsym(callout::instance().libcudart, "cudaStreamSynchronize");
    return method(stream);
}

cudaError_t callout::cudaStreamWaitEvent(cudaStream_t stream,
        cudaEvent_t event, unsigned int flags) {
    cudaStreamWaitEvent_t method = (cudaStreamWaitEvent_t)
        dlsym(callout::instance().libcudart, "cudaStreamWaitEvent");
    return method(stream, event, flags);
}

cudaError_t callout::cudaThreadSynchronize() {
    cudaThreadSynchronize_t method = (cudaThreadSynchronize_t) dlsym(
        callout::instance().libcudart, "cudaThreadSynchronize");
    return method();
}

void callout::cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    registerFunction_t method = (registerFunction_t)
        dlsym(callout::instance().libcudart, "__cudaRegisterFunction");
    return method(fatCubinHandle, hostFun, deviceFun, deviceName,
        thread_limit, tid, bid, bDim, gDim, wSize);
}

void** callout::cudaRegisterFatBinary(void *fatCubin) {
    registerFatBinary_t method = (registerFatBinary_t)
        dlsym(callout::instance().libcudart, "__cudaRegisterFatBinary");
    return method(fatCubin);
}

void callout::cudaUnregisterFatBinary(void **fatCubinHandle) {
    unregisterFatBinary_t method = (unregisterFatBinary_t)
        dlsym(callout::instance().libcudart, "__cudaUnregisterFatBinary");
    return method(fatCubinHandle);
}

cudaError_t callout::cudaUnbindTexture(const struct textureReference *texref) {
    cudaUnbindTexture_t method = (cudaUnbindTexture_t) dlsym(
        callout::instance().libcudart, "cudaUnbindTexture");
    return method(texref);
}

