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

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <crt/host_runtime.h>
#include <map>
#include <panoptes/backtrace.h>
#include <panoptes/callout.h>
#include <panoptes/context.h>
#include <panoptes/global_context.h>
#include <panoptes/interposer.h>
#include <panoptes/panoptes.h>
#include <valgrind/memcheck.h>

using namespace panoptes;

#ifdef __CPLUSPLUS
extern "C" {
#endif

int __panoptes__running_on_panoptes(void) {
    return 1;
}

void** __cudaRegisterFatBinary(void *fatCubin) {
    backtrace_t::instance().refresh();
    return global_context::instance().cudaRegisterFatBinary(fatCubin);
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    backtrace_t::instance().refresh();
    global_context::instance().cudaRegisterFunction(
        fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
        bid, bDim, gDim, wSize);
}

void __cudaRegisterTexture(void **fatCubinHandle,
        const struct textureReference *hostVar, const void **deviceAddress,
        const char *deviceName, int dim, int norm, int ext) {
    backtrace_t::instance().refresh();
    global_context::instance().cudaRegisterTexture(
        fatCubinHandle, hostVar, deviceAddress, deviceName,
        dim, norm, ext);
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global) {
    backtrace_t::instance().refresh();
    global_context::instance().cudaRegisterVar(fatCubinHandle, hostVar,
        deviceAddress, deviceName, ext, size, constant, global);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    backtrace_t::instance().refresh();
    global_context::instance().cudaUnregisterFatBinary(fatCubinHandle);
}

cudaError_t cudaBindSurfaceToArray(const struct surfaceReference *surfref,
        const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc) {
    backtrace_t::instance().refresh();
    return global_context::instance().context()->cudaBindSurfaceToArray(
        surfref, array, desc);
}

cudaError_t cudaBindTexture(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t size) {
    backtrace_t::instance().refresh();
    return global_context::instance().context()->cudaBindTexture(
        offset, texref, devPtr, desc, size);
}

cudaError_t cudaBindTexture2D(size_t *offset, const struct
        textureReference *texref, const void *devPtr, const struct
        cudaChannelFormatDesc *desc, size_t width, size_t height, size_t
        pitch) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaBindTexture2D(
        offset, texref, devPtr, desc, width, height, pitch);
}

cudaError_t cudaBindTextureToArray(const struct textureReference *texref,
        const struct cudaArray *array, const struct
        cudaChannelFormatDesc *desc) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaBindTextureToArray(
        texref, array, desc);
}

cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaChooseDevice(device, prop);
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaConfigureCall(gridDim,
        blockDim, sharedMem, stream);
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int
        peerDevice) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaDeviceCanAccessPeer(
        canAccessPeer, device, peerDevice);
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaDeviceDisablePeerAccess(peerDevice);
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaDeviceEnablePeerAccess(
        peerDevice, flags);
}

cudaError_t cudaDeviceGetByPCIBusId(int *device, char *pciBusId) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaDeviceGetByPCIBusId(device,
        pciBusId);
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaDeviceGetCacheConfig(
        pCacheConfig);
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaDeviceGetLimit(
        pValue, limit);
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaDeviceGetPCIBusId(pciBusId,
        len, device);
}

cudaError_t cudaDeviceReset() {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaDeviceReset();
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaDeviceSetCacheConfig(
        cacheConfig);
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaDeviceSetLimit(
        limit, value);
}

cudaError_t cudaDeviceSynchronize() {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaDeviceSynchronize();
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaDriverGetVersion(
        driverVersion);
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaEventCreate(event);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaEventCreateWithFlags(
        event, flags);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaEventDestroy(event);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
        cudaEvent_t end) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaEventElapsedTime(
        ms, start, end);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaEventQuery(event);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaEventRecord(event,
        stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaEventSynchronize(event);
}

cudaError_t cudaFree(void *devPtr) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaFree(devPtr);
}

cudaError_t cudaFreeArray(struct cudaArray *array) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaFreeArray(array);
}

cudaError_t cudaFreeHost(void *ptr) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaFreeHost(ptr);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const
        void *func) {
#else
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const
        char *func) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaFuncGetAttributes(
        attr, func);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache
        cacheConfig) {
#else
cudaError_t cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache
        cacheConfig) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaFuncSetCacheConfig(
        func, cacheConfig);
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const
        struct cudaArray *array) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGetChannelDesc(desc,
        array);
}

cudaError_t cudaGetDevice(int *device) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaGetDevice(device);
}

cudaError_t cudaGetDeviceCount(int *count) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaGetDeviceCount(count);
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGetDeviceProperties(
        prop, device);
}

/**
 * Not implemented:
 *
 * const char * cudaGetErrorString(cudaError_t error);
 */

cudaError_t cudaGetExportTable(const void **ppExportTable, const cudaUUID_t
        *pExportTableId) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGetExportTable(
        ppExportTable, pExportTableId);
}

cudaError_t cudaGetLastError(void) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGetLastError();
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaGetSurfaceReference(const struct surfaceReference **surfRef,
        const void *symbol) {
#else
cudaError_t cudaGetSurfaceReference(const struct surfaceReference **surfRef,
        const char *symbol) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGetSurfaceReference(
        surfRef, symbol);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
#else
cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGetSymbolAddress(
        devPtr, symbol);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
#else
cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGetSymbolSize(size,
        symbol);
}

cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, const struct
        textureReference *texref) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGetTextureAlignmentOffset(
        offset, texref);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaGetTextureReference(const struct textureReference **texref,
        const void *symbol) {
#else
cudaError_t cudaGetTextureReference(const struct textureReference **texref,
        const char *symbol) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGetTextureReference(
        texref, symbol);
}

cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t
        *resources, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGraphicsMapResources(
        count, resources, stream);
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size,
        cudaGraphicsResource_t resource) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->
        cudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource,
        unsigned int flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->
        cudaGraphicsResourceSetMapFlags(resource, flags);
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(struct cudaArray **array,
        cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int
        mipLevel) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->
        cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex,
        mipLevel);
}

cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t
        *resources, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGraphicsUnmapResources(
        count, resources, stream);
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaGraphicsUnregisterResource(
       resource);
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaHostAlloc(pHost, size,
        flags);
}

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaHostGetDevicePointer(
        pDevice, pHost, flags);
}

cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaHostGetFlags(pFlags,
        pHost);
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaHostRegister(ptr, size,
        flags);
}

cudaError_t cudaHostUnregister(void *ptr) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaHostUnregister(ptr);
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
        cudaEvent_t event) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->
        cudaIpcGetEventHandle(handle, event);
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,
        cudaIpcEventHandle_t handle) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->
        cudaIpcOpenEventHandle(event, handle);
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->
        cudaIpcGetMemHandle(handle, devPtr);
}

cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle,
        unsigned int flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->
        cudaIpcOpenMemHandle(devPtr, handle, flags);
}

cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaIpcCloseMemHandle(devPtr);
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMalloc(devPtr, size);
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct
        cudaExtent extent) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMalloc3D(
        pitchedDevPtr, extent);
}

cudaError_t cudaMalloc3DArray(struct cudaArray** array, const struct
        cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int
        flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMalloc3DArray(array,
        desc, extent, flags);
}

cudaError_t cudaMallocArray(struct cudaArray **array, const struct
        cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int
        flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMallocArray(array,
        desc, width, height, flags);
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMallocHost(ptr, size);
}

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width,
        size_t height) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMallocPitch(devPtr,
        pitch, width, height);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t size,
        enum cudaMemcpyKind kind) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy(dst, src, size,
        kind);
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t
        pitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy2D(dst, dpitch,
        src, pitch, width, height, kind);
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
        cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy2DAsync(dst,
        dpitch, src, spitch, width, height, kind, stream);
}

cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind
        kind) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy2DArrayToArray(dst,
        wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height,
        kind);
}

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct
        cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t
        height, enum cudaMemcpyKind kind) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy2DFromArray(dst,
        dpitch, src, wOffset, hOffset, width, height, kind);
}

cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct
        cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t
        height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy2DFromArrayAsync(
        dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
}

cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width, size_t
        height, enum cudaMemcpyKind kind) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy2DToArray(dst,
        wOffset, hOffset, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width, size_t
        height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy2DToArrayAsync(
        dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy3D(p);
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
        cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy3DAsync(p, stream);
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy3DPeer(p);
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p,
        cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpy3DPeerAsync(p,
        stream);
}

cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyArrayToArray(dst,
        wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
        enum cudaMemcpyKind kind, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyAsync(dst, src,
        count, kind, stream);
}

cudaError_t cudaMemcpyFromArray(void *dst, const struct cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count,
        enum cudaMemcpyKind kind) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyFromArray(dst, src,
        wOffset, hOffset, count, kind);
}

cudaError_t cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind,
        cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyFromArrayAsync(
        dst, src, wOffset, hOffset, count, kind, stream);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count,
        size_t offset, enum cudaMemcpyKind kind) {
#else
cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count,
        size_t offset, enum cudaMemcpyKind kind) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyFromSymbol(dst,
        symbol, count, offset, kind);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t
        count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
#else
cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t
        count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyFromSymbolAsync(
        dst, symbol, count, offset, kind, stream);
}

cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int
        srcDevice, size_t count) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyPeer(dst,
        dstDevice, src, srcDevice, count);
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int
        srcDevice, size_t count, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyPeerAsync(dst,
        dstDevice, src, srcDevice, count, stream);
}

cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t
        hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyToArray(dst,
        wOffset, hOffset, src, count, kind);
}

cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind
        kind, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyToArrayAsync(dst,
        wOffset, hOffset, src, count, kind, stream);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t
        count, size_t offset, enum cudaMemcpyKind kind) {
#else
cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t
        count, size_t offset, enum cudaMemcpyKind kind) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyToSymbol(symbol,
        src, count, offset, kind);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
        size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t
        stream) {
#else
cudaError_t cudaMemcpyToSymbolAsync(const char *symbol, const void *src,
        size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t
        stream) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemcpyToSymbolAsync(
        symbol, src, count, offset, kind, stream);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemGetInfo(free, total);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemset(devPtr, value,
        count);
}

cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t
        width, size_t height) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemset2D(devPtr, pitch,
        value, width, height);
}

cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t
        width, size_t height, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemset2DAsync(devPtr,
        pitch, value, width, height, stream);
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value,
        struct cudaExtent extent) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemset3D(pitchedDevPtr,
        value, extent);
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value,
        struct cudaExtent extent, cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemset3DAsync(
        pitchedDevPtr, value, extent, stream);
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
        cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaMemsetAsync(devPtr,
        value, count, stream);
}

#if CUDART_VERSION >= 5000 /* 5.0 */
cudaError_t cudaLaunch(const void *entry) {
#else
cudaError_t cudaLaunch(const char *entry) {
#endif
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaLaunch(entry);
}

cudaError_t cudaPeekAtLastError(void) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaPeekAtLastError();
}

/**
 * Legacy, pre-CUDA 4.1
 */
#if CUDA_VERSION >= 4010 /* 4.1 */
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *, void *);
#endif

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
        void *ptr) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaPointerGetAttributes(
        attributes, ptr);
}

#if CUDA_VERSION >= 4010 /* 4.1 */
/**
 * CUDA 4.1 version
 */
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
        const void *ptr) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaPointerGetAttributes(
        attributes, ptr);
}
#endif

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaRuntimeGetVersion(
        runtimeVersion);
}

cudaError_t cudaSetDevice(int device) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaSetDevice(device);
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    backtrace_t::instance().refresh();
    return global_context::instance().cudaSetDeviceFlags(flags);
}

cudaError_t cudaSetDoubleForDevice(double *d) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaSetDoubleForDevice(d);
}

cudaError_t cudaSetDoubleForHost(double *d) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaSetDoubleForHost(d);
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaSetupArgument(arg, size,
        offset);
}

cudaError_t cudaSetValidDevices(int *device_arr, int len) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaSetValidDevices(device_arr, len);
}

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaStreamCreate(pStream);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaStreamDestroy(stream);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaStreamQuery(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaStreamSynchronize(stream);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
        unsigned int flags) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaStreamWaitEvent(stream,
        event, flags);
}

cudaError_t cudaThreadExit(void) {
    backtrace_t::instance().refresh();

    return global_context::instance().cudaThreadExit();
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaThreadGetCacheConfig(
        pCacheConfig);
}

cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaThreadGetLimit(pValue,
        limit);
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaThreadSetCacheConfig(
        cacheConfig);
}

cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaThreadSetLimit(limit,
        value);
}

cudaError_t cudaThreadSynchronize() {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaThreadSynchronize();
}

cudaError_t cudaUnbindTexture(const struct textureReference *texref) {
    backtrace_t::instance().refresh();

    return global_context::instance().context()->cudaUnbindTexture(texref);
}

#ifdef __CPLUSPLUS
}
#endif
