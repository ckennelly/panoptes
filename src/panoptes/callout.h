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

#ifndef __PANOPTES__CALLOUT_H_
#define __PANOPTES__CALLOUT_H_

#include <boost/utility.hpp>
#include <cuda_runtime.h>

namespace panoptes {

/**
 * This serves as a singleton for managing access to the underlying CUDA
 * library implementation.
 */
class callout : boost::noncopyable {
public:
    static void cudaRegisterFunction(void **, const char *, char *,
        const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
    static void** cudaRegisterFatBinary(void *);
    static void cudaUnregisterFatBinary(void **);

    static cudaError_t cudaBindSurfaceToArray(
        const struct surfaceReference *surfref, const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc);
    static cudaError_t cudaBindTexture(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t size);
    static cudaError_t cudaBindTexture2D(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        size_t pitch);
    static cudaError_t cudaBindTextureToArray(
        const struct textureReference *texref, const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc);
    static cudaError_t cudaChooseDevice(int *device,
        const struct cudaDeviceProp *prop);
    static cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream);
    static cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z,
        int w, enum cudaChannelFormatKind f);
    static cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
        int peerDevice);
    static cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
    static cudaError_t cudaDeviceEnablePeerAccess(int peerDevice,
        unsigned int flags);
    static cudaError_t cudaDeviceGetByPCIBusId(int *device, char *pciBusId);
    static cudaError_t cudaDeviceGetCacheConfig(
        enum cudaFuncCache *pCacheConfig);
    static cudaError_t cudaDeviceGetLimit(size_t *pValue,
        enum cudaLimit limit);
    static cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len,
        int device);
    static cudaError_t cudaDeviceSetCacheConfig(
        enum cudaFuncCache cacheConfig);
    static cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
    static cudaError_t cudaDriverGetVersion(int *driverVersion);
    static cudaError_t cudaDeviceReset(void);
    static cudaError_t cudaDeviceSynchronize(void);
    static cudaError_t cudaEventCreate(cudaEvent_t *event);
    static cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
        unsigned int flags);
    static cudaError_t cudaEventDestroy(cudaEvent_t event);
    static cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
        cudaEvent_t end);
    static cudaError_t cudaEventQuery(cudaEvent_t event);
    static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
    static cudaError_t cudaEventSynchronize(cudaEvent_t);
    static cudaError_t cudaFree(void *devPtr);
    static cudaError_t cudaFreeArray(struct cudaArray *array);
    static cudaError_t cudaFreeHost(void *ptr);
    static cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
        const char *func);
    static cudaError_t cudaFuncSetCacheConfig(const char *func,
        enum cudaFuncCache cacheConfig);
    static cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc,
        const struct cudaArray *array);
    static cudaError_t cudaGetDevice(int *device);
    static cudaError_t cudaGetDeviceCount(int *count);
    static cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,
        int device);
    const char * cudaGetErrorString(cudaError_t error);
    static cudaError_t cudaGetExportTable(const void **ppExportTable,
        const cudaUUID_t *pExportTableId);
    static cudaError_t cudaGetLastError(void);
    static cudaError_t cudaGetSurfaceReference(
        const struct surfaceReference **surfRef, const void *symbol);
    static cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol);
    static cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol);
    static cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
        const struct textureReference *texref);
    static cudaError_t cudaGetTextureReference(
        const struct textureReference **texref, const char *symbol);
    static cudaError_t cudaGraphicsMapResources(int count,
        cudaGraphicsResource_t *resources, cudaStream_t stream);
    static cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr,
        size_t *size, cudaGraphicsResource_t resource);
    static cudaError_t cudaGraphicsResourceSetMapFlags(
        cudaGraphicsResource_t resource, unsigned int flags);
    static cudaError_t cudaGraphicsSubResourceGetMappedArray(
        struct cudaArray **array, cudaGraphicsResource_t resource,
        unsigned int arrayIndex, unsigned int mipLevel);
    static cudaError_t cudaGraphicsUnmapResources(int count,
        cudaGraphicsResource_t *resources, cudaStream_t stream);
    static cudaError_t cudaGraphicsUnregisterResource(
        cudaGraphicsResource_t resource);
    static cudaError_t cudaHostAlloc(void **pHost, size_t size,
        unsigned int flags);
    static cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags);
    static cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
    static cudaError_t cudaHostRegister(void *ptr, size_t size,
        unsigned int flags);
    static cudaError_t cudaHostUnregister(void *ptr);
    #if CUDART_VERSION >= 4010 /* 4.1 */
    static cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
        cudaEvent_t event);
    static cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,
        cudaIpcEventHandle_t handle);
    static cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,
        void *devPtr);
    static cudaError_t cudaIpcOpenMemHandle(void **devPtr,
        cudaIpcMemHandle_t handle, unsigned int flags);
    static cudaError_t cudaIpcCloseMemHandle(void *devPtr);
    #endif
    static cudaError_t cudaMalloc(void **devPtr, size_t size);
    static cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr,
        struct cudaExtent extent);
    static cudaError_t cudaMalloc3DArray(struct cudaArray** array,
        const struct cudaChannelFormatDesc *desc, struct cudaExtent extent,
        unsigned int flags);
    static cudaError_t cudaMallocArray(struct cudaArray **array,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags);
    static cudaError_t cudaMallocHost(void **ptr, size_t size);
    static cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
        size_t width, size_t height);
    static cudaError_t cudaMemcpy(void *dst, const void *src, size_t size,
        enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
        size_t pitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
        const void *src, size_t spitch, size_t width, size_t height,
        enum cudaMemcpyKind kind, cudaStream_t stream);
    static cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst,
        size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src,
        size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height,
        enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t width, size_t height, enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t width, size_t height, enum cudaMemcpyKind kind,
        cudaStream_t stream);
    static cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t spitch,
        size_t width, size_t height, enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst,
        size_t wOffset, size_t hOFfset, const void *src, size_t spitch,
        size_t width, size_t height, enum cudaMemcpyKind kind,
        cudaStream_t stream);
    static cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
    static cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
        cudaStream_t stream);
    static cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);
    static cudaError_t cudaMemcpy3DPeerAsync(
        const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream);
    static cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst,
        size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src,
        size_t wOffsetSrc, size_t hOffsetSrc, size_t count,
        enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpyAsync(void *dst, const void *src,
        size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
    static cudaError_t cudaMemcpyFromArray(void *dst,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t count, enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpyFromArrayAsync(void *dst,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
    static cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol,
        size_t count, size_t offset, enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol,
        size_t count, size_t offset, enum cudaMemcpyKind kind,
        cudaStream_t stream);
    static cudaError_t cudaMemcpyPeer(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count);
    static cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count, cudaStream_t stream);
    static cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count,
        enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t count,
        enum cudaMemcpyKind kind, cudaStream_t stream);
    static cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src,
        size_t count, size_t offset, enum cudaMemcpyKind kind);
    static cudaError_t cudaMemcpyToSymbolAsync(const char *symbol,
        const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind,
        cudaStream_t stream);
    static cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
    static cudaError_t cudaMemset(void *devPtr, int value, size_t count);
    static cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value,
        size_t width, size_t height);
    static cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value,
        size_t width, size_t height, cudaStream_t stream);
    static cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr,
        int value, struct cudaExtent extent);
    static cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr,
        int value, struct cudaExtent extent, cudaStream_t stream);
    static cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
        cudaStream_t stream);
    static cudaError_t cudaLaunch(const char *entry);
    static cudaError_t cudaPeekAtLastError(void);
    static cudaError_t cudaPointerGetAttributes(
        struct cudaPointerAttributes *attributes, void *ptr);
    static cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
    static cudaError_t cudaSetDevice(int device);
    static cudaError_t cudaSetDeviceFlags(unsigned int flags);
    static cudaError_t cudaSetDoubleForDevice(double *d);
    static cudaError_t cudaSetDoubleForHost(double *d);
    static cudaError_t cudaSetupArgument(const void *arg, size_t size,
        size_t offset);
    static cudaError_t cudaSetValidDevices(int *device_arr, int len);
    static cudaError_t cudaStreamCreate(cudaStream_t *pStream);
    static cudaError_t cudaStreamDestroy(cudaStream_t stream);
    static cudaError_t cudaStreamQuery(cudaStream_t stream);
    static cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    static cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
        cudaEvent_t event, unsigned int flags);
    static cudaError_t cudaThreadExit(void);
    static cudaError_t cudaThreadGetCacheConfig(
        enum cudaFuncCache *pCacheConfig);
    static cudaError_t cudaThreadGetLimit(size_t *pValue,
        enum cudaLimit limit);
    static cudaError_t cudaThreadSetCacheConfig(
        enum cudaFuncCache cacheConfig);
    static cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value);
    static cudaError_t cudaThreadSynchronize();
    static cudaError_t cudaUnbindTexture(
        const struct textureReference *texref);

    ~callout();
protected:
    static callout & instance();
private:
    /* Private constructor */
    callout();

    void * libcudart;
};

} // end namespace panoptes

#endif // __PANOPTES__CALLOUT_H_
