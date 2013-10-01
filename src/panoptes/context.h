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

#ifndef __PANOPTES__CONTEXT_H__
#define __PANOPTES__CONTEXT_H__

#include <boost/thread/mutex.hpp>
#include <boost/thread/tss.hpp>
#include <boost/unordered_map.hpp>
#include <cuda.h>
#include <panoptes/host_gpu_vector.h>
#include <panoptes/gpu_pool.h>
#include <panoptes/metadata.h>
#include <stack>

namespace panoptes {

namespace internal {
    // Forward declaration
    struct modules_t;
    struct module_t;
    struct arg_t;
    struct call_t;
}

struct ptx_t;
class global_context;

class cuda_context {
public:
    /**
     * Adds the kernel parameters to the call stack.
     */
    virtual cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream);

    /**
     * Launches the kernel.
     */
    virtual cudaError_t cudaLaunch(const void *entry);

    /**
     * Provides cudaSetupArgument functionality.
     */
    virtual cudaError_t cudaSetupArgument(const void *arg, size_t size,
        size_t offset);

    /**
     * Synchronizes all threads in this context.
     */
    virtual cudaError_t cudaThreadSynchronize();

    virtual cudaError_t cudaBindSurfaceToArray(
        const struct surfaceReference *surfref, const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc);
    virtual cudaError_t cudaBindTexture(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t size);
    virtual cudaError_t cudaBindTexture2D(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        size_t pitch);
    virtual cudaError_t cudaBindTextureToArray(
        const struct textureReference *texref, const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc);
    virtual cudaError_t cudaChooseDevice(int *device,
        const struct cudaDeviceProp *prop);
    virtual cudaError_t cudaDeviceGetCacheConfig(
        enum cudaFuncCache *pCacheConfig);
    virtual cudaError_t cudaDeviceGetLimit(size_t *pValue,
        enum cudaLimit limit);
    virtual cudaError_t cudaDeviceSetCacheConfig(
        enum cudaFuncCache cacheConfig);
    virtual cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
    virtual cudaError_t cudaDeviceSynchronize();
    virtual cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z,
        int w, enum cudaChannelFormatKind f);
    virtual cudaError_t cudaDriverGetVersion(int *driverVersion);
    virtual cudaError_t cudaEventCreate(cudaEvent_t *event);
    virtual cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
        unsigned int flags);
    virtual cudaError_t cudaEventDestroy(cudaEvent_t event);
    virtual cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
        cudaEvent_t end);
    virtual cudaError_t cudaEventQuery(cudaEvent_t event);
    virtual cudaError_t cudaEventRecord(cudaEvent_t event,
        cudaStream_t stream);
    virtual cudaError_t cudaEventSynchronize(cudaEvent_t);
    virtual cudaError_t cudaFree(void *devPtr);
    virtual cudaError_t cudaFreeArray(struct cudaArray *array);
    virtual cudaError_t cudaFreeHost(void *ptr);
    virtual cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
        const void *func);
    virtual cudaError_t cudaFuncSetCacheConfig(const void *func,
        enum cudaFuncCache cacheConfig);
    virtual cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc,
        const struct cudaArray *array);
    virtual cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,
        int device);
    virtual cudaError_t cudaGetExportTable(const void **ppExportTable,
        const cudaUUID_t *pExportTableId);
    virtual cudaError_t cudaGetLastError();
    virtual cudaError_t cudaGetSurfaceReference(
        const struct surfaceReference **surfRef, const void *symbol);
    virtual cudaError_t cudaGetSymbolAddress(void **devPtr,
        const void *symbol);
    virtual cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
    virtual cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
        const struct textureReference *texref);
    virtual cudaError_t cudaGetTextureReference(
        const struct textureReference **texref, const void *symbol);
    virtual cudaError_t cudaGraphicsMapResources(int count,
        cudaGraphicsResource_t *resources, cudaStream_t stream);
    virtual cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr,
        size_t *size, cudaGraphicsResource_t resource);
    virtual cudaError_t cudaGraphicsResourceSetMapFlags(
        cudaGraphicsResource_t resource, unsigned int flags);
    virtual cudaError_t cudaGraphicsSubResourceGetMappedArray(
        struct cudaArray **array, cudaGraphicsResource_t resource,
        unsigned int arrayIndex, unsigned int mipLevel);
    virtual cudaError_t cudaGraphicsUnmapResources(int count,
        cudaGraphicsResource_t *resources, cudaStream_t stream);
    virtual cudaError_t cudaGraphicsUnregisterResource(
        cudaGraphicsResource_t resource);
    virtual cudaError_t cudaHostAlloc(void **pHost, size_t size,
        unsigned int flags);
    virtual cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags);
    virtual cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
    virtual cudaError_t cudaHostRegister(void *ptr, size_t size,
        unsigned int flags);
    virtual cudaError_t cudaHostUnregister(void *ptr);
    virtual cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
        cudaEvent_t event);
    virtual cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,
        cudaIpcEventHandle_t handle);
    virtual cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,
        void *devPtr);
    virtual cudaError_t cudaIpcOpenMemHandle(void **devPtr,
        cudaIpcMemHandle_t handle, unsigned int flags);
    virtual cudaError_t cudaIpcCloseMemHandle(void *devPtr);
    virtual cudaError_t cudaMalloc(void **devPtr, size_t size);
    virtual cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr,
        struct cudaExtent extent);
    virtual cudaError_t cudaMalloc3DArray(struct cudaArray** array,
        const struct cudaChannelFormatDesc *desc, struct cudaExtent extent,
        unsigned int flags);
    virtual cudaError_t cudaMallocArray(struct cudaArray **array,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags);
    virtual cudaError_t cudaMallocHost(void **ptr, size_t size);
    virtual cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
        size_t width, size_t height);
    virtual cudaError_t cudaMemcpy(void *dst, const void *src, size_t size,
        enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
        size_t pitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
        const void *src, size_t spitch, size_t width, size_t height,
        enum cudaMemcpyKind kind, cudaStream_t stream);
    virtual cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst,
        size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src,
        size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height,
        enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t width, size_t height, enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t width, size_t height, enum cudaMemcpyKind kind,
        cudaStream_t stream);
    virtual cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t spitch,
        size_t width, size_t height, enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst,
        size_t wOffset, size_t hOFfset, const void *src, size_t spitch,
        size_t width, size_t height, enum cudaMemcpyKind kind,
        cudaStream_t stream);
    virtual cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
    virtual cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
        cudaStream_t stream);
    virtual cudaError_t cudaMemcpy3DPeer(
        const struct cudaMemcpy3DPeerParms *p);
    virtual cudaError_t cudaMemcpy3DPeerAsync(
        const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream);
    virtual cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst,
        size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src,
        size_t wOffsetSrc, size_t hOffsetSrc, size_t count,
        enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpyAsync(void *dst, const void *src,
        size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
    virtual cudaError_t cudaMemcpyFromArray(void *dst,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t count, enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpyFromArrayAsync(void *dst,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
    virtual cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
        size_t count, size_t offset, enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpyFromSymbolAsync(void *dst,
        const void *symbol, size_t count, size_t offset,
        enum cudaMemcpyKind kind, cudaStream_t stream);
    virtual cudaError_t cudaMemcpyPeer(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count);
    virtual cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count, cudaStream_t stream);
    virtual cudaError_t cudaMemcpyToArray(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t count,
        enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t count,
        enum cudaMemcpyKind kind, cudaStream_t stream);
    virtual cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
        size_t count, size_t offset, enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpyToSymbolAsync(const void *symbol,
        const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind,
        cudaStream_t stream);
    virtual cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
    virtual cudaError_t cudaMemset(void *devPtr, int value, size_t count);
    virtual cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value,
        size_t width, size_t height);
    virtual cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch,
        int value, size_t width, size_t height, cudaStream_t stream);
    virtual cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr,
        int value, struct cudaExtent extent);
    virtual cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr,
        int value, struct cudaExtent extent, cudaStream_t stream);
    virtual cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
        cudaStream_t stream);
    virtual cudaError_t cudaPeekAtLastError();
    virtual cudaError_t cudaPointerGetAttributes(
        struct cudaPointerAttributes *attributes, const void *ptr);
    virtual cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
    virtual cudaError_t cudaSetDoubleForDevice(double *d);
    virtual cudaError_t cudaSetDoubleForHost(double *d);
    virtual cudaError_t cudaStreamCreate(cudaStream_t *pStream);
    virtual cudaError_t cudaStreamDestroy(cudaStream_t stream);
    virtual cudaError_t cudaStreamQuery(cudaStream_t stream);
    virtual cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    virtual cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
        cudaEvent_t event, unsigned int flags);
    virtual cudaError_t cudaThreadGetCacheConfig(
        enum cudaFuncCache *pCacheConfig);
    virtual cudaError_t cudaThreadGetLimit(size_t *pValue,
        enum cudaLimit limit);
    virtual cudaError_t cudaThreadSetCacheConfig(
        enum cudaFuncCache cacheConfig);
    virtual cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value);
    virtual cudaError_t cudaUnbindTexture(
        const struct textureReference *texref);

    cuda_context(global_context * ctx, int device, unsigned int flags);
    virtual ~cuda_context();
    virtual void clear();
protected:
    /**
     * Loads device information.
     */
    bool load_device_info();

    /**
     * Device information.
     */
    bool info_loaded_;
    struct cudaDeviceProp info_;

    /**
     * Sets the thread's last error.
     */
    friend class global_context;
    cudaError_t setLastError(cudaError_t error) const;

    /**
     * Returns the user readable name of the function given its address.
     */
    const char * get_entry_name(const void * entry) const;
protected:
    /**
     * This context's device
     */
    const int device_;

    /**
     * Call argument stack
     */
    typedef std::stack<internal::call_t *> call_stack_t;
    struct td {
        ~td();
        call_stack_t call_stack;
    };
private:
    boost::thread_specific_ptr<td> thread_data_;
protected:
    td & thread_data();
    call_stack_t & call_stack();

    /**
     * The cuda_context has state which needs to be protected against
     * concurrent updates.
     */
    mutable boost::mutex mx_;

    /**
     * The CUDA runtime version.
     */
    int runtime_version_;
    int driver_version_;

    global_context * global() { return global_; }
    const global_context * global() const { return global_; }

    internal::modules_t * modules_;
private:
    global_context * const global_;
    mutable cudaError_t error_;

    /**
     * The raw context
     */
    CUcontext ctx_;
}; // end class cuda_context

} // end namespace panoptes

#endif // __PANOPTES__CONTEXT_H__
