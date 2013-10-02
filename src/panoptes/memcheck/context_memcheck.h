/**
 * Panoptes - A framework for detecting memory errors in GPU-based programs
 * Copyright (C) 2011 Chris Kennelly <chris@ckennelly.com>
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

#ifndef __PANOPTES__CONTEXT_MEMCHECK_H__
#define __PANOPTES__CONTEXT_MEMCHECK_H__

#include <boost/icl/interval_set.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_set.hpp>
#include <panoptes/context.h>
#include <panoptes/global_context.h>
#include <panoptes/gpu_stack.h>
#include <panoptes/memcheck/global_memcheck_state.h>
#include <ptx_io/ptx_ir.h>

namespace panoptes {

// Forward declaration
namespace internal {
    struct check_t;
    struct event_t;
    struct stream_t;
    struct array_t;
    struct texture_t;
    struct instrumentation_t;
}
class global_context_memcheck;

struct error_buffer_t {
    uint32_t data[1024];
};

class cuda_context_memcheck : public cuda_context {
public:
    typedef gpu_pool<adata_chunk> apool_t;
    typedef gpu_pool<vdata_chunk> vpool_t;

    explicit cuda_context_memcheck(
        global_context_memcheck * g, int device, unsigned int flags);
    virtual ~cuda_context_memcheck();

    /**
     * Quickly clears the state of the context.
     */
    virtual void clear();

/*
    virtual cudaError_t cudaBindSurfaceToArray(
        const struct surfaceReference *surfref, const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc); */
    virtual cudaError_t cudaBindTexture(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t size);
    virtual cudaError_t cudaBindTexture2D(size_t *offset, const struct
        textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        size_t pitch);
    virtual cudaError_t cudaBindTextureToArray(
        const struct textureReference *texref, const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc);
    virtual cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream);
    virtual cudaError_t cudaDeviceSynchronize();
    virtual cudaError_t cudaEventCreate(cudaEvent_t *event);
    virtual cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
        unsigned int flags);
    virtual cudaError_t cudaEventDestroy(cudaEvent_t event);
    virtual cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
        cudaEvent_t end);
    virtual cudaError_t cudaEventQuery(cudaEvent_t event);
    virtual cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
    virtual cudaError_t cudaEventSynchronize(cudaEvent_t);
    virtual cudaError_t cudaFree(void *devPtr);
    virtual cudaError_t cudaFreeArray(struct cudaArray *array);
    virtual cudaError_t cudaFreeHost(void *ptr);
    virtual cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,
        int device); /*
    virtual cudaError_t cudaGetSurfaceReference(
        const struct surfaceReference **surfRef, const char *symbol); */
    virtual cudaError_t cudaHostAlloc(void **pHost, size_t size,
        unsigned int flags);
    virtual cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags);
    virtual cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
    virtual cudaError_t cudaHostRegister(void *ptr, size_t size,
        unsigned int flags);
    virtual cudaError_t cudaHostUnregister(void *ptr);
    /*
    virtual cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
        cudaEvent_t event);
    virtual cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,
        cudaIpcEventHandle_t handle);
    virtual cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,
        void *devPtr);
    virtual cudaError_t cudaIpcOpenMemHandle(void **devPtr,
        cudaIpcMemHandle_t handle, unsigned int flags);
    virtual cudaError_t cudaIpcCloseMemHandle(void *devPtr); */
    virtual cudaError_t cudaLaunch(const void *entry);
    virtual cudaError_t cudaMalloc(void **devPtr, size_t size);
    virtual cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr,
        struct cudaExtent extent);
    virtual cudaError_t cudaMalloc3DArray(struct cudaArray** array,
        const struct cudaChannelFormatDesc *desc, struct cudaExtent extent,
        unsigned int flags);
    virtual cudaError_t cudaMallocArray(struct cudaArray **array, const struct
        cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags);
    virtual cudaError_t cudaMallocHost(void **ptr, size_t size);
    virtual cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
        size_t width, size_t height);
    virtual cudaError_t cudaMemcpy(void *dst, const void *src, size_t size,
        enum cudaMemcpyKind kind);
    /*
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
    virtual cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);
    virtual cudaError_t cudaMemcpy3DPeerAsync(
        const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream);
    virtual cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst,
        size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src,
        size_t wOffsetSrc, size_t hOffsetSrc, size_t count,
        enum cudaMemcpyKind kind); */
    virtual cudaError_t cudaMemcpyAsync(void *dst, const void *src,
        size_t count, enum cudaMemcpyKind kind, cudaStream_t stream); /*
    virtual cudaError_t cudaMemcpyFromArray(void *dst,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t count, enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpyFromArrayAsync(void *dst,
        const struct cudaArray *src, size_t wOffset, size_t hOffset,
        size_t count, enum cudaMemcpyKind kind, cudaStream_t stream); */
    virtual cudaError_t cudaMemcpyPeer(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count);
    virtual cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count, cudaStream_t stream); /*
    virtual cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count,
        enum cudaMemcpyKind kind);
    virtual cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t count,
        enum cudaMemcpyKind kind, cudaStream_t stream); */
    virtual cudaError_t cudaMemset(void *devPtr, int value, size_t count);
  /*virtual cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value,
        size_t width, size_t height);
    virtual cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value,
        size_t width, size_t height, cudaStream_t stream);
    virtual cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr,
        int value, struct cudaExtent extent);
    virtual cudaError_t cudaMemset3DAsync(
        struct cudaPitchedPtr pitchedDevPtr, int value,
        struct cudaExtent extent, cudaStream_t stream); */
    virtual cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
        cudaStream_t stream);
    virtual cudaError_t cudaStreamCreate(cudaStream_t *pStream);
    virtual cudaError_t cudaStreamDestroy(cudaStream_t stream);
    virtual cudaError_t cudaStreamQuery(cudaStream_t stream);
    virtual cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    virtual cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
        cudaEvent_t event, unsigned int flags);
    virtual cudaError_t cudaUnbindTexture(
        const struct textureReference *texref);
protected:
    /**
     * Implementation for cudaMemcpy/cudaMemcpyAsync.  The parameters are
     * similar to those of cudaMemcpyAsync, except the stream is passed in with
     * one further level of indirection.  If the pointer is NULL, the
     * implementation behaves according to the rules of cudaMemcpy.
     */
    cudaError_t cudaMemcpyImplementation(void *dst, const void *src,
        size_t count, enum cudaMemcpyKind kind, cudaStream_t *stream);

    cudaError_t cudaMemcpyPeerImplementation(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count, cudaStream_t *stream);

    /**
     * Checks whether a device access is valid.  Returns true if it is.
     * SIGSEGV is raised if signal is true.
     */
    bool check_access_device(const void * ptr, size_t len, bool signal) const;

    /**
     * Checks whether a host access is valid.  Returns true if it is.
     *
     * This requires the program to be running under Valgrind.  If Valgrind
     * is not available, the call always returns true.
     */
    bool check_access_host(const void * ptr, size_t len) const;

    /**
     * Checks whether a host access is pinned.  Returns true if it is and false
     * otherwise, filling in the offset which is not pinned.
     */
    bool check_host_pinned(const void * ptr, size_t len, size_t * offset) const;

    /**
     * Checks whether the pointer falls into a known device range.
     *
     * If block_ptr and block_offset are non-NULL and ptr is a known device
     * pointer, these values are filled in with the appropriate information.
     */
    bool is_device_pointer(const void * ptr, const void ** block_ptr,
        size_t * block_offset) const;

    /**
     * Adds a device allocation.
     *
     * must_free indicates that we should flag the block to Valgrind (if
     * present)
     */
    void add_device_allocation(const void * device_ptr, size_t size,
        bool must_free);

    /**
     * Removes a device allocation.
     */
    cudaError_t remove_device_allocation(const void * device_ptr);

    /**
     * Clear the validity bits of the device over a range.
     */
    bool validity_clear(const void * ptr, size_t len,
        internal::stream_t * stream);

    /**
     * Copy the validity bits on the device.
     */
    bool validity_copy(void * dst, const void * gpu, size_t len,
        internal::stream_t * stream);

    bool validity_copy(void * dst, cuda_context_memcheck * dstCtx,
        const void * src, const cuda_context_memcheck * srcCtx, size_t count,
        internal::stream_t * stream);

    /**
     * Copies the validity bits off the device and into Valgrind.  If Valgrind
     * is not running, this call is a no-op.
     *
     * Returns true if successful.
     */
    bool validity_download(void * host, const void * gpu, size_t len,
        internal::stream_t * stream) const;

    /**
     * Sets the validity bits on the device.
     */
    bool validity_set(const void * ptr, size_t len,
        internal::stream_t * stream);

    /**
     * Copies the validity bits out of Valgrind and onto the device.  If
     * Valgrind is not running, this call is a no-op.
     *
     * Returns true if successful.
     */
    bool validity_upload(void * host, const void * gpu, size_t len,
        internal::stream_t * stream);

    /**
     * Initializes the chunk handle to be no access, no validity.
     */
    void initialize_achunk(apool_t::handle_t * handle) const;
    void initialize_vchunk(vpool_t::handle_t * handle) const;

    /* Backend for computing validity textures details. */
    cudaError_t get_validity_texture(const struct textureReference *texref,
        const void *devPtr, size_t size, internal::texture_t **tex,
        const void **validity_ptr);

    /**
     * Cleans up the previously bound texture.
     */
    void release_texture(internal::texture_t * texture,
        const struct textureReference * texref);
    void load_validity_texref(internal::texture_t * texture,
        const struct textureReference * texref);
    void bind_validity_texref(internal::texture_t * texture,
        const struct textureReference * texref,
        const struct cudaChannelFormatDesc *desc,
        const void * validity_ptr, size_t size);
    cudaError_t bind_validity_texref2d(internal::texture_t * texture,
        const struct textureReference * texref,
        const struct cudaChannelFormatDesc *desc,
        const void * validity_ptr, size_t pitch, size_t height);

    global_context_memcheck * global();
    const global_context_memcheck * global() const;

    /**
     * Manipulates/queries recently freed list.
     */
    bool is_recently_freed(const void * ptr) const;
    void add_recent_free(const void * ptr, size_t size);
    void remove_recent_free(const void * ptr, size_t size);
private:
    /**
     * Master list pointing into chunks by their corresponding upper address
     * bits
     */
    host_gpu_vector<metadata_ptrs>    master_;
    std::vector<apool_t::handle_t *>  achunks_;
    std::vector<vpool_t::handle_t *>  vchunks_;
    state_ptr_t                       state_;

    /* Storage for auxillary chunk info (host-based) */
    struct chunk_aux_t {
        chunk_aux_t() : allocations(0), large_chunk(false) { }
        ~chunk_aux_t() { }

        /* Number of outstanding allocations for a chunk */
        unsigned allocations;

        /* Metadata for tracking large chunks. */
        bool large_chunk;
        size_t owner_index;
        vdata_chunk * hroot;
        vdata_chunk * groot;
        vpool_t::handle_t * handle;
    };
    /* We can have at most 1 << lg_chunk_bytes allocations at 1 byte each
     * in use for a chunk.  Debugging this overflow would be painful.
     *
     * "The C language does not exist; neither does Java, C++, and C#. While a
     * language may exist as an abstract idea, and even have a pile of paper (a
     * standard) purporting to define it, a standard is not a compiler. What
     * language do people write code in? The character strings accepted by
     * their compiler." (Bessey, et. al. CACM. Volume 53, Issue 2. Feb. 2010.)
     * In an ideal world, this static assertion would be placed in the above
     * class and simply take sizeof(allocations).  g++ happily accepts this
     * "C++" code; clang does not.  As fate would have it, this syntax is
     * probably illegal (however temptingly convienent it might be) for now
     * (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html).
     */
    BOOST_STATIC_ASSERT(sizeof(((chunk_aux_t *) 0)->allocations) * CHAR_BIT >
        lg_chunk_bytes);

    std::vector<chunk_aux_t         > chunks_aux_;

    /**
     * Allocated chunk vectors.
     */
    apool_t apool_;
    vpool_t vpool_;

    /**
     * We keep a single immutable handle as the default target for uninitialized
     * blocks.
     */
    apool_t::handle_t * default_achunk_;
    vpool_t::handle_t * default_vchunk_;

    /**
     * We keep an immutable handle as the default target for fully addressable
     * chunks.
     */
    apool_t::handle_t * initialized_achunk_;

    struct device_aux_t {
        size_t size;
        typedef boost::unordered_set<const struct textureReference *>
            binding_set_t;
        binding_set_t bindings;
        bool must_free;
    };

    /**
     * List of all allocations.
     */
    typedef std::map<const void *, device_aux_t> amap_t;
    amap_t device_allocations_;
    typedef std::map<const void *, size_t> aumap_t;
    aumap_t udevice_allocations_;

    /**
     * List of recently freed device allocations.
     */
    typedef boost::icl::interval_set<const uint8_t *> free_map_t;
    free_map_t recent_frees_;

    /**
     * Storage for auxillary host allocation information
     */
    enum host_allocation_t {
        allocation_hostalloc,
        allocation_mallochost,
        allocation_registered
    };

    struct host_aux_t {
        host_allocation_t allocation_type;
        unsigned int flags;
        size_t size;
    };

    typedef std::map<const void *, host_aux_t> ahmap_t;
    ahmap_t host_allocations_;

    /**
     * Occasionally, we're given a pointer that came out of a host allocation
     * is not the host allocation starting pointer itself.  We store the
     * upper bound information so we can retrieve the base pointer.
     */
    typedef std::map<const void *, size_t> acmap_t;
    acmap_t uhost_allocations_;

    /**
     * List of dimensions for opaque allocations.
     */
    typedef boost::unordered_map<const struct cudaArray *,
        internal::array_t *> oamap_t;
    oamap_t opaque_dimensions_;

    /**
     * We keep metadata on outstanding streams.
     */
    typedef boost::unordered_map<void **, internal::stream_t *> stream_map_t;
    stream_map_t streams_;

    /**
     * We keep metadata on outstanding, user-created events.
     */
    typedef boost::unordered_map<void **, internal::event_t *> event_map_t;
    event_map_t events_;

    /**
     * We keep metadata on outstanding, bound textures.
     */
    typedef boost::unordered_map<const struct textureReference *,
        internal::texture_t *> texture_map_t;
    texture_map_t bound_textures_;

    /**
     * Keep track of streams in-use for call stacks.
     */
    struct td {
        ~td();
        std::stack<internal::stream_t *> stream_stack;
    };
    boost::thread_specific_ptr<td> thread_data_;
    td & thread_data();

    /**
     * Error reporting buffers.
     */
    friend struct internal::check_t;
    gpu_stack<uint32_t> error_counts_;
    gpu_stack<error_buffer_t> error_buffers_;

    /**
     * Indicates the value received from RUNNING_UNDER_VALGRIND.
     */
    const unsigned valgrind_;

    /**
     * The page size.
     */
    const size_t pagesize_;

    /**
     * A mutex of our own.
     */
    mutable boost::mutex mx_;
};

} // end namespace panoptes

#endif // __PANOPTES__CONTEXT_MEMCHECK_H__
