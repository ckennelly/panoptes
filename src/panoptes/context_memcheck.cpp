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

#include <boost/scoped_array.hpp>
#include <cstdio>
#include <cxxabi.h>
#include <list>
#include <panoptes/backtrace.h>
#include <panoptes/context_internal.h>
#include <panoptes/context_memcheck_internal.h>
#include <panoptes/context_memcheck.h>
#include <panoptes/global_context_memcheck.h>
#include <panoptes/logger.h>
#include <panoptes/ptx_formatter.h>
#include <panoptes/utilities.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <valgrind/memcheck.h>

using namespace panoptes;
using internal::__master_symbol;
using internal::create_handle;
using internal::free_handle;
using internal::max_errors;

typedef boost::unique_lock<boost::mutex> scoped_lock;
typedef global_context_memcheck global_t;

/**
 * cuda_context_memcheck does not provide:
 *  cudaGetSymbolAddress
 *  cudaGetSymbolSize
 *      These are implemented in cuda_context and interact with the loaded
 *      modules
 *
 *  cudaGetTextureAlignmentOffset
 *  cudaGetTextureReference
 *
 *  cudaPointerGetAttributes
 *      We do a few calls in cuda_context to work around getting interference
 *      from driver/runtime API mixing.  While we could provide more
 *      information here, there is no way of exposing it without changing the
 *      outward behavior of the system.
 *
 *  cudaMemcpyFromSymbol
 *  cudaMemcpyFromSymbolAsync
 *  cudaMemcpyToSymbol
 *  cudaMemcpyToSymbolAsync
 *      These are decomposed into a symbol lookup and a memcpy within
 *      cuda_context.
 *
 *  cudaMemGetInfo
 *      This is passed on to the CUDA library by cuda_context
 *
 *  cudaRuntimeGetVersion
 *      This is implemented by cuda_context.  There's no need to change the
 *      outward behavior of the system.
 */

namespace {
    enum array_allocator_t {
        array_malloc,
        array_malloc3d
    };

    const char * array_allocator_name(array_allocator_t t) {
        static const char str_malloc[]     = "cudaMallocArray";
        static const char str_malloc3d[]   = "cudaMalloc3DArray";

        switch (t) {
            case array_malloc:
                return str_malloc;
            case array_malloc3d:
                return str_malloc3d;
        }

        assert(0 && "Invalid array_allocator_t.");
        return NULL;
    }
}

namespace panoptes {
namespace internal {
struct check_t;
}
}

struct internal::array_t : boost::noncopyable {
    explicit array_t(struct cudaChannelFormatDesc);
    array_t(struct cudaChannelFormatDesc, struct cudaExtent);
    ~array_t();

    struct cudaChannelFormatDesc desc;
    size_t x, y, z;
    array_allocator_t allocator;
    void * validity;
    backtrace_t allocation_bt;

    typedef boost::unordered_set<const struct textureReference *>
        binding_set_t;
    binding_set_t bindings;

    size_t size() const;
};

struct internal::check_t {
    check_t(cuda_context_memcheck * c, const char * ename);
    virtual ~check_t();

    virtual cudaError_t check(cudaError_t);

    cuda_context_memcheck * const context;
    const char            * const entry_name;
    uint32_t              *       error_count;
    error_buffer_t        *       error_buffer;
    backtrace_t bt;
};

internal::instrumentation_t::instrumentation_t() { }

internal::check_t::check_t(cuda_context_memcheck * c, const char * ename) :
        context(c), entry_name(ename), bt(backtrace_t::instance()) {
    error_count  = NULL;
    error_buffer = NULL;

    unsigned attempts = 0;
    for (; attempts < 3; attempts++) {
        try {
            if (!(error_count)) {
                error_count  = context->error_counts_.pop();
            }

            if (!(error_buffer)) {
                error_buffer = context->error_buffers_.pop();
            }

            break;
        } catch (std::bad_alloc) {
            /* Synchronize the device, clean up some resources, and try
             * again. */
            context->cudaDeviceSynchronize();
            continue;
        }
    }

    if (!(error_count) || !(error_buffer)) {
        /* We expect to have picked up neither. */
        assert(!(error_count));
        assert(!(error_buffer));

        context->error_counts_.push(error_count);
        context->error_buffers_.push(error_buffer);

        /* TODO Log this, then die. */
        assert(0 && "Exhausted resource limits.");
        return;
    }

    /* Initialize the error count to 0. */
    cudaError_t ret;
    ret = callout::cudaMemset(error_count, 0, sizeof(*error_count));
    /* TODO:  Handle more gracefully. */
    assert(ret == cudaSuccess);
}

internal::check_t::~check_t() {
    context->error_counts_.push(error_count);
    context->error_buffers_.push(error_buffer);
}

cudaError_t internal::check_t::check(cudaError_t r) {
    if (r == cudaErrorNotReady) {
        return cudaSuccess;
    } else if (r != cudaSuccess) {
        /**
         * Our instrumentation should keep this from happening.
         */
        return r;
    }

    /**
     * TODO:  Use pinned memory.
     */
    uint32_t     count;
    error_buffer_t host;

    cudaError_t  ret;
    ret = callout::cudaMemcpy(&count, error_count,  sizeof(count),
        cudaMemcpyDeviceToHost);
    assert(ret == cudaSuccess);
    ret = callout::cudaMemcpy(&host,  error_buffer, sizeof(host),
        cudaMemcpyDeviceToHost);
    assert(ret == cudaSuccess);

    if (count > 0) {
        typedef global_t::entry_info_map_t entry_info_map_t;
        global_t * const global = context->global();
        entry_info_map_t::const_iterator it =
            global->entry_info_.find(entry_name);
        if (it == global->entry_info_.end()) {
            /* We couldn't find the metadata. */
            assert(0 && "The impossible happened.");
            return cudaSuccess;
        }

        std::string buffer;

        char buf[256];

        int status;
        char * demangled = abi::__cxa_demangle(entry_name,
            NULL /* output_buffer */, NULL /* length */, &status);
        const char * name;
        if (status == 0) {
            name = demangled;
        } else {
            name = entry_name;
        }

        int sret = snprintf(buf, sizeof(buf),
            "Encountered %u errors in '%s'.\n", count, name);
        if (status == 0) {
            free(demangled);
        }

        assert(sret < (int) sizeof(buf));
        buffer += buf;

        const instrumentation_t * inst = it->second.inst;
        const uint32_t known_errors =
            static_cast<uint32_t>(inst->errors.size());

        count = std::min(count, 32u);
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t e = host.data[i] - 1u;
            if (e >= known_errors) {
                sret = snprintf(buf, sizeof(buf),
                    "Error %u: Unknown error code (%u).\n", i, e);
                assert(sret < (int) sizeof(buf));
                buffer += buf;

                /**
                 * This was probably a severe error.
                 */
                ret = cudaErrorLaunchFailure;
                continue;
            }

            typedef instrumentation_t::error_desc_t error_desc_t;
            const error_desc_t & desc = inst->errors[e];
            std::stringstream ss;
            ss << desc.orig;

            switch (desc.type) {
                case instrumentation_t::no_error: break;
                case instrumentation_t::wild_branch:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Wild branch at %s", i, ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;
                    break;
                case instrumentation_t::wild_prefetch:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Wild prefetch at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;
                    break;
                case instrumentation_t::misaligned_prefetch:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Misaligned prefetch at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_prefetch_global:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds global prefetch at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_prefetch_local:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds local prefetch at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_atomic_shared:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds shared atomic at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_atomic_global:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds global atomic at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_ld_global:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds global load at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_ld_local:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds local load at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_ld_shared:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds local load at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_st_global:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds global store at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_st_local:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds local store at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::outofbounds_st_shared:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Out of bounds local store at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;

                    ret = cudaErrorLaunchFailure;
                    break;
                case instrumentation_t::wild_texture:
                    sret = snprintf(buf, sizeof(buf),
                        "Error %u: Wild texture load at %s", i,
                        ss.str().c_str());
                    assert(sret < (int) sizeof(buf));
                    buffer += buf;
                    break;
            }
        }

        logger::instance().print(buffer.c_str());

        sret = snprintf(buf, sizeof(buf), "Kernel launched by:");
        assert(sret < (int) sizeof(buf) - 1);
        logger::instance().print(buf, bt);
    } /* else: no errors. */

    return ret;
}

enum event_type_t {
    user_created,
    panoptes_created
};

struct internal::event_t {
    /**
     * The event takes ownership of the check_t given to it.
     *
     * If a stream is provided, the stream takes ownership over this event_t.
     */
    event_t(event_type_t et, unsigned int flags,
        stream_t * cs, check_t * check);
    ~event_t();

    cudaError_t query();
    cudaError_t record(stream_t * cs);
    cudaError_t synchronize();

    check_t *           checker;
    cudaEvent_t         event;
    event_type_t const  event_type;
    const unsigned int  flags;
    unsigned int        references;
    stream_t *          stream;
    size_t              sequence_number;
};

struct internal::stream_t {
    stream_t();
    ~stream_t();

    static stream_t * stream_zero();

    void        add_event(event_t * ev);
    cudaError_t busy();
    cudaError_t synchronize();
    cudaError_t synchronize(event_t * ev);

    cudaError_t  last_error;
    cudaStream_t stream;
protected:
    typedef std::list<event_t *> event_list_t;
    event_list_t events_;
    size_t       event_count_;
    size_t       event_marker_;
private:
    explicit stream_t(cudaStream_t s);
};

struct internal::texture_t {
    texture_t();
    ~texture_t();

    const void * bound_pointer;

    bool array_bound;
    internal::array_t * bound_array;

    bool has_validity_texref;
    CUtexref validity_texref;

    size_t       bound_size;
    backtrace_t  binding;

    typedef std::set<const void *> allocation_vt;
    allocation_vt allocations;
};

internal::texture_t::texture_t() : bound_pointer(NULL), array_bound(false),
    bound_array(NULL), has_validity_texref(false) { }
internal::texture_t::~texture_t() { }

internal::event_t::event_t(event_type_t et, unsigned int flags_,
        stream_t * cs, check_t * check) :
        checker(check), event_type(et),
        flags((et == panoptes_created) ?
            cudaEventDisableTiming : flags_), references(1u), stream(cs) {
    /** TODO Don't ignore return values */
    callout::cudaEventCreateWithFlags(&event, flags);

    if (cs) {
        callout::cudaEventRecord(event, stream->stream);
        stream->add_event(this);
    }
}

internal::event_t::~event_t() {
    callout::cudaEventDestroy(event);
    delete checker;
}

cudaError_t internal::event_t::query() {
    return callout::cudaEventQuery(event);
}

cudaError_t internal::event_t::record(stream_t * cs) {
    assert(cs);

    stream = cs;
    cudaError_t ret = callout::cudaEventRecord(event, stream->stream);
    stream->add_event(this);
    return ret;
}

cudaError_t internal::event_t::synchronize() {
    assert(references > 0);

    if (stream) {
        cudaError_t ret = stream->synchronize(this);
        return ret;
    } else {
        return cudaSuccess;
    }
}

internal::stream_t::stream_t() : last_error(cudaSuccess), event_count_(1),
    event_marker_(0) { }

internal::stream_t::~stream_t() {
    synchronize();
}

internal::stream_t::stream_t(cudaStream_t s) : last_error(cudaSuccess),
    stream(s), event_count_(1), event_marker_(0) { }

internal::stream_t * internal::stream_t::stream_zero() {
    return new stream_t(0);
}

/**
 * We mark the event with a unique sequence number.  If we attempt to
 * synchronize with an event_t that has a sequence number in the future,
 * we will know that the event has also been recorded later in the stream.
 */
void internal::stream_t::add_event(event_t * ev) {
    ev->sequence_number = event_count_++;
    ev->references++;
    events_.push_back(ev);
}

cudaError_t internal::stream_t::busy() {
    if (events_.size() == 0) {
        return last_error;
    }

    /**
     * TODO:  Walk forward on the stream and clean things up
     */

    /**
     * Start from the last event and work backwards until we find one that
     * belongs to this stream *and* is in its proper place.
     */
    size_t expected = event_count_;
    for (event_list_t::const_reverse_iterator it = events_.rbegin();
            it != events_.rend(); ++it) {
        event_t * const ev = *it;
        if (ev->sequence_number == expected && ev->stream == this) {
            /* Found an acceptable event. */
            return ev->query();
        }

        assert(expected > 0);
        expected--;
    }

    /*
     * Give up and report the last error.
     */
    return last_error;
}

cudaError_t internal::stream_t::synchronize() {
    return synchronize(NULL);
}

cudaError_t internal::stream_t::synchronize(event_t * target) {
    bool match = false;

    for (event_list_t::iterator it = events_.begin();
            it != events_.end() && !(match); ) {
        ++event_marker_;

        event_t * const ev = *it;

        bool deleted = false;
        if (ev->sequence_number == event_marker_ && ev->stream == this) {
            /* Found an acceptable event. */
            cudaError_t ret = callout::cudaEventSynchronize(ev->event);
            if (ret != cudaSuccess) {
                last_error = ret;
            }

            if (ev->event_type == panoptes_created) {
                deleted = true;
                if (ev->checker) {
                    ret = ev->checker->check(ev->query());
                    if (ret != cudaSuccess) {
                        last_error = ret;
                    }
                }

                delete ev;
            } else {
                /*
                 * This is a user-created event and the user probably would
                 * like to know what happened.
                 */
                if (ev == target) {
                    match = true;
                }
            }
        }

        if (!(deleted)) {
            ev->references--;
            if (ev->references == 0) {
                if (ev->checker) {
                    cudaError_t ret = ev->checker->check(ev->query());
                    if (ret != cudaSuccess) {
                        last_error = ret;
                    }
                }

                delete ev;
            }
        }

        event_list_t::iterator next = it;
        ++next;
        events_.erase(it);
        it = next;
    }

    return last_error;
}

bool cuda_context_memcheck::check_access_device(const void * ptr,
        size_t len, bool signal) const {
    /**
     * Scan through device allocations in the address range [ptr, ptr + len)
     * to check for addressability.
     */
    const uint8_t * const uptr = static_cast<const uint8_t *>(ptr);
    size_t asize, offset;
    for (offset = 0; offset < len; ) {
        uint8_t * const search = const_cast<uint8_t *>(uptr + offset);

        const void * aptr = NULL;
        asize = 0;

        aumap_t::const_iterator it = udevice_allocations_.upper_bound(search);
        if (it == udevice_allocations_.end()) {
            amap_t::const_iterator jit =
                device_allocations_.lower_bound(search);
            if (jit != device_allocations_.end()) {
                aptr  = jit->first;
                asize = jit->second.size;
            }
        } else {
            aptr  = static_cast<const uint8_t *>(it->first) - it->second;
            asize = it->second;
        }

        if (aptr == NULL) {
            break;
        }

        const uint8_t * base  = static_cast<const uint8_t *>(aptr);
        if (base > search) {
            break;
        }
        const size_t head     = (size_t) (search - base);
        if (head > len - offset) {
            break;
        }

        const size_t rem      = std::min(asize - head, len - offset);
        offset += rem;
    }

    if (offset != len) {
        if (offset == 0) {
            // Invalid read starts at usrc + offset
            char msg[128];
            int ret;

            if (is_recently_freed(ptr)) {
                ret = snprintf(msg, sizeof(msg),
                    "Invalid device read of size %zu\n"
                    " Address %p is part of (recently) free'd allocation.",
                    len, ptr);
                signal = false;
            } else {
                ret = snprintf(msg, sizeof(msg),
                    "Invalid device read of size %zu\n"
                    " Address %p is not malloc'd or (recently) free'd.\n"
                    " Possible host pointer?", len, ptr);
            }

            // sizeof(msg) is small, so the cast is safe.
            assert(ret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);
        } else {
            // Invalid read starts at usrc + offset
            //
            // Technically, we don't know whether the full read of
            // (len - offset) additional bytes is invalid.
            char msg[128];
            int ret = snprintf(msg, sizeof(msg),
                "Invalid device read of size %zu\n"
                " Address %p is 0 bytes after a block of size %zu alloc'd\n",
                len - offset, uptr + offset, asize);
            // sizeof(msg) is small, so the cast is safe.
            assert(ret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);
        }

        if (signal) {
            raise(SIGSEGV);
        }

        return false;
    }

    return true;
}

bool cuda_context_memcheck::check_access_host(const void * ptr,
        size_t len) const {
    return valgrind_ ?
        (VALGRIND_CHECK_MEM_IS_ADDRESSABLE(ptr, len) == 0) : true;
}

/**
 * TODO:  On setup, setup SIGSEGV signal handler.
 */
cuda_context_memcheck::cuda_context_memcheck(
            global_context_memcheck * g, int device, unsigned int flags) :
        cuda_context(g, device, flags),
        master_(1 << (lg_max_memory - lg_chunk_bytes)),
        achunks_   (1 << (lg_max_memory - lg_chunk_bytes)),
        vchunks_   (1 << (lg_max_memory - lg_chunk_bytes)), state_(g->state()),
        chunks_aux_(1 << (lg_max_memory - lg_chunk_bytes)), apool_(4, 16),
        vpool_(4, 16), error_counts_(max_errors), error_buffers_(max_errors),
        valgrind_((unsigned) (RUNNING_ON_VALGRIND)),
        pagesize_((size_t) sysconf(_SC_PAGESIZE)) {
    default_achunk_ = apool_.allocate();
    default_vchunk_ = vpool_.allocate();
    initialize_achunk(default_achunk_);
    initialize_vchunk(default_vchunk_);

    /* Mark an adata_chunk as fully addressable. */
    initialized_achunk_ = apool_.allocate();
    memset(initialized_achunk_->host()->a_data, 0xFF,
        sizeof(initialized_achunk_->host()->a_data));

    // Initialize master list to point at the default chunk
    metadata_ptrs default_ptrs;
    default_ptrs.adata = default_achunk_->gpu();
    default_ptrs.vdata = default_vchunk_->gpu();

    metadata_ptrs * const master = master_.host();
    for (size_t i = 0; i < (1u << (lg_max_memory - lg_chunk_bytes)); i++) {
        master[i]   = default_ptrs;

        achunks_[i] = default_achunk_;
        vchunks_[i] = default_vchunk_;
    }

    // Copy pool to GPU
    apool_.to_gpu();
    vpool_.to_gpu();

    state_->register_master(device, default_ptrs, &master_);

    /**
     * Load any registered variables.
     */
    for (global_t::variable_definition_map_t::const_iterator it =
            g->variable_definitions_.begin();
            it != g->variable_definitions_.end(); ++it) {
        const std::string & deviceName = it->first .second;
        const char *        hostVar    = it->second.hostVar;

        /* Do not trust the caller provided size. */
        cudaError_t ret;
        size_t true_size;
        ret = cuda_context::cudaGetSymbolSize(&true_size, deviceName.c_str());
        if (ret != cudaSuccess) {
            /* Somehow our registration did not take. */
            continue;
        }

        void * ptr;
        cuda_context::cudaGetSymbolAddress(&ptr, hostVar);
        if (ret != cudaSuccess) {
            /* Somehow our registration did not take. */
            continue;
        }

        add_device_allocation(ptr, true_size, false);

        const variable_t & variable = it->second.ptx;
        const ptx_t * parent_ptx    = it->second.parent_ptx;
        /**
         * In PTX 2.3 and later, variables are initialized to 0 by default.
         */
        const bool ptx_23 = parent_ptx->version_major > 2 ||
            (parent_ptx->version_major == 2 && parent_ptx->version_minor >= 3);
        if ((variable.has_initializer || ptx_23) &&
                !((variable.is_array && variable.array_flexible))) {
            validity_set(ptr, variable.size(), NULL);
        }
    }

    /**
     * Initialize the master lookup symbol.
     * TODO:  Add error checking.
     */
    using internal::modules_t;
    bool found_somewhere = false;
    for (modules_t::module_vt::iterator it = modules_->modules.begin();
            it != modules_->modules.end(); ++it) {
        using internal::module_t;
        module_t * const module = *it;
        CUdeviceptr dptr;
        CUresult ret = cuModuleGetGlobal(&dptr, NULL, module->module,
            __master_symbol);
        if (ret == CUDA_ERROR_NOT_FOUND) {
            continue;
        } else if (ret == CUDA_SUCCESS) {
            found_somewhere = true;

            void * symbol = (void *) dptr;
            void * master_addr = master_.gpu();
            cudaError_t rret = callout::cudaMemcpy(symbol, &master_addr,
                sizeof(master_addr), cudaMemcpyHostToDevice);
            assert(rret == cudaSuccess);
        } // else ignore the error
    }

    if (!(found_somewhere || g->modules_.size() == 0)) {
        char msg[] = "Unable to find master symbol.";
        logger::instance().print(msg);
    }

    // Insert the zero stream
    streams_.insert(stream_map_t::value_type(NULL,
        internal::stream_t::stream_zero()));
}

cuda_context_memcheck::~cuda_context_memcheck() {
    /**
     * Unregister.
     */
    state_->unregister_master(device_);

    /**
     * Cleanup outstanding streams.
     */
    for (stream_map_t::iterator it = streams_.begin();
            it != streams_.end(); ++it) {
        if (it->second->busy() != cudaSuccess) {
            /** TODO:  A backtrace of the original allocation would be
             * nice... */
            char msg[128];
            int ret = snprintf(msg, sizeof(msg),
                "cudaStream_t has outstanding work at shutdown.\n");

            // sizeof(msg) is small, so the cast is safe.
            assert(ret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);
        }

        if (it->first) {
            free_handle(it->first);
        }

        delete it->second;
    }

    /**
     * Cleanup outstanding events.
     */
    for (event_map_t::iterator it = events_.begin();
            it != events_.end(); ++it) {
        free_handle(it->first);
        delete it->second;
    }

    /**
     * Cleanup outstanding array allocations.
     */
    for (oamap_t::iterator it = opaque_dimensions_.begin();
            it != opaque_dimensions_.end(); ++it) {
        /* TODO:  Warn. */
        (void) callout::cudaFreeArray(
            const_cast<struct cudaArray *>(it->first));
        delete it->second;
    }

    /**
     * Cleanup outstanding bound textures.
     */
    for (texture_map_t::iterator it = bound_textures_.begin();
            it != bound_textures_.end(); ++it) {
        delete it->second;
    }

    for (size_t i = 0; i < 1 << (lg_max_memory - lg_chunk_bytes); i++) {
        apool_t::handle_t * const a = achunks_[i];
        if (a != default_achunk_ && a != initialized_achunk_) {
            apool_.free(a);
        }

        if (vchunks_[i] != default_vchunk_) {
            vpool_.free(vchunks_[i]);
        }
    }

    apool_.free(default_achunk_);
    apool_.free(initialized_achunk_);
    vpool_.free(default_vchunk_);
}

void cuda_context_memcheck::initialize_achunk(
        apool_t::handle_t * handle) const {
    /* Do work on host */
    adata_chunk * host = handle->host();
    memset(host->a_data, 0,    sizeof(host->a_data));
}

void cuda_context_memcheck::initialize_vchunk(
        vpool_t::handle_t * handle) const {
    /* Do work on host */
    vdata_chunk * host = handle->host();
    memset(host->v_data, 0xFF, sizeof(host->v_data));
}

cuda_context_memcheck::td & cuda_context_memcheck::thread_data() {
    td * t = thread_data_.get();
    if (!(t)) {
        t = new td();
        thread_data_.reset(t);
    }
    return *t;
}

cudaError_t cuda_context_memcheck::cudaConfigureCall(dim3 gridDim,
        dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    /**
     * Translate the caller's stream handle into a real cudaStream_t.
     */

    cudaStream_t real_stream;
    void ** handle = reinterpret_cast<void **>(stream);

    scoped_lock lock(mx_);
    stream_map_t::iterator sit = streams_.find(handle);
    if (sit == streams_.end()) {
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaConfigureCall called with invalid stream\n");
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        return cudaErrorInvalidConfiguration;
    } else {
        real_stream = sit->second->stream;
        thread_data().stream_stack.push(sit->second);
    }

    return cuda_context::cudaConfigureCall(gridDim, blockDim, sharedMem,
        real_stream);
}

cudaError_t cuda_context_memcheck::cudaDeviceSynchronize() {
    /**
     * Synchronize all of the streams we know about.
     */
    scoped_lock lock(mx_);

    cudaError_t ret = cudaSuccess;

    for (stream_map_t::iterator it = streams_.begin();
            it != streams_.end(); ++it) {
        cudaError_t tmp = it->second->synchronize();
        if (tmp != cudaSuccess) {
            ret = tmp;
        }
    }

    /**
     * Then do a real cudaDeviceSynchronize for good measure.
     */
    cudaError_t tmp = callout::cudaDeviceSynchronize();
    if (tmp != cudaSuccess) {
        ret = tmp;
    }

    return ret;
}

cudaError_t cuda_context_memcheck::cudaEventCreate(cudaEvent_t *event) {
    if (!(event)) {
        return cudaErrorInvalidValue;
    }

    internal::event_t * ev = new internal::event_t(user_created,
        cudaEventDefault, NULL, NULL);
    void ** handle = create_handle();

    scoped_lock lock(mx_);
    events_.insert(event_map_t::value_type(handle, ev));

    *event = reinterpret_cast<cudaEvent_t>(handle);
    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaEventCreateWithFlags(
        cudaEvent_t *event, unsigned int flags) {
    if (!(event)) {
        return cudaErrorInvalidValue;
    }

    internal::event_t * ev = new internal::event_t(user_created,
        flags, NULL, NULL);
    void ** handle = create_handle();

    scoped_lock lock(mx_);
    events_.insert(event_map_t::value_type(handle, ev));

    *event = reinterpret_cast<cudaEvent_t>(handle);
    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaEventDestroy(cudaEvent_t event) {
    void ** handle = reinterpret_cast<void **>(event);

    scoped_lock lock(mx_);
    event_map_t::iterator it = events_.find(handle);
    if (it == events_.end()) {
        /**
         * CUDA segfaults here.
         */
        raise(SIGSEGV);
        return cudaErrorInvalidValue;
    }

    internal::event_t * ev = it->second;
    events_.erase(it);
    free_handle(handle);

    cudaError_t ret = ev->query();
    /* If the event hasn't happened yet, this is not an error. */
    if (ret == cudaErrorNotReady) {
        ret = cudaSuccess;
    }

    ev->references--;
    if (ev->references == 0) {
        delete ev;
    }

    return ret;
}

cudaError_t cuda_context_memcheck::cudaEventElapsedTime(float * ms,
        cudaEvent_t start, cudaEvent_t end) {
    void ** estart  = reinterpret_cast<void **>(start);
    void ** eend    = reinterpret_cast<void **>(end);

    if (!(ms) || !(check_access_host(ms, sizeof(*ms)))) {
        /**
         * CUDA segfaults here.
         */
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaEventElapsedTime ms argument (%p) is not addressable.\n",
            ms);
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        raise(SIGSEGV);
        return cudaErrorInvalidValue;
    }

    scoped_lock lock(mx_);

    event_map_t::iterator start_it = events_.find(estart);
    if (start_it == events_.end()) {
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaEventElapsedTime start argument is an invalid event.\n");
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        assert(ms);
        (void) VALGRIND_MAKE_MEM_UNDEFINED(ms, sizeof(*ms));

        return cudaErrorInvalidResourceHandle;
    }

    event_map_t::iterator end_it = events_.find(eend);
    if (end_it == events_.end()) {
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaEventElapsedTime end argument is an invalid event.\n");
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        assert(ms);
        (void) VALGRIND_MAKE_MEM_UNDEFINED(ms, sizeof(*ms));

        return cudaErrorInvalidResourceHandle;
    }

    const unsigned int start_flags  = start_it->second->flags;
    const bool         start_timing = ~start_flags & cudaEventDisableTiming;
    const unsigned int end_flags    = end_it  ->second->flags;
    const bool         end_timing   = ~end_flags   & cudaEventDisableTiming;

    if (!(start_timing && end_timing)) {
        char msg[128];
        int ret;
        if (start_timing) {
              ret = snprintf(msg, sizeof(msg),
                "cudaEventElapsedTime end event was created with "
                "cudaEventDisableTiming.\n");
        } else if (end_timing) {
             ret = snprintf(msg, sizeof(msg),
                "cudaEventElapsedTime start event was created with "
                "cudaEventDisableTiming.\n");
        } else { /* Neither */
            ret = snprintf(msg, sizeof(msg),
                "cudaEventElapsedTime start and end events were created with "
                "cudaEventDisableTiming.\n");
        }

        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        assert(ms);
        (void) VALGRIND_MAKE_MEM_UNDEFINED(ms, sizeof(*ms));

        return cudaErrorInvalidResourceHandle;
    }

    const bool start_recorded   = start_it->second->stream;
    const bool end_recorded     = end_it  ->second->stream;
    if (!(start_recorded && end_recorded)) {
        char msg[128];
        int ret;
        if (start_recorded) {
              ret = snprintf(msg, sizeof(msg),
                "cudaEventElapsedTime end event has not been recorded.\n");
        } else if (end_recorded) {
             ret = snprintf(msg, sizeof(msg),
                "cudaEventElapsedTime start event has not been recorded\n");
        } else { /* Neither */
            ret = snprintf(msg, sizeof(msg),
                "cudaEventElapsedTime start and end events have not been "
                "recorded\n");
        }

        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        assert(ms);
        (void) VALGRIND_MAKE_MEM_UNDEFINED(ms, sizeof(*ms));

        return cudaErrorInvalidResourceHandle;
    }

    cudaError_t ret = callout::cudaEventElapsedTime(ms,
        start_it->second->event, end_it->second->event);
    if (ret != cudaSuccess) {
        assert(ms);
        (void) VALGRIND_MAKE_MEM_UNDEFINED(ms, sizeof(*ms));
    }

    return ret;
}

cudaError_t cuda_context_memcheck::cudaEventQuery(cudaEvent_t event) {
    void ** ehandle = reinterpret_cast<void **>(event);

    scoped_lock lock(mx_);

    event_map_t::iterator eit = events_.find(ehandle);
    if (eit == events_.end()) {
        /**
         * CUDA segfaults here.
         */
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaEventQuery called on invalid event.\n");
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        raise(SIGSEGV);
        return cudaErrorInvalidValue;
    }

    return eit->second->query();
}

cudaError_t cuda_context_memcheck::cudaEventRecord(cudaEvent_t event,
        cudaStream_t stream) {
    void ** ehandle = reinterpret_cast<void **>(event);
    void ** shandle = reinterpret_cast<void **>(stream);

    scoped_lock lock(mx_);
    stream_map_t::iterator sit = streams_.find(shandle);
    if (sit == streams_.end()) {
        /**
         * CUDA returns an error code (10709) that isn't on any of the
         * published lists.  Any program depending on that behavior will
         * be as easily broken by Panoptes returning an arbitrary value as it
         * will be if this behavior changes in the next release of CUDA.
         */
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaEventRecord called on invalid stream.\n");
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        return cudaErrorInvalidResourceHandle;
    }

    event_map_t::iterator eit = events_.find(ehandle);
    if (eit == events_.end()) {
        /**
         * CUDA segfaults here (prior to CUDA 5).
         */
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaEventRecord called on invalid event.\n");
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        if (runtime_version_ < 5000 /* 5.0 */) {
            raise(SIGSEGV);
        }
        return cudaErrorUnknown;
    }

    return eit->second->record(sit->second);
}

cudaError_t cuda_context_memcheck::cudaEventSynchronize(cudaEvent_t event) {
    void ** ehandle = reinterpret_cast<void **>(event);

    scoped_lock lock(mx_);

    event_map_t::iterator eit = events_.find(ehandle);
    if (eit == events_.end()) {
        /**
         * CUDA segfaults here.
         */
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaEventSynchronize called on invalid event.\n");
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        if (runtime_version_ < 4010 /* 4.1 */) {
            raise(SIGSEGV);
        }
        return cudaErrorUnknown;
    }

    return eit->second->synchronize();
}

cudaError_t cuda_context_memcheck::cudaFree(void *devPtr) {
    if (!(devPtr)) {
        /* cudaFree(NULL) is a no op */
        return setLastError(cudaSuccess);
    }

    scoped_lock lock(mx_);
    oamap_t::const_iterator it = opaque_dimensions_.find(
        static_cast<struct cudaArray *>(devPtr));
    if (it != opaque_dimensions_.end()) {
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "Mismatched cudaFree / cudaFreeArray / cudaFreeHost\n"
            " Address %p allocated by %s\n", devPtr,
            array_allocator_name(it->second->allocator));
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        return cudaErrorInvalidDevicePointer;
    }

    cudaError_t ret = remove_device_allocation(devPtr);
    if (ret != cudaSuccess) {
        return setLastError(ret);
    }

    /* Actually free */
    return setLastError(cuda_context::cudaFree(devPtr));
}

cudaError_t cuda_context_memcheck::cudaFreeArray(struct cudaArray *array) {
    if (!(array)) {
        /* cudaFreeArray(NULL) is a no op */
        return setLastError(cudaSuccess);
    }

    scoped_lock lock(mx_);
    oamap_t::iterator it = opaque_dimensions_.find(array);
    if (it == opaque_dimensions_.end()) {
        char msg[128];
        int ret;

        const void *    block_ptr;
        size_t          block_offset;

        cudaError_t cret;

        if (is_device_pointer(array, &block_ptr, &block_offset)) {
            ret = snprintf(msg, sizeof(msg),
                "Mismatched cudaFree / cudaFreeArray / cudaFreeHost\n"
                " Address %p is device pointer %zu bytes inside %p\n",
                array, block_offset, block_ptr);
            cret = cudaErrorInvalidDevicePointer;
        } else {
            ret = snprintf(msg, sizeof(msg),
                "Mismatched cudaFree / cudaFreeArray / cudaFreeHost\n"
                " Address %p is a host pointer.\n",
                array);
            cret = cudaErrorInvalidValue;
        }

        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        return cret;
    } else {
        typedef internal::array_t::binding_set_t::const_iterator binding_it;
        for (binding_it jit = it->second->bindings.begin();
                jit != it->second->bindings.end(); ++jit) {
            texture_map_t::const_iterator kit = bound_textures_.find(*jit);
            if (kit == bound_textures_.end()) {
                /* Inconsistency.  We still think a texture is bound to us that
                 * does not exist. */
                assert(0 && "The imposssible happened.");
                continue;
            }

            /* Print warning. */
            char msg[128];
            int pret;
            pret = snprintf(msg, sizeof(msg), "Texture %p still bound to "
                "cudaArray %p currently being freed.",
                kit->first, array);
            // sizeof(msg) is small, so the cast is safe.
            assert(pret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);

            pret = snprintf(msg, sizeof(msg),
                "Texture most recently bound by:");
            assert(pret < (int) sizeof(msg) - 1);
            logger::instance().print(msg, kit->second->binding);

            /* Cleanup, leave array_bound set. */
            kit->second->bound_array = NULL;
        }

        delete it->second;
        opaque_dimensions_.erase(it);
    }

    return cuda_context::cudaFreeArray(array);
}

cudaError_t cuda_context_memcheck::cudaFreeHost(void * ptr) {
    if (!(ptr)) {
        /* cudaFreeHost(NULL) is a no op */
        return setLastError(cudaSuccess);
    }

    scoped_lock lock(mx_);
    ahmap_t::iterator it = host_allocations_.find(ptr);
    if (it == host_allocations_.end()) {
        char msg[128];
        int ret;

        oamap_t::const_iterator jit = opaque_dimensions_.find(
            static_cast<const struct cudaArray *>(ptr));
        if (jit != opaque_dimensions_.end()) {
            ret = snprintf(msg, sizeof(msg),
                "Mismatched cudaFree / cudaFreeArray / cudaFreeHost\n"
                " Address %p is a cudaArray.\n", ptr);
            assert(ret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);

            ret = snprintf(msg, sizeof(msg), "cudaArray allocated by by:");
            assert(ret < (int) sizeof(msg) - 1);
            logger::instance().print(msg, jit->second->allocation_bt);

            return cudaErrorInvalidHostPointer;
        }

        const void *    block_ptr;
        size_t          block_offset;

        if (is_device_pointer(ptr, &block_ptr, &block_offset)) {
            ret = snprintf(msg, sizeof(msg),
                "Mismatched cudaFree / cudaFreeArray / cudaFreeHost\n"
                " Address %p is device pointer %zu bytes inside %p\n",
                ptr, block_offset, block_ptr);
        } else {
            ret = snprintf(msg, sizeof(msg),
                "Mismatched cudaFree / cudaFreeArray / cudaFreeHost\n"
                " Address %p is a host pointer allocated elsewhere.\n",
                ptr);
        }

        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        return cudaErrorInvalidHostPointer;
    }

    /* Erase and free */
    const size_t ptr_size = it->second.size;
    host_allocations_.erase(it);

    /* Find the upperbound and remove it */
    const uint8_t * upper_ptr = static_cast<const uint8_t *>(ptr) + ptr_size;
    acmap_t::iterator uit = uhost_allocations_.find(upper_ptr);
    if (uit == uhost_allocations_.end()) {
        /* Glaring inconsistency.  TODO:  There may be a better error. */
        assert(0 && "The impossible happened.");
        return cudaErrorInvalidHostPointer;
    }
    uhost_allocations_.erase(uit);

    VALGRIND_FREELIKE_BLOCK(ptr, ptr_size);

    return setLastError(callout::cudaFreeHost(ptr));
}

cudaError_t cuda_context_memcheck::cudaGetDeviceProperties(
        struct cudaDeviceProp *prop, int device) {
    struct cudaDeviceProp prop_;

    if (device == device_) {
        if (!(load_device_info())) {
            /* TODO:  Handle this more appropriately. */
            return cudaErrorUnknown;
        }

        prop_ = info_;
    } else {
        cudaError_t ret =
            cuda_context::cudaGetDeviceProperties(&prop_, device);
        if (ret != cudaSuccess) {
            return ret;
        }
    }

    prop_.canMapHostMemory = 0;
    *prop = prop_;

    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaHostAlloc(void **pHost, size_t size,
        unsigned int flags) {
    if (!(pHost)) {
        return cudaErrorInvalidValue;
    }

    /**
     * Mask out request to map
     */
    const unsigned int clean_flags = flags &
        ~static_cast<unsigned int>(cudaHostAllocMapped);

    cudaError_t ret = callout::cudaHostAlloc(pHost, size, clean_flags);
    if (ret != cudaSuccess) {
        return ret;
    }

    void * const ptr_ = *pHost;

    scoped_lock lock(mx_);

    host_aux_t aux;
    aux.allocation_type = allocation_hostalloc;
    aux.flags           = flags;
    aux.size            = size;

    host_allocations_.insert(ahmap_t::value_type(ptr_, aux));

    /**
     * Prepare upperbound.
     */
    const uint8_t * upper_ptr = static_cast<const uint8_t *>(ptr_) + size;
    uhost_allocations_.insert(acmap_t::value_type(upper_ptr, size));

    VALGRIND_MALLOCLIKE_BLOCK(ptr_, size, 0, 0);
    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaHostGetDevicePointer(void **pDevice,
        void *pHost, unsigned int flags) {
    if (flags != 0) {
        return cudaErrorInvalidValue;
    } else if (!(pHost)) {
        return cudaErrorInvalidValue;
    }

    /**
     * Check for allocation.
     */
    scoped_lock lock(mx_);
    acmap_t::const_iterator uit = uhost_allocations_.upper_bound(pHost);
    if (uit == uhost_allocations_.end() ||
            static_cast<const uint8_t *>(uit->first) - uit->second > pHost) {
        ahmap_t::const_iterator it = host_allocations_.lower_bound(pHost);
        if (it == host_allocations_.end()) {
            return cudaErrorInvalidValue;
        }
    }

    if (!(pDevice)) {
        /**
         * CUDA segfaults here.
         */
        raise(SIGSEGV);
    }

    /**
     * Panoptes does not support mapped allocations.
     */
    return cudaErrorMemoryAllocation;
}

cudaError_t cuda_context_memcheck::cudaHostGetFlags(unsigned int *pFlags,
        void *pHost) {
    if (!(pHost)) {
        return cudaErrorInvalidValue;
    }

    scoped_lock lock(mx_);

    unsigned int flags;

    acmap_t::const_iterator uit = uhost_allocations_.upper_bound(pHost);
    if (uit == uhost_allocations_.end() ||
            static_cast<const uint8_t *>(uit->first) - uit->second > pHost) {
        ahmap_t::const_iterator it = host_allocations_.lower_bound(pHost);
        if (it == host_allocations_.end()) {
            return cudaErrorInvalidValue;
        } else {
            flags = it->second.flags;
        }
    } else {
        const void * base_ptr =
            static_cast<const uint8_t *>(uit->first) - uit->second;

        ahmap_t::const_iterator it = host_allocations_.lower_bound(base_ptr);
        if (it == host_allocations_.end()) {
            /* Inconsistency */
            assert(0 && "The impossible happened.");
            return cudaErrorInvalidValue;
        }

        flags = it->second.flags;
    }

    if (!(pFlags)) {
        /**
         * CUDA segfaults here.
         */
        raise(SIGSEGV);
    }

    *pFlags = flags;
    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaHostRegister(void *ptr, size_t size,
        unsigned int flags) {
    if (!(ptr)) {
        return cudaErrorInvalidValue;
    }

    const void * upper_ptr = static_cast<const uint8_t *>(ptr) + size;

    scoped_lock lock(mx_);

    /**
     * Check for overlapping registrations.
     */
    acmap_t::const_iterator uit = uhost_allocations_.upper_bound(ptr);
    if (uit != uhost_allocations_.end()) {
        const void * const existing_lower =
            static_cast<const uint8_t *>(uit->first) - uit->second;
        if (existing_lower <= upper_ptr) {
            char msg[128];
            int ret;

            // We need to lookup the type of existing allocation.
            ahmap_t::const_iterator lit =
                host_allocations_.find(existing_lower);
            if (lit == host_allocations_.end()) {
                assert(0 && "The impossible happened.");
                return cudaErrorUnknown;
            }

            switch (lit->second.allocation_type) {
                case allocation_hostalloc:
                case allocation_mallochost:
                    ret = snprintf(msg, sizeof(msg), "cudaHostRegistration "
                        "for [%p, %p) overlaps with pinned host allocation "
                        "[%p, %p)", ptr, upper_ptr,
                        existing_lower, uit->first);
                    // sizeof(msg) is small, so the cast is safe.
                    assert(ret < (int) sizeof(msg) - 1);
                    logger::instance().print(msg);

                    return cudaErrorUnknown;
                case allocation_registered:
                    ret = snprintf(msg, sizeof(msg), "cudaHostRegistration "
                        "for [%p, %p) overlaps with existing host "
                        "registration [%p, %p)", ptr, upper_ptr,
                        existing_lower, uit->first);
                    // sizeof(msg) is small, so the cast is safe.
                    assert(ret < (int) sizeof(msg) - 1);
                    logger::instance().print(msg);

                    #if CUDART_VERSION >= 4010 /* 4.1 */
                    return cudaErrorHostMemoryAlreadyRegistered;
                    #else
                    return cudaErrorUnknown;
                    #endif
                default:
                    assert(0 && "Invalid allocation type encountered.");
                    return cudaErrorUnknown;
            }
        }
    }

    ahmap_t::const_iterator lit = host_allocations_.lower_bound(ptr);
    if (lit != host_allocations_.end()) {
        const void * const existing_upper =
            static_cast<const uint8_t *>(lit->first) + lit->second.size;
        if (existing_upper > ptr) {
            char msg[128];
            int ret;

            switch (lit->second.allocation_type) {
                case allocation_hostalloc:
                case allocation_mallochost:
                    ret = snprintf(msg, sizeof(msg), "cudaHostRegistration "
                        "for [%p, %p) overlaps with pinned host allocation "
                        "[%p, %p)", ptr, upper_ptr, lit->first,
                        existing_upper);
                    // sizeof(msg) is small, so the cast is safe.
                    assert(ret < (int) sizeof(msg) - 1);
                    logger::instance().print(msg);

                    return cudaErrorUnknown;
                case allocation_registered:
                    ret = snprintf(msg, sizeof(msg), "cudaHostRegistration "
                        "for [%p, %p) overlaps with existing host "
                        "registration [%p, %p)", ptr, upper_ptr,
                        lit->first, existing_upper);
                    // sizeof(msg) is small, so the cast is safe.
                    assert(ret < (int) sizeof(msg) - 1);
                    logger::instance().print(msg);

                    #if CUDA_VERSION >= 4010 /* 4.1 */
                    return cudaErrorHostMemoryAlreadyRegistered;
                    #else
                    return cudaErrorUnknown;
                    #endif
                default:
                    assert(0 && "Invalid allocation type encountered.");
                    return cudaErrorUnknown;
            }
        }
    }

    /**
     * Mask out request to map
     */
    const unsigned int clean_flags = flags &
        ~static_cast<unsigned int>(cudaHostAllocMapped);

    cudaError_t ret = callout::cudaHostRegister(ptr, size, clean_flags);
    if (ret != cudaSuccess) {
        return ret;
    }

    host_aux_t aux;
    aux.allocation_type = allocation_registered;
    aux.flags           = flags;
    aux.size            = size;

    host_allocations_.insert(ahmap_t::value_type(ptr, aux));

    /**
     * Prepare upperbound.
     */
    const uint8_t * upper = static_cast<const uint8_t *>(ptr) + size;
    uhost_allocations_.insert(acmap_t::value_type(upper, size));

    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaHostUnregister(void *ptr) {
    scoped_lock lock(mx_);
    ahmap_t::iterator it = host_allocations_.find(ptr);
    if (it == host_allocations_.end()) {
        return cudaErrorInvalidValue;
    }

    if (it->second.allocation_type != allocation_registered) {
        return cudaErrorInvalidValue;
    }

    // Find corresponding entry in uhost_allocations_
    size_t size  = it->second.size;
    const void * upper = static_cast<const uint8_t *>(ptr) + size;
    acmap_t::iterator uit = uhost_allocations_.find(upper);
    if (uit == uhost_allocations_.end()) {
        assert(0 && "The impossible happened.");
        return cudaErrorInvalidValue;
    }

    host_allocations_.erase(it);
    uhost_allocations_.erase(uit);

    return callout::cudaHostUnregister(ptr);
}

cudaError_t cuda_context_memcheck::remove_device_allocation(
        const void * device_ptr) {
    /**
     * Retrieve the pointer from the allocation list and erase it.
     */
    amap_t::iterator it = device_allocations_.find(device_ptr);
    if (it == device_allocations_.end()) {
        return cudaErrorInvalidDevicePointer;
    }

    const size_t size = it->second.size;
    const bool must_free = it->second.must_free;
    aumap_t::iterator uit = udevice_allocations_.find(
        static_cast<const uint8_t *>(device_ptr) + size);
    if (uit == udevice_allocations_.end()) {
        assert(0 &&
            "The impossible happened.  Inconsistent upper/lower bounds");
        return cudaErrorInvalidDevicePointer;
    } else {
        assert(uit->second == size);
    }

    /**
     * Note any unbound textures and clean up the state.
     */
    for (device_aux_t::binding_set_t::const_iterator jit =
            it->second.bindings.begin();
            jit != it->second.bindings.end(); ++jit) {
        /* Look up texture information. */
        texture_map_t::const_iterator kit = bound_textures_.find(*jit);
        if (kit == bound_textures_.end()) {
            /* Inconsistency. */
            assert(0 && "The impossible happened.");
            continue;
        }

        /* Print warning */
        char msg[128];
        int pret;
        pret = snprintf(msg, sizeof(msg), "Texture %p still bound to "
            "memory %p currently being freed.",
            kit->first, device_ptr);
        // sizeof(msg) is small, so the cast is safe.
        assert(pret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        pret = snprintf(msg, sizeof(msg), "Texture most recently bound by:");
        assert(pret < (int) sizeof(msg) - 1);
        logger::instance().print(msg, kit->second->binding);

        /* Cleanup */
        kit->second->allocations.erase(device_ptr);
    }

    device_allocations_.erase(it);
    udevice_allocations_.erase(uit);

    /**
     * Add to recently freed list.
     */
    add_recent_free(device_ptr, size);

    uintptr_t ptr = reinterpret_cast<uintptr_t>(device_ptr);

    const size_t chunk_bytes = 1 << lg_chunk_bytes;
    const size_t first_chunk =  ptr                           >> lg_chunk_bytes;
    const size_t last_chunk  = (ptr + size + chunk_bytes - 1) >> lg_chunk_bytes;
    bool master_dirty        = false;
    bool synchronized        = false;

    typedef global_memcheck_state::chunk_updates_t updates_t;
    updates_t updates;

    for (size_t i = first_chunk; i < last_chunk; i++) {
        /* The host is authoritative for addressability data */
        assert(achunks_[i] != default_achunk_);

        /* Decrement usage */
        assert(chunks_aux_[i].allocations > 0);
        chunks_aux_[i].allocations--;

        if (chunks_aux_[i].allocations == 0) {
            /* Free */
            if (achunks_[i] != initialized_achunk_) {
                /* There cannot be anything touching the device */
                if (!(synchronized)) {
                    cudaError_t sret = callout::cudaDeviceSynchronize();
                    if (sret != cudaSuccess) {
                        return sret;
                    }

                    synchronized = true;
                }

                apool_.free(achunks_[i]);
            }

            if (chunks_aux_[i].large_chunk) {
                /* Only free if this is the owner. */
                if (chunks_aux_[i].owner_index == i) {
                    callout::cudaFree(chunks_aux_[i].groot);
                    chunks_aux_[i].groot = NULL;

                    callout::cudaFreeHost(chunks_aux_[i].hroot);
                    chunks_aux_[i].hroot = NULL;
                }

                chunks_aux_[i].large_chunk = false;
                delete chunks_aux_[i].handle;
                chunks_aux_[i].handle = NULL;
            } else {
                vpool_.free(vchunks_[i]);
            }

            achunks_[i] = default_achunk_;
            vchunks_[i] = default_vchunk_;
            metadata_ptrs ptrs;
            ptrs.adata = achunks_[i]->gpu();
            ptrs.vdata = vchunks_[i]->gpu();

            updates.push_back(updates_t::value_type(i, ptrs));
            master_dirty = true;

            continue;
        }

        /* Set addressability bits...
         *
         * Starting at (ptr       ) or  i      * chunk_bytes,x    whichever is greater.
         * Ending at   (ptr + size) or (i + 1) * chunk_bytes - 1, whichever is less.
         */
        uintptr_t start = std::max(ptr,         i * chunk_bytes)     & (chunk_bytes - 1);
        uintptr_t end   = std::min(ptr + size - i * chunk_bytes, chunk_bytes);

        /* These form the ranges for applying memset to a_data.  They may
         * differ from start and end because the last byte on each end needs to
         * be adjusted. */
        uintptr_t astart, aend;

        if ((start & (CHAR_BIT - 1)) != 0) {
            /* This write isn't a whole byte */
            uint8_t mask    = static_cast<uint8_t>(
                0xFF << (start & (CHAR_BIT - 1)));
            astart          = (start + CHAR_BIT - 1) & ~((uintptr_t) CHAR_BIT);

            /* If the new start > end, we need to unset the high order bits */
            if (astart > end) {
                assert(astart - end < CHAR_BIT);
                /* This sets the high order bits which are not accessible */
                const uint8_t end_mask = static_cast<uint8_t>(0xFF <<
                    (CHAR_BIT - (astart - end)));

                /* Mask these out */
                mask = mask & static_cast<uint8_t>(~end_mask);
            }

            /* Clear these bits */
            achunks_[i]->host()->a_data[start / CHAR_BIT] &=
                static_cast<uint8_t>(~mask);
        } else {
            astart = start;
        }

        if (end > astart && (end & (CHAR_BIT - 1)) != 0) {
            uint8_t mask = static_cast<uint8_t>(~(0xFF <<
                (end & (CHAR_BIT - 1))));
            achunks_[i]->host()->a_data[end / CHAR_BIT] &=
                static_cast<uint8_t>(~mask);
            aend = end & ~((uintptr_t) CHAR_BIT - 1);
        } else {
            aend = end;
        }

        if (aend > astart) {
            memset(static_cast<uint8_t *>(achunks_[i]->host()->a_data) +
                astart / CHAR_BIT, 0x0, (aend - astart) / CHAR_BIT);
        }

        /* Push to the device */
        cudaError_t tret = callout::cudaMemcpy(achunks_[i]->gpu()->a_data,
            achunks_[i]->host()->a_data, sizeof(achunks_[i]->host()->a_data),
            cudaMemcpyHostToDevice);
        if (tret != cudaSuccess) {
            return tret;
        }

        if (valgrind_) {
            /* Set vbits. */
            assert(vchunks_[i] != default_vchunk_);
            vdata_chunk * vdata = vchunks_[i]->gpu();
            uint8_t * vptr = reinterpret_cast<uint8_t *>(vdata->v_data) + start;
            assert(end - start <= sizeof(vdata->v_data));

            tret = callout::cudaMemset(vptr, 0xFF, end - start);
            if (tret != cudaSuccess) {
                return tret;
            }
        }
    }

    /* If any chunks were freed, update the master list */
    assert(master_dirty ^ updates.empty());
    if (master_dirty) {
        state_->update_master(device_, false, updates);
    }

    if (valgrind_ && must_free) {
        VALGRIND_FREELIKE_BLOCK(device_ptr, 0);
    }

    return cudaSuccess;
}

void cuda_context_memcheck::clear() {
    for (stream_map_t::iterator it = streams_.begin();
            it != streams_.end(); ++it) {
        if (it->first) {
            free_handle(it->first);
        }

        delete it->second;
    }
    streams_.clear();

    if (!(valgrind_)) {
        /* None of the operations below have any impact without Valgrind. */
        return;
    }

    for (amap_t::const_iterator it = device_allocations_.begin();
            it != device_allocations_.end(); ++it) {
        if (it->second.must_free) {
            /* Remove Valgrind registration. */
            VALGRIND_FREELIKE_BLOCK(it->first, 0);
        }
    }
}

cudaError_t cuda_context_memcheck::cudaMalloc(void **devPtr, size_t size) {
    if (!(devPtr)) {
        return cudaErrorInvalidValue;
    }

    cudaError_t ret = cuda_context::cudaMalloc(devPtr, size);
    if (ret == cudaSuccess) {
        scoped_lock lock(mx_);
        add_device_allocation(*devPtr, size, true);
    }

    return setLastError(ret);
}

void cuda_context_memcheck::add_device_allocation(const void * device_ptr,
        size_t size, bool must_free) {
    if (!(device_ptr)) {
        /* TODO:  Maybe warn if size > 0? */
        return;
    }

    uintptr_t ptr = reinterpret_cast<uintptr_t>(device_ptr);
    /* TODO:  Use a more appropriate error */
    assert(ptr < (((uintptr_t) 1u) << lg_max_memory));

    /* Check that this has not been allocated for some freak reason. */
    {
        amap_t::const_iterator lit =
            device_allocations_.lower_bound(device_ptr);
        if (!(lit == device_allocations_.end() ||
                reinterpret_cast<uintptr_t>(lit->first) > ptr ||
                reinterpret_cast<uintptr_t>(lit->first) +
                lit->second.size <= ptr)) {
            const void * const new_start  = device_ptr;
            const void * const new_end    =
                static_cast<const uint8_t *>(device_ptr) + size;
            const void * const old_start  = lit->first;
            const void * const old_end    =
                static_cast<const uint8_t *>(lit->first) + lit->second.size;

            assert((new_end > old_start && new_start <= old_end) ||
                   (old_end > new_start && old_end   <= new_end));

            char msg[256];
            int msgret = snprintf(msg, sizeof(msg),
                "Overlapping device memory allocation detected.  New "
                "allocation is on [%p, %p); old allocation is [%p, %p).",
                new_start, new_end, old_start, old_end);
            // sizeof(msg) is small, so the cast is safe.
            assert(msgret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);

            return;
        }

        amap_t::const_iterator uit =
            device_allocations_.upper_bound(device_ptr);
        if (!(uit == device_allocations_.end() ||
                ptr + size <= reinterpret_cast<uintptr_t>(lit->first))) {
            const void * const new_start  = device_ptr;
            const void * const new_end    =
                static_cast<const uint8_t *>(device_ptr) + size;
            const void * const old_start  = lit->first;
            const void * const old_end    =
                static_cast<const uint8_t *>(lit->first) + lit->second.size;

            assert((new_end > old_start && new_start <= old_end) ||
                   (old_end > new_start && old_end   <= new_end));

            char msg[256];
            int msgret = snprintf(msg, sizeof(msg),
                "Overlapping device memory allocation detected.  New "
                "allocation is on [%p, %p); old allocation is [%p, %p).",
                new_start, new_start, old_start, old_end);
            // sizeof(msg) is small, so the cast is safe.
            assert(msgret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);

            return;
        }

        device_aux_t aux;
        aux.size = size;
        aux.must_free = must_free;

        device_allocations_.insert(amap_t::value_type(device_ptr, aux));
        udevice_allocations_.insert(aumap_t::value_type(
            static_cast<const uint8_t *>(device_ptr) + size,
            size));
    }

    /**
     * Remove this block from the recently freed list (if needed).
     */
    remove_recent_free(device_ptr, size);

    /* The authoritative copy of master_ is always kept on the host, so we can
     * make any needed modifications and then push it.
     */
    const size_t chunk_bytes = 1 << lg_chunk_bytes;
    const size_t first_chunk =  ptr                           >> lg_chunk_bytes;
    const size_t last_chunk  = (ptr + size + chunk_bytes - 1) >> lg_chunk_bytes;
    bool master_dirty        = false;

    typedef global_memcheck_state::chunk_updates_t updates_t;
    updates_t updates;

    /* bool existing_dirtied    = false; */

    for (size_t i = first_chunk; i < last_chunk; i++) {
        bool has_update = false;
        const size_t first_bytes =
            std::max(ptr, i * chunk_bytes) & (chunk_bytes - 1);
        const size_t last_bytes =
            std::min(ptr + size, (i + 1) * chunk_bytes) & (chunk_bytes - 1);

        if (achunks_[i] == default_achunk_) {
            if (first_bytes == 0 && last_bytes == 0) {
                /* Reuse initialized chunk. */
                achunks_[i] = initialized_achunk_;
            } else {
                assert(i == first_chunk || i == last_chunk - 1);
                /* Allocate another chunk */
                apool_t::handle_t * c = apool_.allocate();
                achunks_[i] = c;
            }

            has_update = true;
        }

        if (vchunks_[i] == default_vchunk_) {
            /* Allocate another chunk */
            vpool_t::handle_t * c = vpool_.allocate();
            vchunks_[i] = c;

            has_update = true;
        } /* else {
            existing_dirtied = true;
        } */

        if (has_update) {
            metadata_ptrs tmp;
            tmp.adata = achunks_[i]->gpu();
            tmp.vdata = vchunks_[i]->gpu();

            updates.push_back(updates_t::value_type(i, tmp));
            master_dirty = true;
        }
    }

    /**
     * TODO:  Consult existing_dirtied flag here and conditionally download
     * the state of the buffers.  We upload whole blocks, but this can have
     * the adverse side effect of not downloading whole blocks when we freshly
     * allocate a chunk out of an existing block.
     */
    /* if (existing_dirtied) { */
        apool_.to_host(&achunks_[first_chunk], &achunks_[last_chunk]);
        vpool_.to_host(&vchunks_[first_chunk], &vchunks_[last_chunk]);
    /* } */

    for (size_t i = first_chunk; i < last_chunk; i++) {
        assert(achunks_[i] != default_achunk_);
        chunks_aux_[i].allocations++;

        /* Set addressability bits...
         *
         * Starting at (ptr       ) or  i      * chunk_bytes,     whichever is
         *  greater.
         * Ending at   (ptr + size) or (i + 1) * chunk_bytes - 1, whichever is
         *  less.
         */
        uintptr_t start =
            std::max(ptr,         i * chunk_bytes)     & (chunk_bytes - 1);
        uintptr_t end   =
            std::min(ptr + size - i * chunk_bytes, chunk_bytes);

        /* These form the ranges for applying memset to a_data.  They may
         * differ from start and end because the last byte on each end needs to
         * be adjusted. */
        uintptr_t astart, aend;

        if ((start & (CHAR_BIT - 1)) != 0) {
            /* This write isn't a whole byte */
            uint8_t mask    =
                static_cast<uint8_t>(0xFF << (start & (CHAR_BIT - 1)));
            astart          = (start + CHAR_BIT - 1) & ~((uintptr_t) CHAR_BIT);

            /* If the new start > end, we need to unset the high order bits */
            if (astart > end) {
                assert(astart - end < CHAR_BIT);
                /* This sets the high order bits which are not accessible */
                const uint8_t end_mask = static_cast<uint8_t>(0xFF <<
                    (CHAR_BIT - (astart - end)));

                /* Mask these out */
                mask = mask & static_cast<uint8_t>(~end_mask);
            }

            /* Set this bit */
            achunks_[i]->host()->a_data[start / CHAR_BIT] = mask;
        } else {
            astart = start;
        }

        if (end > astart && (end & (CHAR_BIT - 1)) != 0) {
            uint8_t mask = static_cast<uint8_t>(~(0xFF <<
                (end & (CHAR_BIT - 1))));
            achunks_[i]->host()->a_data[end / CHAR_BIT] = mask;
            aend = end & ~((uintptr_t) CHAR_BIT - 1);
        } else {
            aend = end;
        }

        if (aend > astart && (aend < chunk_bytes || astart > 0)) {
            assert(achunks_[i] != initialized_achunk_);
            memset(static_cast<uint8_t *>(achunks_[i]->host()->a_data) +
                astart / CHAR_BIT, 0xFF, (aend - astart) / CHAR_BIT);
        }

        if (valgrind_) {
            vdata_chunk * vdata = vchunks_[i]->host();
            uint8_t * vptr = reinterpret_cast<uint8_t *>(vdata->v_data) + start;
            assert(end - start <= sizeof(vdata->v_data));
            memset(vptr, 0xFF, end - start);
        }
    }

    /*
     * Just in case the CUDA runtime didn't, mprotect these pages so we can
     * handle SIGSEGV's to notice erroneous dereferences.
     *
     * Rounding to the bottom of the page requires we extend the length of the
     * protected region a bit.
     */
    const size_t diff = ptr & (pagesize_ - 1);
    int mret = mprotect((void *) (ptr - diff), size + diff, PROT_NONE);
    switch (mret) {
        case EACCES:
        case EINVAL:
        case ENOMEM:
        default:
            break;
    }

    apool_.to_gpu(&achunks_[first_chunk], &achunks_[last_chunk]);
    vpool_.to_gpu(&vchunks_[first_chunk], &vchunks_[last_chunk]);
    assert(master_dirty ^ updates.empty());
    if (master_dirty) {
        state_->update_master(device_, true, updates);
    }

    if (valgrind_ && must_free) {
        VALGRIND_MALLOCLIKE_BLOCK(device_ptr, size, 0, 0);
        (void) VALGRIND_MAKE_MEM_NOACCESS(device_ptr, size);
    }
}

cudaError_t cuda_context_memcheck::cudaMalloc3D(struct cudaPitchedPtr
        *pitchedDevPtr, struct cudaExtent extent) {
    cudaError_t ret = callout::cudaMalloc3D(pitchedDevPtr, extent);
    if (ret != cudaSuccess) {
        return ret;
    }

    /**
     * We expect CUDA to segfault on a non-zero allocation request with a
     * NULL pitchedDevPtr, but to be safe...
     */
    size_t extent_size = extent.depth * extent.height * extent.width;
    assert(extent_size == 0 || pitchedDevPtr);

    struct cudaPitchedPtr * ptr = pitchedDevPtr;
    size_t allocd_size = ptr->pitch   * ptr->xsize    * ptr->ysize;
    assert(allocd_size >= extent_size);

    scoped_lock lock(mx_);
    add_device_allocation(ptr->ptr, allocd_size, true);
    return setLastError(cudaSuccess);
}

internal::array_t::array_t(struct cudaChannelFormatDesc _desc) :
        desc(_desc), x(0), y(0), z(0), validity(NULL) { }

internal::array_t::array_t(struct cudaChannelFormatDesc _desc,
        struct cudaExtent extent) : desc(_desc), x(0), y(0), z(0),
        validity(NULL) {
    if (extent.width == 0) {
        return;
    }

    x = extent.width;

    if (extent.height == 0) {
        return;
    }

    y = extent.height;

    if (extent.depth == 0) {
        return;
    }

    z = extent.depth;
}

size_t internal::array_t::size() const {
    if (x == 0) {
        return 0;
    }

    if (y == 0) {
        return x;
    }

    if (z == 0) {
        return x * y;
    }

    return x * y * z;
}

internal::array_t::~array_t() {
    (void) callout::cudaFree(validity);
}

cudaError_t cuda_context_memcheck::cudaMalloc3DArray(struct cudaArray** array,
        const struct cudaChannelFormatDesc *desc, struct cudaExtent extent,
        unsigned int flags) {
    if (!(array)) {
        /* No place to store the pointer. */
        return cudaErrorInvalidValue;
    }

    size_t units;
    if (extent.depth > 0) {
        units = extent.depth * extent.height * extent.width;
    } else if (extent.height > 0) {
        units = extent.height * extent.width;
    } else {
        units = extent.width;
    }

    if (units == 0) {
        /* Nothing to do */
        *array  = NULL;
        return cudaSuccess;
    }

    if (!(desc)) {
        /* Unable to proceed without a descriptor. */
        return cudaErrorInvalidValue;
    }

    /**
     * Grab device properties.  This should be a fast call into our
     * implementation which should have this cached.
     */
    struct cudaDeviceProp prop;
    cudaError_t prop_ret = cudaGetDeviceProperties(&prop, device_);
    if (prop_ret != cudaSuccess) {
        return prop_ret;
    }

    /**
     * Check that the values are in-range per the documentation.
     */
    const unsigned dims = (extent.width ? 1 : 0) + (extent.height ? 1 : 0) +
        (extent.depth ? 1 : 0);
    assert(dims > 0); /* Since units > 0 */
    size_t max_x = 0, max_y = 0, max_z = 0;
    switch (dims) {
        case 1:
            assert(prop.maxTexture1D >= 0);
            max_x = (size_t) prop.maxTexture1D;
            break;
        case 2:
            assert(prop.maxTexture2D[0] >= 0);
            assert(prop.maxTexture2D[1] >= 0);
            max_x = (size_t) prop.maxTexture2D[0];
            max_y = (size_t) prop.maxTexture2D[1];
            break;
        case 3:
            assert(prop.maxTexture3D[0] >= 0);
            assert(prop.maxTexture3D[1] >= 0);
            assert(prop.maxTexture3D[2] >= 0);
            max_x = (size_t) prop.maxTexture3D[0];
            max_y = (size_t) prop.maxTexture3D[1];
            max_z = (size_t) prop.maxTexture3D[2];
            break;
    }

    if (extent.width > max_x || extent.height > max_y ||
            extent.depth > max_z) {
        return cudaErrorInvalidValue;
    }

    if (desc->x < 0 || desc->y < 0 || desc->z < 0 || desc->w < 0) {
        return cudaErrorInvalidChannelDescriptor;
    }

    /* Safe due to the above check. */
    const size_t bits   = (size_t) desc->x + (size_t) desc->y +
        (size_t) desc->z + (size_t) desc->w;
    /* Round up */
    const size_t bytes  = (bits + CHAR_BIT - 1) / CHAR_BIT;
    const size_t total  = units * bytes;

    /**
     * Allocate.
     */
    cudaError_t ret = callout::cudaMalloc3DArray(array, desc, extent, flags);
    if (ret != cudaSuccess) {
        *array = NULL;
        return ret;
    }

    internal::array_t * details = new internal::array_t(*desc, extent);
    details->allocator      = array_malloc3d;
    details->allocation_bt  = backtrace_t::instance();

    ret = callout::cudaMalloc(&details->validity, total);
    if (ret != cudaSuccess) {
        /* Failed.  Cleanup. */
        cudaFreeArray(*array);
        *array = NULL;

        delete details;
        return ret;
    }

    ret = callout::cudaMemset(details->validity, 0, total);
    if (ret != cudaSuccess) {
        /* Failed.  Cleanup. */
        cudaFreeArray(*array);
        *array = NULL;

        delete details;
        return ret;
    }

    scoped_lock lock(mx_);
    opaque_dimensions_.insert(oamap_t::value_type(*array, details));
    return setLastError(cudaSuccess);
}

cudaError_t cuda_context_memcheck::cudaMallocArray(struct cudaArray** array,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags) {
    if (!(array)) {
        /* No place to store the pointer. */
        return cudaErrorInvalidValue;
    }

    size_t units;
    if (height > 0) {
        units = width * height;
    } else {
        units = width;
    }

    if (units == 0) {
        /* Nothing to do */
        *array  = NULL;
        return cudaSuccess;
    }

    if (!(desc)) {
        /* Unable to proceed without a descriptor. */
        return cudaErrorInvalidValue;
    }

    /**
     * Grab device properties.  This should be a fast call into our
     * implementation which should have this cached.
     */
    struct cudaDeviceProp prop;
    cudaError_t prop_ret = cudaGetDeviceProperties(&prop, device_);
    if (prop_ret != cudaSuccess) {
        return prop_ret;
    }

    /**
     * Check that the values are in-range per the documentation.
     */
    const unsigned dims =
        (width ? 1 : 0) + (height ? 1 : 0);
    assert(dims > 0); /* Since units > 0 */
    size_t max_x = 0, max_y = 0;
    switch (dims) {
        case 1:
            assert(prop.maxTexture1D >= 0);
            max_x = (size_t) prop.maxTexture1D;
            break;
        case 2:
            assert(prop.maxTexture2D[0] >= 0);
            assert(prop.maxTexture2D[1] >= 0);
            max_x = (size_t) prop.maxTexture2D[0];
            max_y = (size_t) prop.maxTexture2D[1];
            break;
    }

    if (width > max_x || height > max_y) {
        return cudaErrorInvalidValue;
    }

    if (desc->x < 0 || desc->y < 0 || desc->z < 0 || desc->w < 0) {
        return cudaErrorInvalidChannelDescriptor;
    }

    /* Safe due to the above check. */
    const size_t bits   = (size_t) desc->x + (size_t) desc->y +
        (size_t) desc->z + (size_t) desc->w;
    /* Round up */
    const size_t bytes  = (bits + CHAR_BIT - 1) / CHAR_BIT;
    const size_t total  = units * bytes;

    cudaError_t ret = callout::cudaMallocArray(array, desc, width, height,
        flags);
    if (ret != cudaSuccess) {
        return ret;
    }

    internal::array_t * details = new internal::array_t(*desc);
    details->x              = width;
    details->y              = height;
    details->allocator      = array_malloc;
    details->allocation_bt  = backtrace_t::instance();

    ret = callout::cudaMalloc(&details->validity, total);
    if (ret != cudaSuccess) {
        /* Failed.  Cleanup. */
        cudaFreeArray(*array);
        *array = NULL;

        delete details;
        return ret;
    }

    ret = callout::cudaMemset(details->validity, 0, total);
    if (ret != cudaSuccess) {
        /* Failed.  Cleanup. */
        cudaFreeArray(*array);
        *array = NULL;

        delete details;
        return ret;
    }

    scoped_lock lock(mx_);
    opaque_dimensions_.insert(oamap_t::value_type(*array, details));
    return setLastError(cudaSuccess);
}

cudaError_t cuda_context_memcheck::cudaMallocHost(void **ptr, size_t size) {
    if (!(ptr)) {
        return cudaErrorInvalidValue;
    }

    cudaError_t ret = callout::cudaMallocHost(ptr, size);
    if (ret != cudaSuccess) {
        return ret;
    }

    void * const ptr_ = *ptr;

    scoped_lock lock(mx_);

    host_aux_t aux;
    aux.allocation_type = allocation_mallochost;
    aux.flags           = cudaHostAllocDefault;
    aux.size            = size;

    host_allocations_.insert(ahmap_t::value_type(ptr_, aux));

    /**
     * Prepare upperbound.
     */
    const uint8_t * upper_ptr = static_cast<const uint8_t *>(ptr_) + size;
    uhost_allocations_.insert(acmap_t::value_type(upper_ptr, size));

    VALGRIND_MALLOCLIKE_BLOCK(ptr_, size, 0, 0);
    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaMallocPitch(void **devPtr,
        size_t *pitch, size_t width, size_t height) {
    if (!(devPtr)) {
        return cudaErrorInvalidValue;
    }

    if (!(pitch)) {
        return cudaErrorInvalidValue;
    }

    cudaError_t ret = cuda_context::cudaMallocPitch(devPtr, pitch, width,
        height);
    if (ret == cudaSuccess) {
        scoped_lock lock(mx_);
        const size_t size = *pitch * height;
        add_device_allocation(*devPtr, size, true);
    }

    return setLastError(ret);
}

bool cuda_context_memcheck::check_host_pinned(const void * ptr, size_t len,
        size_t * offset) const {
    size_t checked;
    for (checked = 0; checked < len; ) {
        /**
         * Find the upperbound of the block (if any) containing ptr + checked.
         */
        const void * p = static_cast<const uint8_t *>(ptr) + checked;
        acmap_t::const_iterator it = uhost_allocations_.lower_bound(p);
        if (it == uhost_allocations_.end()) {
            /**
             * There are no blocks containing p, so it cannot be pinned
             * approximately.
             */
            if (offset) {
                *offset = checked;
            }

            return false;
        }

        /**
         * We may have overshot, so check that the lower bound of this block
         * is less than or equal to p.
         */
        const void * lb = static_cast<const uint8_t *>(it->first) - it->second;
        if (lb > p) {
            if (offset) {
                *offset = checked;
            }

            return false;
        } else if (p == it->first) {
            ahmap_t::const_iterator jit = host_allocations_.find(p);
            if (jit == host_allocations_.end()) {
                if (offset) {
                    *offset = checked;
                }

                return false;
            }

            checked += jit->second.size;
        } else { /* else: p > lb */
            /**
             *
             * Advance for length of containing block beyond p.
             */
            size_t old_checked = checked;
            checked += it->second - (size_t)
                (static_cast<const uint8_t *>(p) -
                 static_cast<const uint8_t *>(lb));
            assert(checked > old_checked);
        }
    }

    return true;
}

cudaError_t cuda_context_memcheck::cudaMemcpy(void *dst, const void *src,
        size_t size, enum cudaMemcpyKind kind) {
    return cudaMemcpyImplementation(dst, src, size, kind, NULL);
}

cudaError_t cuda_context_memcheck::cudaMemcpyAsync(void *dst, const void *src,
        size_t size, enum cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpyImplementation(dst, src, size, kind, &stream);
}

cudaError_t cuda_context_memcheck::cudaMemcpyImplementation(void *dst,
        const void *src, size_t size, enum cudaMemcpyKind kind,
        cudaStream_t *stream) {
    void ** handle = stream ? reinterpret_cast<void **>(*stream) : NULL;

    scoped_lock lock(mx_);

    stream_map_t::iterator it = streams_.find(handle);
    internal::stream_t * const cs =
        (!(stream) || it == streams_.end()) ? NULL : it->second;

    if (kind == cudaMemcpyDefault) {
        const bool ddst = is_device_pointer(dst, NULL, NULL);
        const bool dsrc = is_device_pointer(src, NULL, NULL);

               if (ddst && dsrc) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (!ddst && dsrc) {
            kind = cudaMemcpyDeviceToHost;
        } else if (ddst && !dsrc) {
            kind = cudaMemcpyHostToDevice;
        } else if (!ddst && !dsrc) {
            kind = cudaMemcpyHostToHost;
        }
    }

    /**
     * If the transfer is asynchronous and between the device and the host,
     * the host memory must be pinned.  Per the CUDA documentation on
     * cudaMemcpyAsync:
     *
     * "It only works on page-locked host memory and returns an error if a
     *  pointer to pageable memory is passed as input."
     *
     * Empirically, these transfers do not return errors so we suppress
     * error codes originating from this point and just log it.  These checks
     * occur *after* we've checked addressibility.
     */

    switch (kind) {
    case cudaMemcpyDeviceToDevice:
        if (check_access_device(dst, size, true) &&
                check_access_device(src, size, true)) {
            cudaError_t ret;
            if (cs) {
                ret = callout::cudaMemcpyAsync(dst, src, size, kind,
                    cs->stream);
            } else {
                ret = callout::cudaMemcpy(dst, src, size, kind);
            }

            if (ret == cudaSuccess) {
                validity_copy(dst, src, size, cs);
            } else {
                /* Simply mark the region as invalid */
                validity_clear(dst, size, cs);
            }

            if (cs) {
                new internal::event_t(panoptes_created,
                    cudaEventDefault, cs, NULL);
            }

            return ret;
        } else {
            return cudaErrorInvalidValue;
        }
    case cudaMemcpyDeviceToHost: {
        if (!(check_access_device(src, size, true))) {
            /* Fallthrough to return. */
        } else if (!(check_access_host(dst, size))) {
            /* Valgrind is running and it found an error */
            raise(SIGSEGV);
        } else {
            /* Valgrind is okay with dst or Valgrind is not running */
            cudaError_t ret;
            if (cs) {
                size_t offset;
                if (!(check_host_pinned(dst, size, &offset))) {
                    char msg[128];
                    int msgret = snprintf(msg, sizeof(msg),
                        "cudaMemcpyAsync must use pinned host memory: "
                        "%p is not pinned at offset %zu.", dst, offset);
                    // sizeof(msg) is small, so the cast is safe.
                    assert(msgret < (int) sizeof(msg) - 1);
                    logger::instance().print(msg);
                }

                ret = callout::cudaMemcpyAsync(dst, src, size, kind,
                    cs->stream);
            } else {
                ret = callout::cudaMemcpy(dst, src, size, kind);
            }

            if (valgrind_) {
                if (ret == cudaSuccess) {
                    validity_download(dst, src, size, cs);
                } else {
                    /* Simply mark the region as invalid */
                    (void) VALGRIND_MAKE_MEM_UNDEFINED(dst, size);
                }
            }

            if (cs) {
                new internal::event_t(panoptes_created,
                    cudaEventDefault, cs, NULL);
            }

            return ret;
        }

        return cudaErrorInvalidValue;
    }
    case cudaMemcpyHostToDevice:
        if (!(check_access_device(dst, size, true))) {
            /* Fallthrough to return. */
        } else if (!(check_access_host(src, size))) {
            /* Valgrind is running and it found an error */
            raise(SIGSEGV);
        } else {
            /* Valgrind is okay with dst or Valgrind is not running */
            cudaError_t ret;

            if (cs) {
                size_t offset;
                if (!(check_host_pinned(src, size, &offset))) {
                    char msg[128];
                    int msgret = snprintf(msg, sizeof(msg),
                        "cudaMemcpyAsync must use pinned host memory: "
                        "%p is not pinned at offset %zu.", src, offset);
                    // sizeof(msg) is small, so the cast is safe.
                    assert(msgret < (int) sizeof(msg) - 1);
                    logger::instance().print(msg);
                }

                ret = callout::cudaMemcpyAsync(dst, src, size, kind,
                    cs->stream);
            } else {
                ret = callout::cudaMemcpy(dst, src, size, kind);
            }

            if (valgrind_) {
                if (ret == cudaSuccess) {
                    validity_upload(dst, src, size, cs);
                } else {
                    /* Simply mark the region as invalid */
                    validity_clear(dst, size, cs);
                }
            }

            if (cs) {
                new internal::event_t(panoptes_created,
                    cudaEventDefault, cs, NULL);
            }

            return ret;
        }

        return cudaErrorInvalidValue;
    case cudaMemcpyHostToHost:
        if (runtime_version_ >= 4010 /* 4.1 */) {
            /**
             *
             * For reasons that are not entirely clear, cudaMemcpyHostToHost in
             * CUDA 4.1 is quite adept at performing seemingly impossible
             * memory copies without segfault or error.  In an ideal world, we
             * would use our own memcpy here (as host-to-host predates CUDA)
             * but we can't perform impossible copies (NULL to NULL with
             * size > 0 segfaults for mere mortals).
             */
            return callout::cudaMemcpy(dst, src, size, kind);
        } else {
            memcpy(dst, src, size);
            return cudaSuccess;
        }
    default:
        return cudaErrorInvalidMemcpyDirection;
    }
}

cudaError_t cuda_context_memcheck::cudaMemset(void *devPtr, int value,
        size_t count) {
    if (count == 0) {
        // No-op
        return setLastError(cudaSuccess);
    } else if (!(check_access_device(devPtr, count, true))) {
        return setLastError(cudaErrorInvalidValue);
    }

    cudaError_t ret = callout::cudaMemset(devPtr, value, count);
    if (ret == cudaSuccess) {
        /* Mark as valid */
        validity_set(devPtr, count, NULL);
    }

    return setLastError(ret);
}

cudaError_t cuda_context_memcheck::cudaMemsetAsync(void *devPtr, int value,
        size_t count, cudaStream_t cs) {
    if (count == 0) {
        // No-op
        return setLastError(cudaSuccess);
    } else if (!(check_access_device(devPtr, count, true))) {
        return setLastError(cudaErrorInvalidValue);
    }

    internal::stream_t * stream;
    void ** handle = reinterpret_cast<void **>(cs);

    {
        // Scope of lock
        scoped_lock lock(mx_);

        stream_map_t::iterator sit = streams_.find(handle);
        if (sit == streams_.end()) {
            char msg[128];
            int ret = snprintf(msg, sizeof(msg),
                "cudaMemsetAsync called with invalid stream\n");
            // sizeof(msg) is small, so the cast is safe.
            assert(ret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);

            return cudaErrorInvalidValue;
        } else {
            stream = sit->second;
        }
    }


    cudaError_t ret = callout::cudaMemsetAsync(devPtr, value, count,
        stream->stream);
    if (ret == cudaSuccess) {
        /* Mark as valid */
        validity_set(devPtr, count, stream);
    }

    return setLastError(ret);
}

cudaError_t cuda_context_memcheck::cudaStreamCreate(cudaStream_t *pStream) {
    if (!(pStream)) {
        return cudaErrorInvalidValue;
    }

    internal::stream_t * stream_metadata = new internal::stream_t();
    cudaError_t ret = callout::cudaStreamCreate(&stream_metadata->stream);
    if (ret != cudaSuccess) {
        delete stream_metadata;
        return ret;
    }

    void ** handle = create_handle();

    scoped_lock lock(mx_);
    streams_.insert(stream_map_t::value_type(handle, stream_metadata));

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(handle);

    state_->register_stream(stream, device_);
    *pStream = stream;
    return ret;
}

cudaError_t cuda_context_memcheck::cudaStreamDestroy(cudaStream_t stream) {
    void ** handle = reinterpret_cast<void **>(stream);

    scoped_lock lock(mx_);
    stream_map_t::iterator it = streams_.find(handle);
    if (it == streams_.end()) {
        int device;
        if (state_->lookup_stream(stream, &device)) {
            /* The stream belongs to another context. */
            assert(device != device_);
            return global()->context((unsigned) device)->
                cudaStreamDestroy(stream);
        }

        /**
         * CUDA segfaults here pre-CUDA 5.0.
         */
        if (driver_version_ < 5000) {
            raise(SIGSEGV);
        }

        return cudaErrorUnknown;
    }

    internal::stream_t * stream_metadata = it->second;
    streams_.erase(it);
    state_->unregister_stream(stream);

    free_handle(handle);

    cudaError_t ret = stream_metadata->busy();
    delete stream_metadata;
    return ret;
}

cudaError_t cuda_context_memcheck::cudaStreamQuery(cudaStream_t stream) {
    void ** handle = reinterpret_cast<void **>(stream);

    scoped_lock lock(mx_);
    stream_map_t::iterator it = streams_.find(handle);
    if (it == streams_.end()) {
        int device;
        if (state_->lookup_stream(stream, &device)) {
            /* The stream belongs to another context. */
            assert(device != device_);
            return global()->context((unsigned) device)->
                cudaStreamQuery(stream);
        }

        /**
         * CUDA segfaults here pre-CUDA 5.0.
         */
        if (driver_version_ < 5000) {
            raise(SIGSEGV);
        }

        return cudaErrorUnknown;
    }

    internal::stream_t * stream_metadata = it->second;
    assert(stream_metadata);

    return stream_metadata->busy();
}

cudaError_t cuda_context_memcheck::cudaStreamSynchronize(cudaStream_t stream) {
    void ** handle = reinterpret_cast<void **>(stream);

    scoped_lock lock(mx_);
    stream_map_t::iterator it = streams_.find(handle);
    if (it == streams_.end()) {
        int device;
        if (state_->lookup_stream(stream, &device)) {
            /* The stream belongs to another context. */
            assert(device != device_);
            return global()->context((unsigned) device)->
                cudaStreamSynchronize(stream);
        }

        /**
         * CUDA segfaults here pre-CUDA 5.0.
         */
        if (driver_version_ < 5000) {
            raise(SIGSEGV);
        }

        return cudaErrorUnknown;
    }

    internal::stream_t * stream_metadata = it->second;
    assert(stream_metadata);

    return stream_metadata->synchronize();
}

cudaError_t cuda_context_memcheck::cudaStreamWaitEvent(cudaStream_t stream,
        cudaEvent_t event, unsigned int flags) {
    if (flags != 0) {
        return cudaErrorInvalidValue;
    }

    void ** shandle = reinterpret_cast<void **>(stream);
    void ** ehandle = reinterpret_cast<void **>(event);

    scoped_lock lock(mx_);

    cudaStream_t real_stream;
    stream_map_t::iterator sit = streams_.find(shandle);
    if (sit == streams_.end()) {
        int device;
        if (state_->lookup_stream(stream, &device)) {
            /* The stream belongs to another context. */
            assert(device != device_);
            return global()->context((unsigned) device)->
                cudaStreamWaitEvent(stream, event, flags);
        }

        real_stream = NULL;
    } else {
        real_stream = sit->second->stream;
    }

    event_map_t::iterator eit = events_.find(ehandle);
    if (eit == events_.end()) {
        /**
         * CUDA segfaults here.
         */
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaStreamWaitEvent event argument is invalid.\n");
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        raise(SIGSEGV);
        return cudaErrorInvalidResourceHandle;
    }

    return callout::cudaStreamWaitEvent(real_stream,
        eit->second->event, flags);
}

bool cuda_context_memcheck::is_device_pointer(const void * ptr,
        const void ** block_ptr, size_t * block_offset) const {
    const aumap_t::const_iterator it = udevice_allocations_.lower_bound(ptr);
    if (it == udevice_allocations_.end()) {
        return false;
    }

    const uint8_t * base = static_cast<const uint8_t *>(it->first);
    const bool is_device = (base - it->second <= ptr) && (ptr <= base);
    if (is_device && block_ptr && block_offset) {
        *block_ptr      = base - it->second;
        const uint8_t * uptr = static_cast<const uint8_t *>(ptr);
        assert(base >= uptr); /* Guaranteed by lower_bound(). */
        *block_offset   = it->second - (size_t) (base - uptr);
    }

    return is_device;
}

bool cuda_context_memcheck::validity_clear(const void * ptr, size_t len,
        internal::stream_t * stream) {
    if (!(valgrind_)) {
        // No-op.
        return true;
    }

    const size_t chunk_bytes = 1 << lg_chunk_bytes;

    const uintptr_t upptr    = reinterpret_cast<uintptr_t>(ptr);
    const size_t first_chunk =  upptr                         >> lg_chunk_bytes;
    const size_t last_chunk  = (upptr + len + chunk_bytes - 1)>> lg_chunk_bytes;

    for (size_t i = first_chunk, voffset = 0; i < last_chunk; i++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(vchunks_[i] != default_vchunk_);

        const uintptr_t chunk_start = i * chunk_bytes;
        const uintptr_t chunk_mask  = chunk_bytes - 1;

        uintptr_t start = std::max(upptr,        chunk_start) & chunk_mask;
        uintptr_t end   = std::min(upptr + len - chunk_start, chunk_bytes);
        assert(end >= start);
        uintptr_t diff  = end - start;

        /* Copy validity bits onto device from buffer */
        uint8_t * const vptr =
            reinterpret_cast<uint8_t *>(vchunks_[i]->gpu()->v_data) + start;

        cudaError_t ret;
        if (stream) {
            ret = callout::cudaMemsetAsync(vptr, 0xFF, end - start,
                stream->stream);
        } else {
            ret = callout::cudaMemset(vptr, 0xFF, end - start);
        }

        if (ret != cudaSuccess) {
            /* This should not happen as chunks should always be fully
             * addressible on the device.  We cannot overrun the host
             * buffer as:
             *
             * start in [0, chunk_bytes)
             * end   in [0, chunk_bytes]
             *
             * So end - start in [0, chunk_bytes].
             */
            assert(0 && "The impossible happened");
            return false;
        }

        voffset += diff;
    }

    return true;
}

bool cuda_context_memcheck::validity_copy(void * dst, const void * src,
        size_t len, internal::stream_t * stream) {
    if (!(valgrind_)) {
        // No-op.
        return true;
    }

    cudaStream_t cs;
    if (stream) {
        cs = stream->stream;
    } else {
        /**
         * TODO:  Have a pool for streams.
         */

        cudaError_t cs_ret = callout::cudaStreamCreate(&cs);
        if (cs_ret != cudaSuccess) {
            return false;
        }
    }

    const size_t chunk_bytes = 1 << lg_chunk_bytes;

    const uintptr_t upsrc    = reinterpret_cast<uintptr_t>(src);
    const size_t first_chunk =  upsrc                         >> lg_chunk_bytes;
    const size_t last_chunk  = (upsrc + len + chunk_bytes - 1)>> lg_chunk_bytes;

    for (size_t srcidx = first_chunk, voffset = 0; srcidx < last_chunk;
            srcidx++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(vchunks_[srcidx] != default_vchunk_);

        const uintptr_t chunk_start = srcidx * chunk_bytes;
        const uintptr_t chunk_mask  = chunk_bytes - 1;

        uintptr_t sstart = std::max(upsrc,        chunk_start) & chunk_mask;
        uintptr_t send   = std::min(upsrc + len - chunk_start, chunk_bytes);
        assert(send >= sstart);
        uintptr_t srcrem = send - sstart;
        if (srcrem == 0) {
            break;
        }

        /**
         * Look up chunk indexes for destination.
         */
        const uintptr_t updst  = reinterpret_cast<uintptr_t>(dst) + voffset;
        const size_t    dstidx = updst >> lg_chunk_bytes;
        assert(vchunks_[dstidx] != default_vchunk_);
        const uintptr_t dstoff = updst & (chunk_bytes - 1);
        /* dstrem: number of bytes remaining in this chunk (dstidx) */
        const uintptr_t dstrem = chunk_bytes - dstoff;

        if (dstrem >= srcrem) {
            /* Single copy. */
            uint8_t * vdst =
                reinterpret_cast<uint8_t *>(vchunks_[dstidx]->gpu()->v_data);
            uint8_t * vsrc =
                reinterpret_cast<uint8_t *>(vchunks_[srcidx]->gpu()->v_data);

            callout::cudaMemcpyAsync(vdst + dstoff, vsrc + sstart, srcrem,
                cudaMemcpyDeviceToDevice, cs);
        } else {
            /* First copy (remainder of first destination chunk) */
            uint8_t * vdst =
                reinterpret_cast<uint8_t *>(vchunks_[dstidx]->gpu()->v_data);
            uint8_t * vsrc =
                reinterpret_cast<uint8_t *>(vchunks_[srcidx]->gpu()->v_data);

            callout::cudaMemcpyAsync(vdst + dstoff, vsrc + sstart, dstrem,
                cudaMemcpyDeviceToDevice, cs);
            /*
             * Second copy (second destination chunk).  Writing starts at the
             * very beginning of the destination chunk (so the offset is 0).
             * Reading ends no later than the end of the source chunk, so no
             * concern has to be given for moving on to the next source chunk
             * right here.
             */
            const size_t nxtidx = dstidx + 1;
            assert(vchunks_[nxtidx] != default_vchunk_);
            callout::cudaMemcpyAsync(vchunks_[nxtidx]->gpu()->v_data,
                vsrc + sstart + dstrem, srcrem - dstrem,
                cudaMemcpyDeviceToDevice, cs);
        }

        voffset += srcrem;
    }

    if (!(stream)) {
        cudaError_t cs_ret;

        cs_ret = callout::cudaStreamSynchronize(cs);
        if (cs_ret != cudaSuccess) {
            return false;
        }

        cs_ret = callout::cudaStreamDestroy(cs);
        if (cs_ret != cudaSuccess) {
            return false;
        }
    }

    return true;
}

bool cuda_context_memcheck::validity_copy(void * dst,
        cuda_context_memcheck * dstCtx, const void * src,
        const cuda_context_memcheck * srcCtx, size_t count,
        internal::stream_t * stream) {
    if (!(valgrind_)) {
        // No-op.
        return true;
    }

    cudaStream_t cs;
    if (stream) {
        cs = stream->stream;
    } else {
        /**
         * TODO:  Have a pool for streams.
         */

        cudaError_t cs_ret = callout::cudaStreamCreate(&cs);
        if (cs_ret != cudaSuccess) {
            return false;
        }
    }

    const size_t chunk_bytes = 1 << lg_chunk_bytes;

    const uintptr_t upsrc    = reinterpret_cast<uintptr_t>(src);
    const size_t first_chunk =  upsrc                        >> lg_chunk_bytes;
    const size_t last_chunk  = (upsrc + count +chunk_bytes-1)>> lg_chunk_bytes;

    const int dstDevice = static_cast<int>(dstCtx->device_);
    const int srcDevice = static_cast<int>(srcCtx->device_);

    for (size_t srcidx = first_chunk, voffset = 0; srcidx < last_chunk;
            srcidx++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        const vpool_t::handle_t * const srcChunk = srcCtx->vchunks_[srcidx];
        assert(srcChunk != srcCtx->default_vchunk_);

        const uintptr_t chunk_start = srcidx * chunk_bytes;
        const uintptr_t chunk_mask  = chunk_bytes - 1;

        uintptr_t sstart = std::max(upsrc,          chunk_start) & chunk_mask;
        uintptr_t send   = std::min(upsrc + count - chunk_start, chunk_bytes);
        assert(send >= sstart);
        uintptr_t srcrem = send - sstart;
        if (srcrem == 0) {
            break;
        }

        /**
         * Look up chunk indexes for destination.
         */
        const uintptr_t updst  = reinterpret_cast<uintptr_t>(dst) + voffset;
        const size_t    dstidx = updst >> lg_chunk_bytes;

        vpool_t::handle_t * const dstChunk = dstCtx->vchunks_[dstidx];
        assert(dstChunk != dstCtx->default_vchunk_);
        const uintptr_t dstoff = updst & (chunk_bytes - 1);
        /* dstrem: number of bytes remaining in this chunk (dstidx) */
        const uintptr_t dstrem = chunk_bytes - dstoff;

        if (dstrem >= srcrem) {
            /* Single copy. */
            uint8_t * vdst =
                reinterpret_cast<uint8_t *>(dstChunk->gpu()->v_data);
            const uint8_t * vsrc =
                reinterpret_cast<const uint8_t *>(srcChunk->gpu()->v_data);

            callout::cudaMemcpyPeerAsync(vdst + dstoff, dstDevice,
                vsrc + sstart, srcDevice, srcrem, cs);
        } else {
            /* First copy (remainder of first destination chunk) */
            uint8_t * vdst =
                reinterpret_cast<uint8_t *>(dstChunk->gpu()->v_data);
            const uint8_t * vsrc =
                reinterpret_cast<const uint8_t *>(srcChunk->gpu()->v_data);

            callout::cudaMemcpyPeerAsync(vdst + dstoff, dstDevice,
                vsrc + sstart, srcDevice, dstrem, cs);
            /*
             * Second copy (second destination chunk).  Writing starts at the
             * very beginning of the destination chunk (so the offset is 0).
             * Reading ends no later than the end of the source chunk, so no
             * concern has to be given for moving on to the next source chunk
             * right here.
             */
            const size_t nxtidx = dstidx + 1;
            vpool_t::handle_t * const nxtChunk = dstCtx->vchunks_[nxtidx];
            assert(nxtChunk != dstCtx->default_vchunk_);
            callout::cudaMemcpyPeerAsync(nxtChunk->gpu()->v_data, dstDevice,
                vsrc + sstart + dstrem, srcDevice, srcrem - dstrem, cs);
        }

        voffset += srcrem;
    }

    if (!(stream)) {
        cudaError_t cs_ret;

        cs_ret = callout::cudaStreamSynchronize(cs);
        if (cs_ret != cudaSuccess) {
            return false;
        }

        cs_ret = callout::cudaStreamDestroy(cs);
        if (cs_ret != cudaSuccess) {
            return false;
        }
    }

    return true;
}

bool cuda_context_memcheck::validity_download(void * host, const void *
        gpu, size_t len, internal::stream_t * stream) const {
    if (!(valgrind_)) {
        // No-op
        return true;
    }

    /* Transfer validity bits from device */
    const size_t chunk_bytes = 1 << lg_chunk_bytes;

    /* Asynchronous copies require that we interact with pinned memory */
    void * buffer;
    {
        cudaError_t ret;
        ret = callout::cudaHostAlloc(&buffer, chunk_bytes,
            cudaHostAllocDefault);
        if (ret != cudaSuccess) {
            return false;
        }
        VALGRIND_MALLOCLIKE_BLOCK(buffer, chunk_bytes, 0, 0);
    }

    const uintptr_t upgpu    = reinterpret_cast<uintptr_t>(gpu);
    const size_t first_chunk =  upgpu                         >> lg_chunk_bytes;
    const size_t last_chunk  = (upgpu + len + chunk_bytes - 1)>> lg_chunk_bytes;

    /** TODO:  Pipeline */
    for (size_t i = first_chunk, voffset = 0; i < last_chunk; i++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(vchunks_[i] != default_vchunk_);

        const uintptr_t chunk_start = i * chunk_bytes;
        const uintptr_t chunk_mask  = chunk_bytes - 1;

        uintptr_t start = std::max(upgpu,        chunk_start) & chunk_mask;
        uintptr_t end   = std::min(upgpu + len - chunk_start, chunk_bytes);
        assert(end >= start);
        uintptr_t diff  = end - start;

        /* Copy validity bits off device into buffer */
        uint8_t * const vptr =
            reinterpret_cast<uint8_t *>(vchunks_[i]->gpu()->v_data) + start;

        cudaError_t ret;
        if (stream) {
            callout::cudaMemcpyAsync(buffer, vptr,
                end - start, cudaMemcpyDeviceToHost, stream->stream);
            /*
             * This needs to finish before we can touch buffer.  The advantage
             * of putting it on the stream at all is that we now wait for the
             * caller's previous work on the stream to finish.
             */
            ret = stream->synchronize();
        } else {
            ret = callout::cudaMemcpy(buffer, vptr,
                end - start, cudaMemcpyDeviceToHost);
        }

        if (ret != cudaSuccess) {
            /* This should not happen as chunks should always be fully
             * addressible on the device.  We cannot overrun the host
             * buffer as:
             *
             * start in [0, chunk_bytes)
             * end   in [0, chunk_bytes]
             *
             * So end - start in [0, chunk_bytes].
             */
            assert(0 && "The impossible happened");

            callout::cudaFreeHost(buffer);
            return false;
        }

        /* Transfer bits into Valgrind */
        unsigned set =
            VALGRIND_SET_VBITS(static_cast<uint8_t *>(host) + voffset,
            buffer, diff);
        if (set != 1) {
            assert(0 && "Valgrind failed to set validity bits.");
            return false;
        }

        voffset += diff;
    }

    {
        VALGRIND_FREELIKE_BLOCK(buffer, 0);
        cudaError_t ret = callout::cudaFreeHost(buffer);
        if (ret != cudaSuccess) {
            return false;
        }
    }

    return true;
}

bool cuda_context_memcheck::validity_set(const void * ptr, size_t len,
        internal::stream_t * stream) {
    if (!(valgrind_)) {
        // No-op.
        return true;
    }

    const size_t chunk_bytes = 1 << lg_chunk_bytes;

    const uintptr_t upptr    = reinterpret_cast<uintptr_t>(ptr);
    const size_t first_chunk =  upptr                      >> lg_chunk_bytes;
    const size_t last_chunk  = (upptr + len+chunk_bytes-1) >> lg_chunk_bytes;

    cudaStream_t cs;
    if (stream) {
        cs = stream->stream;
    } else {
        /** TODO Use a pool */
        cudaError_t cs_ret;
        cs_ret = callout::cudaStreamCreate(&cs);
        if (cs_ret != cudaSuccess) {
            return false;
        }
    }

    for (size_t i = first_chunk, voffset = 0; i < last_chunk; i++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(vchunks_[i] != default_vchunk_);

        const uintptr_t chunk_start = i * chunk_bytes;
        const uintptr_t chunk_mask  = chunk_bytes - 1;

        uintptr_t start = std::max(upptr,        chunk_start) & chunk_mask;
        uintptr_t end   = std::min(upptr + len - chunk_start, chunk_bytes);
        assert(end >= start);
        uintptr_t diff  = end - start;

        /* Write validity bits onto device */
        uint8_t * const chunk_ptr =
            reinterpret_cast<uint8_t *>(vchunks_[i]->gpu()->v_data) + start;
        const cudaError_t ret = callout::cudaMemsetAsync(chunk_ptr,
            0x0, end - start, cs);
        if (ret != cudaSuccess) {
            /* This should not happen as chunks should always be fully
             * addressible on the device.  We cannot overrun the host
             * buffer as:
             *
             * start in [0, chunk_bytes)
             * end   in [0, chunk_bytes]
             *
             * So end - start in [0, chunk_bytes].
             */
            assert(0 && "The impossible happened");
            return false;
        }

        voffset += diff;
    }

    if (!(stream)) {
        cudaError_t cs_ret;
        cs_ret = callout::cudaStreamSynchronize(cs);
        if (cs_ret != cudaSuccess) {
            return false;
        }

        cs_ret = callout::cudaStreamDestroy(cs);
        if (cs_ret != cudaSuccess) {
            return false;
        }
    }

    return true;
}

bool cuda_context_memcheck::validity_upload(void * gpu, const void *
        host, size_t len, internal::stream_t * stream) {
    if (!(valgrind_)) {
        // No-op
        return true;
    }

    /* Transfer validity bits to device */
    const size_t chunk_bytes = 1 << lg_chunk_bytes;

    /* Asynchronous copies require that we interact with pinned memory */
    void * buffer;
    {
        cudaError_t ret;
        ret = callout::cudaHostAlloc(&buffer, chunk_bytes,
            cudaHostAllocDefault);
        if (ret != cudaSuccess) {
            return false;
        }
        VALGRIND_MALLOCLIKE_BLOCK(buffer, chunk_bytes, 0, 0);
    }

    const uintptr_t upgpu    = reinterpret_cast<uintptr_t>(gpu);
    const size_t first_chunk =  upgpu                        >> lg_chunk_bytes;
    const size_t last_chunk  = (upgpu + len + chunk_bytes-1) >> lg_chunk_bytes;

    for (size_t i = first_chunk, voffset = 0; i < last_chunk; i++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(vchunks_[i] != default_vchunk_);

        const uintptr_t chunk_start = i * chunk_bytes;
        const uintptr_t chunk_mask  = chunk_bytes - 1;

        uintptr_t start = std::max(upgpu,        chunk_start) & chunk_mask;
        uintptr_t end   = std::min(upgpu + len - chunk_start, chunk_bytes);
        assert(end >= start);
        uintptr_t diff  = end - start;

        /* Transfer bits out of Valgrind */
        const uint8_t * host_chunk =
            static_cast<const uint8_t *>(host) + voffset;
        (void) VALGRIND_GET_VBITS(host_chunk, buffer, diff);

        /* Copy validity bits onto device from buffer */
        uint8_t * const chunk_ptr =
            reinterpret_cast<uint8_t *>(vchunks_[i]->gpu()->v_data) + start;

        cudaError_t ret;
        if (stream) {
            callout::cudaMemcpyAsync(chunk_ptr, buffer, diff,
                cudaMemcpyHostToDevice, stream->stream);

            /**
             * We're going to overwrite buffer shortly (and/or deallocate it),
             * so a sync is required.
             */
            ret = stream->synchronize();
        } else {
            ret = callout::cudaMemcpy(chunk_ptr, buffer, diff,
                cudaMemcpyHostToDevice);
        }

        if (ret != cudaSuccess) {
            /* This should not happen as chunks should always be fully
             * addressible on the device.  We cannot overrun the host
             * buffer as:
             *
             * start in [0, chunk_bytes)
             * end   in [0, chunk_bytes]
             *
             * So end - start in [0, chunk_bytes].
             */
            assert(0 && "The impossible happened");

            callout::cudaFreeHost(buffer);
            return false;
        }

        voffset += diff;
    }

    {
        VALGRIND_FREELIKE_BLOCK(buffer, 0);
        cudaError_t ret = callout::cudaFreeHost(buffer);
        if (ret != cudaSuccess) {
            return false;
        }
    }

    return true;
}

cudaError_t cuda_context_memcheck::cudaLaunch(const void *entry) {
    /**
     * If the entry name is invalid, fail.
     */
    if (entry == NULL) {
        /**
         * CUDA 4.2 returns cudaErrorInvalidDeviceFunction.  CUDA 4.1 and older
         * returns cudaErrorUnknown.
         */
        if (runtime_version_ >= 4020) {
            return cudaErrorInvalidDeviceFunction;
        } else {
            return cudaErrorUnknown;
        }
    }

    scoped_lock lock(mx_);
    /**
     * Go over call stack, pass in validity information and other auxillary
     * information.
     */
    const char * entry_name = get_entry_name(entry);
    global_t * g = global();

    if (entry_name == NULL) {
        /* Try the now depreciated interpretation of entry as the entry name
         * itself. */
        entry_name = static_cast<const char *>(entry);
    }

    global_t::entry_info_map_t::const_iterator it =
        g->entry_info_.find(entry_name);
    if (it == g->entry_info_.end()) {
        return setLastError(cudaErrorInvalidDeviceFunction);
    }

    if (call_stack().size() == 0) {
        /**
         * This isn't a specified return value in the CUDA 4.x documentation
         * for cudaLaunch, but this has been experimentally validated.
         */
        return cudaErrorInvalidConfiguration;
    }

    const size_t pcount = it->second.user_params;
    size_t offset = it->second.user_param_size;

    internal::call_t * call = call_stack().top();
    internal::check_t * checker = new internal::check_t(this, entry_name);

    /* Check validity of launch parameters. */
    VALGRIND_CHECK_VALUE_IS_DEFINED(call->gridDim.x);
    VALGRIND_CHECK_VALUE_IS_DEFINED(call->gridDim.y);
    VALGRIND_CHECK_VALUE_IS_DEFINED(call->gridDim.z);
    VALGRIND_CHECK_VALUE_IS_DEFINED(call->gridDim.x);
    VALGRIND_CHECK_VALUE_IS_DEFINED(call->gridDim.y);
    VALGRIND_CHECK_VALUE_IS_DEFINED(call->gridDim.z);
    VALGRIND_CHECK_VALUE_IS_DEFINED(call->sharedMem);

    { /* Shared memory */
    uintptr_t shared = call->sharedMem;
    offset = (offset + sizeof(shared) - 1u) & ~(sizeof(shared) - 1u);
    cudaSetupArgument(&shared, sizeof(shared), offset);
    offset += sizeof(shared);
    }

    {
        /* Double dynamic shared memory to provide space for validity
         * information. */
        const uint32_t align   = 8;
        const uint32_t aligned = (call->sharedMem + align - 1) & ~(align - 1);
        const uint32_t new_shared_mem = 2u * aligned;
        if (new_shared_mem > call->sharedMem) {
            call->sharedMem = new_shared_mem;
        } /* Else: The caller specified a sharedMem so large that we overflowed
                   a uint32_t.  This call will almost certainly fail once CUDA
                   sees it, so let it be passed into the library as-is. */
    }

    { /* Error count */
    void * error_count = checker->error_count;
    offset = (offset + sizeof(error_count) - 1u) & ~(sizeof(error_count) - 1u);
    cudaSetupArgument(&error_count, sizeof(error_count), offset);
    offset += sizeof(error_count);
    }

    { /* Error buffer */
    void * error_buffer = checker->error_buffer;
    offset = (offset + sizeof(error_buffer) - 1u) &
            ~(sizeof(error_buffer) - 1u);
    cudaSetupArgument(&error_buffer, sizeof(error_buffer), offset);
    offset += sizeof(error_buffer);
    }

    /* Validity info. */
    char buffer[4096];
    if (it->second.user_param_size > sizeof(buffer)) {
        /* TODO:  Log an error about Panoptes limits. */
        delete checker;

        delete call;
        call_stack().pop();

        return cudaErrorInvalidValue;
    }

    if (valgrind_) {
        memset(buffer, 0xFF, sizeof(buffer));

        /* Copy validity information into buffer. */
        const size_t acount = call->args.size() - 3u;
        for (size_t i = 0; i < acount; i++) {
            const internal::arg_t * arg = call->args[i];
            if (arg->size + arg->offset > sizeof(buffer)) {
                delete checker;
                delete call;
                call_stack().pop();

                return cudaErrorInvalidValue;
            }

            (void) VALGRIND_GET_VBITS(arg->arg,
                buffer + arg->offset, arg->size);
        }
    } else {
        /* Assume everything is valid. */
        memset(buffer, 0x0, sizeof(buffer));
    }

    /* Setup all of the arguments. */
    size_t voffset = 0;
    for (size_t i = 0; i < pcount; i++) {
        const param_t & param = it->second.function->params[i];
        const size_t alignment = param.has_align ? param.alignment : 1u;
        /* Align. */
        voffset = (voffset + alignment - 1u) & ~(alignment - 1u);
        offset  = (offset  + alignment - 1u) & ~(alignment - 1u);

        const size_t size = param.size();
        cudaSetupArgument(buffer + voffset, size, offset);

        /* Move forward. */
        voffset += size;
        offset  += size;
    }

    cudaError_t ret = cuda_context::cudaLaunch(entry);
    td & d = thread_data();
    if (ret != cudaSuccess) {
        d.stream_stack.pop();
        delete checker;
        return ret;
    }

    /* Add an event of our own onto the stream. */
    internal::stream_t * cs = d.stream_stack.top();
    d.stream_stack.pop();

    assert(cs);
    try {
        new internal::event_t(panoptes_created, cudaEventDefault, cs, checker);
    } catch (...) {
        delete checker;
        throw;
    }

    return ret;
}

cuda_context_memcheck::td::~td() { }

void cuda_context_memcheck::bind_validity_texref(internal::texture_t * texture,
        const struct textureReference * texref,
        const struct cudaChannelFormatDesc * desc, const void * validity_ptr,
        size_t size) {
    /* Bind validity pointer. */
    size_t internal_offset;
    CUresult curet = cuTexRefSetAddress(&internal_offset,
        texture->validity_texref, (CUdeviceptr) validity_ptr, size);
    if (curet != CUDA_SUCCESS) {
        return;
    }

    const unsigned int flags =
        (texref->normalized ? CU_TRSF_NORMALIZED_COORDINATES : 0) |
        CU_TRSF_READ_AS_INTEGER;

    curet = cuTexRefSetFlags(texture->validity_texref, flags);
    if (curet != CUDA_SUCCESS) {
        return;
    }

    CUfilter_mode filter_mode;
    switch (texref->filterMode) {
        case cudaFilterModePoint:
            filter_mode = CU_TR_FILTER_MODE_POINT;
            break;
        case cudaFilterModeLinear:
            filter_mode = CU_TR_FILTER_MODE_LINEAR;
            break;
    }

    curet = cuTexRefSetFilterMode(texture->validity_texref, filter_mode);
    if (curet != CUDA_SUCCESS) {
        return;
    }

    CUarray_format format    = CU_AD_FORMAT_UNSIGNED_INT32;
    const int n_components = (desc->x + desc->y + desc->z + desc->w + 31) / 32;
    curet = cuTexRefSetFormat(texture->validity_texref, format, n_components);
    if (curet != CUDA_SUCCESS) {
        return;
    }
}

cudaError_t cuda_context_memcheck::bind_validity_texref2d(
        internal::texture_t * texture, const struct textureReference * texref,
        const struct cudaChannelFormatDesc * desc, const void * validity_ptr,
        size_t pitch, size_t height) {
    CUDA_ARRAY_DESCRIPTOR driver_desc;
    driver_desc.Width       = pitch /
        ((desc->x + desc->y + desc->z + desc->w) / CHAR_BIT);
    driver_desc.Height      = height;
    driver_desc.Format      = CU_AD_FORMAT_UNSIGNED_INT32;
    driver_desc.NumChannels = (desc->x + desc->y + desc->z + desc->w + 31) / 32;

    const void *aligned_ptr = reinterpret_cast<const void *>(
        reinterpret_cast<uintptr_t>(validity_ptr) &
        ~(info_.textureAlignment - 1u));

    /* Bind validity pointer. */
    CUresult curet = cuTexRefSetAddress2D(texture->validity_texref, &driver_desc,
        (CUdeviceptr) aligned_ptr, pitch);
    if (curet != CUDA_SUCCESS) {
        return cuToCUDA(curet);
    }

    const unsigned int flags =
        (texref->normalized ? CU_TRSF_NORMALIZED_COORDINATES : 0) |
        CU_TRSF_READ_AS_INTEGER;

    curet = cuTexRefSetFlags(texture->validity_texref, flags);
    if (curet != CUDA_SUCCESS) {
        return cuToCUDA(curet);
    }

    CUfilter_mode filter_mode;
    switch (texref->filterMode) {
        case cudaFilterModePoint:
            filter_mode = CU_TR_FILTER_MODE_POINT;
            break;
        case cudaFilterModeLinear:
            filter_mode = CU_TR_FILTER_MODE_LINEAR;
            break;
    }

    curet = cuTexRefSetFilterMode(texture->validity_texref, filter_mode);
    if (curet != CUDA_SUCCESS) {
        return cuToCUDA(curet);
    }

    for (int dim = 0; dim < 2; dim++) {
        CUaddress_mode mode;
        switch (texref->addressMode[dim]) {
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
            default:
                return cudaErrorInvalidValue;
        }

        curet = cuTexRefSetAddressMode(texture->validity_texref, dim, mode);
        if (curet != CUDA_SUCCESS) {
            return cuToCUDA(curet);
        }
    }

    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaBindTexture(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t size) {
    cudaError_t ret = cuda_context::cudaBindTexture(
        offset, texref, devPtr, desc, size);
    if (ret != cudaSuccess) {
        return ret;
    }

    internal::texture_t *tex;
    const void *validity_ptr;
    ret = get_validity_texture(texref, devPtr, size, &tex, &validity_ptr);
    if (ret != cudaSuccess) {
        return ret;
    }

    bind_validity_texref(tex, texref, desc, validity_ptr, size);
    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaBindTexture2D(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        size_t pitch) {
    cudaError_t ret = cuda_context::cudaBindTexture2D(offset, texref, devPtr,
        desc, width, height, pitch);
    if (ret != cudaSuccess) {
        return ret;
    }

    internal::texture_t *tex;
    const void *validity_ptr;
    ret = get_validity_texture(texref, devPtr, height * pitch, &tex,
        &validity_ptr);
    if (ret != cudaSuccess) {
        return ret;
    }

    return bind_validity_texref2d(tex, texref, desc, validity_ptr, pitch,
        height);
}

cudaError_t cuda_context_memcheck::get_validity_texture(
        const struct textureReference *texref,
        const void *devPtr, size_t size, internal::texture_t **tex,
        const void **validity_ptr) {
    scoped_lock lock(mx_);
    texture_map_t::iterator it = bound_textures_.find(texref);
    if (it == bound_textures_.end()) {
        /* Add a new texture record. */
        it = bound_textures_.insert(texture_map_t::value_type(
            texref, new internal::texture_t())).first;
    } else {
        release_texture(it->second, texref);
    }

    it->second->bound_pointer   = devPtr;
    it->second->bound_size      = size;
    it->second->binding         = backtrace_t::instance();

    *tex = it->second;
    load_validity_texref(it->second, texref);

    /* Note which memory regions have been bound to this texture. */
    const uint8_t * const uptr = static_cast<const uint8_t *>(devPtr);

    size_t asize, coffset;
    for (coffset = 0; coffset < size; ) {
        const uint8_t * const search = uptr + coffset;

        const void * aptr = NULL;
        asize = 0;

        aumap_t::const_iterator jit = udevice_allocations_.upper_bound(search);
        if (jit == udevice_allocations_.end()) {
            amap_t::iterator kit = device_allocations_.lower_bound(search);
            if (kit != device_allocations_.end()) {
                aptr  = kit->first;
                asize = kit->second.size;

                kit->second.bindings.insert(texref);
            }
        } else {
            aptr  = static_cast<const uint8_t *>(jit->first) - jit->second;
            asize = jit->second;

            amap_t::iterator kit = device_allocations_.find(aptr);
            if (kit == device_allocations_.end()) {
                assert(0 && "The impossible happened.");
                break;
            } else {
                kit->second.bindings.insert(texref);
            }
        }

        if (aptr == NULL) {
            break;
        } else {
            it->second->allocations.insert(aptr);
        }

        const uint8_t * base  = static_cast<const uint8_t *>(aptr);
        if (base > search) {
            break;
        }
        const size_t    head  = (size_t) (search - base);
        if (head > size - coffset) {
            break;
        }

        const size_t    rem   = std::min(asize - head, size - coffset);
        coffset += rem;
    }

    /**
     * Print messages if there was an error.
     */
    if (coffset != size) {
        if (coffset == 0) {
            // Invalid read starts at usrc + coffset
            char msg[256];
            int pret;
            if (is_recently_freed(devPtr)) {
                pret = snprintf(msg, sizeof(msg), "Invalid device pointer "
                    " of size %zu bound to texture.\n"
                    " Address %p is part of (recently) free'd allocation.",
                    size, devPtr);
            } else {
                pret = snprintf(msg, sizeof(msg), "Invalid device pointer "
                    " of size %zu bound to texture.\n"
                    " Address %p is not malloc'd or (recently) free'd.\n"
                    " Possible host pointer?", size, devPtr);
            }

            // sizeof(msg) is small, so the cast is safe.
            assert(pret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);
        } else {
            // Invalid read starts at usrc + coffset
            //
            // Technically, we don't know whether the full read of
            // (len - coffset) additional bytes is invalid.
            char msg[128];
            int pret = snprintf(msg, sizeof(msg), "Texture bound %zu bytes "
                "beyond allocation.\n"
                "Address %p is 0 bytes after a block of size %zu alloc'd\n",
                size - coffset, uptr + coffset, asize);
            // sizeof(msg) is small, so the cast is safe.
            assert(pret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);
        }
    }

    const size_t chunk_bytes = 1u << lg_chunk_bytes;
    const uintptr_t uiptr = reinterpret_cast<uintptr_t>(devPtr);
    const size_t chunk_start =  uiptr                           >> lg_chunk_bytes;
    const size_t chunk_end   = (uiptr + size + chunk_bytes - 1) >> lg_chunk_bytes;

    if (chunk_start < chunk_end) {
        /*
         * Coalesce validity blocks.
         *
         * Verify that only a single allocation spans these chunks.
         * TODO: Add support to merge adjacent blocks as necessary.
         */
        bool single_allocation = true;
        bool normal_chunks = false;
        std::set<size_t> large_owners;
        size_t bad_chunk;
        for (size_t i = chunk_start; i < chunk_end; i++) {
            if (chunks_aux_[i].large_chunk) {
                large_owners.insert(chunks_aux_[i].owner_index);
                if (large_owners.size() >= 1) {
                    bad_chunk = i;
                    break;
                }
            } else if (chunks_aux_[i].allocations != 1) {
                single_allocation = false;
                bad_chunk = i;
            } else {
                normal_chunks = true;
            }
        }

        if (!(normal_chunks) && large_owners.size() == 1) {
            /* Do nothing, as we already have a large allocation. */
        } else if (single_allocation && large_owners.size() == 0) {
            /* Nothing else can interact with this device. */
            cudaError_t sret = callout::cudaDeviceSynchronize();
            if (sret != cudaSuccess) {
                return sret;
            }

            /* Allocate buffers spanning the range. */
            const size_t chunks = chunk_end - chunk_start + 1;
            chunk_aux_t & root = chunks_aux_[chunk_start];
            sret = callout::cudaMalloc((void **) &root.groot,
                sizeof(*root.groot) * chunks);
            if (sret != cudaSuccess) {
                return sret;
            }

            sret = callout::cudaMallocHost((void **) &root.hroot,
                sizeof(*root.hroot) * chunks);
            if (sret != cudaSuccess) {
                return sret;
            }

            /* Copy current validity bits. */
            typedef global_memcheck_state::chunk_updates_t updates_t;
            updates_t updates;

            for (size_t i = chunk_start; i < chunk_end; i++) {
                chunk_aux_t & c = chunks_aux_[i];
                c.large_chunk = true;
                c.owner_index = chunk_start;

                const size_t index_offset = i - chunk_start;
                sret = callout::cudaMemcpy(root.groot + index_offset,
                    vchunks_[i]->gpu(), sizeof(*root.groot),
                    cudaMemcpyDeviceToDevice);
                if (sret != cudaSuccess) {
                    return sret;
                }

                c.handle = new vpool_t::handle_t(root.hroot + index_offset,
                    root.groot + index_offset);
                vpool_.free(vchunks_[i]);
                vchunks_[i] = c.handle;

                metadata_ptrs tmp;
                tmp.adata = achunks_[i]->gpu();
                tmp.vdata = vchunks_[i]->gpu();

                updates.push_back(updates_t::value_type(i, tmp));
            }

            if (updates.size() > 0) {
                state_->update_master(device_, true, updates);
            }
        } else {
            char msg[128];
            int pret = snprintf(msg, sizeof(msg), "Instrumentation limit "
                "encountered.\nPointer %p spans chunks containing other "
                "allocations (chunk index %zu).\n", uptr, bad_chunk);
            // sizeof(msg) is small, so the cast is safe.
            assert(pret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);
        }
    } else {
        assert(chunk_start == chunk_end);
    }

    const size_t voffset = uiptr - chunk_start * chunk_bytes;
    vpool_t::handle_t * vhandle = vchunks_[chunk_start];
    assert(vhandle != default_vchunk_);
    *validity_ptr = reinterpret_cast<uint8_t *>(vhandle->gpu()->v_data) +
        voffset;

    return cudaSuccess;
}

cudaError_t cuda_context_memcheck::cudaBindTextureToArray(
        const struct textureReference *texref, const struct cudaArray *array,
        const struct cudaChannelFormatDesc *desc) {
    internal::array_t * internal_array = NULL;

    {
        scoped_lock lock(mx_);
        oamap_t::iterator it = opaque_dimensions_.find(array);
        if (it == opaque_dimensions_.end()) {
            char msg[128];
            int ret = snprintf(msg, sizeof(msg), "cudaBindTextureToArray "
                "called on invalid cudaArray %p.", array);
            // sizeof(msg) is small, so the cast is safe.
            assert(ret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);

            /* Permit cuda_context to be called for subsequent failure. */
        } else {
            internal_array = it->second;
        }
    }

    cudaError_t ret =
        cuda_context::cudaBindTextureToArray(texref, array, desc);
    if (ret != cudaSuccess) {
        return ret;
    }

    scoped_lock lock(mx_);
    texture_map_t::iterator it = bound_textures_.find(texref);
    if (it == bound_textures_.end()) {
        /* Add a new texture record. */
        it = bound_textures_.insert(texture_map_t::value_type(
            texref, new internal::texture_t())).first;
    } else {
        release_texture(it->second, texref);
    }

    it->second->array_bound = true;
    it->second->bound_array = internal_array;
    it->second->binding     = backtrace_t::instance();

    load_validity_texref(it->second, texref);

    if (internal_array) {
        internal_array->bindings.insert(texref);
    }

    bind_validity_texref(it->second, texref, desc, internal_array->validity,
        internal_array->size());
    return ret;
}

void cuda_context_memcheck::load_validity_texref(internal::texture_t * texture,
        const struct textureReference * texref) {
    /* Retrieve the validity texture. */
    if (texture->has_validity_texref) {
        return;
    }

    internal::modules_t::texture_map_t::iterator jit =
        modules_->textures.find(texref);
    if (jit == modules_->textures.end()) {
        assert(0 && "Inconsistent texture index state.");
        return;
    }

    internal::module_t * const mod = jit->second;
    internal::module_t::texture_map_t::iterator kit =
        mod->textures.find(texref);
    if (kit == mod->textures.end()) {
        assert(0 && "Inconsistent texture index state.");
        return;
    }

    internal::module_t::texture_t * tex = kit->second;

    std::string s(internal::__texture_prefix);
    s += tex->deviceName;

    CUresult curet = cuModuleGetTexRef(&texture->validity_texref,
        mod->module, s.c_str());
    if (curet != CUDA_SUCCESS) {
        return;
    }

    texture->has_validity_texref = true;
}

cudaError_t cuda_context_memcheck::cudaUnbindTexture(
        const struct textureReference *texref) {
    cudaError_t ret = cuda_context::cudaUnbindTexture(texref);

    /* Since cudaUnbindTexture does not fail if the texture was never bound,
     * we must handle that gracefully. */
    scoped_lock lock(mx_);
    texture_map_t::iterator it = bound_textures_.find(texref);
    if (it != bound_textures_.end()) {
        release_texture(it->second, texref);
        delete it->second;

        bound_textures_.erase(it);
    }

    return ret;
}

void cuda_context_memcheck::release_texture(internal::texture_t * texture,
        const struct textureReference * texref) {
    assert(!(texture->bound_pointer && texture->array_bound));
    assert(texture->bound_pointer || texture->array_bound);

    if (texture->bound_pointer) {
        /* Cleanup old record. */
        for (internal::texture_t::allocation_vt::iterator it =
                texture->allocations.begin();
                it != texture->allocations.end(); ++it) {
            amap_t::iterator jit = device_allocations_.find(*it);
            if (jit == device_allocations_.end()) {
                /* There is an inconsistency between textures and device
                 * allocations. */
                assert(0 && "The impossible happened.");
                continue;
            }

            jit->second.bindings.erase(texref);
        }

        texture->allocations.clear();

        texture->bound_pointer = NULL;
    } else if (texture->array_bound) {
        if (texture->bound_array) {
            texture->bound_array->bindings.erase(texref);
        }

        texture->array_bound = false;
        texture->bound_array = NULL;
    }
}

global_context_memcheck * cuda_context_memcheck::global() {
    return static_cast<global_context_memcheck *>(cuda_context::global());
}

const global_context_memcheck * cuda_context_memcheck::global() const {
    return static_cast<const global_context_memcheck *>(
        cuda_context::global());
}

cudaError_t cuda_context_memcheck::cudaMemcpyPeer(void *dst, int dstDevice,
        const void *src, int srcDevice, size_t count) {
    return cudaMemcpyPeerImplementation(dst, dstDevice, src, srcDevice, count,
        NULL);
}

cudaError_t cuda_context_memcheck::cudaMemcpyPeerAsync(void *dst,
        int dstDevice, const void *src, int srcDevice, size_t count,
        cudaStream_t stream) {
    return cudaMemcpyPeerImplementation(dst, dstDevice, src, srcDevice, count,
        &stream);
}

namespace {
    /**
     * Locks the three given locks.  If duplicates are specified, they are
     * ignored.
     */
    template<typename Lockable>
    class trilock {
    public:
        trilock(Lockable & m1, Lockable & m2, Lockable & m3) :
                n_locks_(0) {
            typedef std::set<Lockable *> lock_set_t;
            lock_set_t m;
            m.insert(&m1);
            m.insert(&m2);
            m.insert(&m3);

            for (typename lock_set_t::iterator it = m.begin();
                    it != m.end(); ++it) {
                mx_[n_locks_] = *it;
                mx_[n_locks_]->lock();

                n_locks_++;
            }
        }

        ~trilock() {
            for (size_t i = 0; i < n_locks_; i++) {
                assert(mx_[i]);
                mx_[i]->unlock();
            }
        }
    private:
        size_t     n_locks_;
        Lockable * mx_[3];
    };
}

cudaError_t cuda_context_memcheck::cudaMemcpyPeerImplementation(void *dst,
        int dstDevice_, const void *src, int srcDevice_, size_t count,
        cudaStream_t *stream) {
    if (count == 0) {
        return cudaSuccess;
    }

    const unsigned devices = global()->devices();
    if (dstDevice_ < 0 || srcDevice_ < 0) {
        return cudaErrorInvalidDevice;
    }

    const unsigned dstDevice = static_cast<unsigned>(dstDevice_);
    const unsigned srcDevice = static_cast<unsigned>(srcDevice_);
    if (dstDevice >= devices || srcDevice >= devices) {
        return cudaErrorInvalidDevice;
    }

    void ** handle = stream ? reinterpret_cast<void **>(*stream) : NULL;

    cuda_context_memcheck * ctxd =
        static_cast<cuda_context_memcheck *>(global()->context(dstDevice));
    cuda_context_memcheck * ctxs =
        static_cast<cuda_context_memcheck *>(global()->context(srcDevice));

    trilock<boost::mutex>(mx_, ctxd->mx_, ctxs->mx_);

    stream_map_t::iterator it = streams_.find(handle);
    internal::stream_t * const cs =
        (!(stream) || it == streams_.end()) ? NULL : it->second;

    if (!(ctxd->check_access_device(dst, count, false))) {
        return cudaErrorInvalidValue;
    }

    if (!(ctxs->check_access_device(src, count, false))) {
        return cudaErrorInvalidValue;
    }

    cudaError_t ret;
    if (cs) {
        ret = callout::cudaMemcpyPeerAsync(dst, dstDevice_, src, srcDevice_,
            count, cs->stream);
    } else {
        ret = callout::cudaMemcpyPeer(dst, dstDevice_, src, srcDevice_, count);
    }

    if (ret == cudaSuccess) {
        validity_copy(dst, ctxd, src, ctxs, count, cs);
    } else {
        /* Simply mark the region as invalid */
        ctxd->validity_clear(dst, count, cs);
    }

    if (cs) {
        new internal::event_t(panoptes_created,
            cudaEventDefault, cs, NULL);
    }

    return ret;
}

bool cuda_context_memcheck::is_recently_freed(const void * ptr) const {
    return boost::icl::intersects(recent_frees_,
        static_cast<const uint8_t *>(ptr));
}

void cuda_context_memcheck::add_recent_free(const void * ptr, size_t size) {
    /**
     * Remove any overlapping allocations, then add to list.
     */
    remove_recent_free(ptr, size);

    const uint8_t * start = static_cast<const uint8_t *>(ptr);
    const uint8_t * end   = start + size;
    recent_frees_.insert(
        boost::icl::interval<free_map_t::domain_type>::right_open(start, end));
}

void cuda_context_memcheck::remove_recent_free(const void * ptr, size_t size) {
    const uint8_t * start = static_cast<const uint8_t *>(ptr);
    const uint8_t * end   = start + size;

    std::pair<free_map_t::iterator, free_map_t::iterator> range =
        recent_frees_.equal_range(free_map_t::key_type(start, end));
    recent_frees_.erase(range.first, range.second);
}
