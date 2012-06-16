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

#include "backtrace.h"
#include <boost/scoped_array.hpp>
#include <cstdio>
#include "context_internal.h"
#include "context_memcheck.h"
#include <cxxabi.h>
#include <list>
#include "logger.h"
#include "ptx_formatter.h"
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <valgrind/memcheck.h>

using namespace panoptes;
using internal::create_handle;
using internal::free_handle;

typedef boost::unique_lock<boost::mutex> scoped_lock;
typedef global_context_memcheck global_t;

/**
 * cuda_context_memcheck does not provide:
 *  cudaGetSymbolAddress
 *  cudaGetSymbolSize
 *      These are implemented in cuda_context and interact with the loaded modules
 *
 *  cudaGetTextureAlignmentOffset
 *  cudaGetTextureReference
 *
 *  cudaPointerGetAttributes
 *      We do a few calls in cuda_context to work around getting interference from
 *      driver/runtime API mixing.  While we could provide more information here,
 *      there is no way of exposing it without changing the outward behavior of the
 *      system.
 *
 *  cudaMemcpyFromSymbol
 *  cudaMemcpyFromSymbolAsync
 *  cudaMemcpyToSymbol
 *  cudaMemcpyToSymbolAsync
 *      These are decomposed into a symbol lookup and a memcpy within cuda_context.
 *
 *  cudaMemGetInfo
 *      This is passed on to the CUDA library by cuda_context
 *
 *  cudaRuntimeGetVersion
 *      This is implemented by cuda_context.  There's no need to change the outward
 *      behavior of the system.
 */

namespace {
    static const char * __master_symbol   = "__panoptes_master_symbol";
    static const char * __global_ro       = "__panoptes_global_ro";
    static const char * __global_wo       = "__panoptes_global_wo";
    static const char * __validity_prefix = "__panoptes_v";
    static const char * __texture_prefix  = "__panoptes_t";
    static const char * __errors_register = "__panoptes_lerrors";
    static const char * __has_errors      = "__panoptes_haserrors";
    static const char * __vcarry          = "__panoptes_carry";
    static const char * __reg_global_errors  = "__panoptes_gerrors";
    static const char * __reg_global_errorsw = "__panoptes_gerrorsw";
    static const char * __errors_address  = "__panoptes_eaddr";
    static const char * __reg_error_count = "__panoptes_recount";

    static const char * __shared_reg      = "__panoptes_shared_size";
    static const char * __local_reg       = "__panoptes_local_size";

    static const char * __param_shared    = "__panoptes_shared_sizep";
    static const char * __param_error     = "__panoptes_errors";
    static const char * __param_error_count = "__panoptes_error_count";

    static const size_t max_errors        = 1u << 8;

    static const size_t metadata_size     = 3u * sizeof(void *);

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

    std::string make_validity_symbol(const std::string & in) {
        assert(in.size() > 0);
        if (in[0] == '%') {
            return "%__panoptes_v" + in.substr(1);
        } else {
            return __validity_prefix + in;
        }
    }

    type_t pointer_type() {
        switch (sizeof(void *)) {
            case 4u: return b32_type;
            case 8u: return b64_type;
        }

        __builtin_unreachable();
    }

    type_t upointer_type() {
        switch (sizeof(void *)) {
            case 4u: return u32_type;
            case 8u: return u64_type;
        }

        __builtin_unreachable();
    }


    size_t sizeof_type(type_t t) {
        switch (t) {
            case b8_type:
            case s8_type:
            case u8_type:
                return 1u;
            case b16_type:
            case f16_type:
            case s16_type:
            case u16_type:
                return 2u;
            case b32_type:
            case f32_type:
            case s32_type:
            case u32_type:
                return 4u;
            case b64_type:
            case f64_type:
            case s64_type:
            case u64_type:
                return 8u;
            case pred_type:
            case texref_type:
            case invalid_type:
                assert(0 && "Unsupported type.");
                return 0;
        }

        __builtin_unreachable();
    }

    type_t bitwise_type(type_t t) {
        switch (t) {
            case b8_type:
            case s8_type:
            case u8_type:
                return b8_type;
            case b16_type:
            case f16_type:
            case s16_type:
            case u16_type:
                return b16_type;
            case b32_type:
            case f32_type:
            case s32_type:
            case u32_type:
                return b32_type;
            case b64_type:
            case f64_type:
            case s64_type:
            case u64_type:
                return b64_type;
            case pred_type:
            case texref_type:
            case invalid_type:
                assert(0 && "Unsupported type.");
                return invalid_type;
        }

        __builtin_unreachable();
        return invalid_type;
    }

    type_t signed_type(type_t t) {
        switch (t) {
            case b8_type:
            case s8_type:
            case u8_type:
                return s8_type;
            case b16_type:
            case f16_type:
            case s16_type:
            case u16_type:
                return s16_type;
            case b32_type:
            case f32_type:
            case s32_type:
            case u32_type:
                return s32_type;
            case b64_type:
            case f64_type:
            case s64_type:
            case u64_type:
                return s64_type;
            case pred_type:
            case texref_type:
            case invalid_type:
                assert(0 && "Unsupported type.");
                return invalid_type;
        }

        __builtin_unreachable();
        return invalid_type;
    }

    unsigned log_size(size_t s) {
        assert(s <= 64u);
        if (s > 32u) {
            return 4u;
        } else if (s > 16u) {
            return 3u;
        } else if (s > 8u) {
            return 2u;
        } else {
            return 1u;
        }
    }

    type_t bitwise_of(size_t s) {
        if (s > 4u) {
            return b64_type;
        } else if (s > 2u) {
            return b32_type;
        } else if (s > 1u) {
            return b16_type;
        } else {
            return b8_type;
        }
    }

    type_t signed_of(size_t s) {
        if (s > 4u) {
            return s64_type;
        } else if (s > 2u) {
            return s32_type;
        } else if (s > 1u) {
            return s16_type;
        } else {
            return s8_type;
        }
    }

    type_t unsigned_of(size_t s) {
        if (s > 32u) {
            return u64_type;
        } else if (s > 16u) {
            return u32_type;
        } else if (s > 8u) {
            return u16_type;
        } else {
            return u8_type;
        }
    }

    static statement_t make_add(type_t type, const operand_t & dst,
            const operand_t & a, const operand_t & b) {
        statement_t ret;
        ret.op = op_add;
        ret.type = type;
        ret.operands.push_back(dst);
        ret.operands.push_back(a);
        ret.operands.push_back(b);
        return ret;
    }

    static statement_t make_ld(type_t type, space_t space,
            const operand_t & dst, const operand_t & src) {
        statement_t ret;
        ret.op = op_ld;
        ret.space = space;
        ret.type = type;

        ret.operands.push_back(dst);
        ret.operands.push_back(src);

        return ret;
    }

    static statement_t make_mov(type_t type, const operand_t & dst,
            const operand_t & src) {
        statement_t ret;
        ret.op   = op_mov;
        ret.type = type;
        ret.operands.push_back(dst);
        ret.operands.push_back(src);
        return ret;
    }
}

namespace panoptes {
namespace internal {

struct check_t;

}
}

struct internal::array_t : boost::noncopyable {
    array_t(struct cudaChannelFormatDesc);
    array_t(struct cudaChannelFormatDesc, struct cudaExtent);
    ~array_t();

    struct cudaChannelFormatDesc desc;
    size_t x, y, z;
    array_allocator_t allocator;
    void * validity;
    backtrace_t allocation_bt;

    typedef boost::unordered_set<const struct textureReference *> binding_set_t;
    binding_set_t bindings;

    size_t size() const;
};

struct internal::check_t {
    check_t(cuda_context_memcheck * c, const char * ename);
    virtual ~check_t();

    virtual void check(cudaError_t);

    cuda_context_memcheck * const context;
    const char            * const entry_name;
    uint32_t              *       error_count;
    error_buffer_t        *       error_buffer;
    backtrace_t bt;
};

struct internal::instrumentation_t {
    instrumentation_t() { }

    enum error_type_t {
        no_error,
        wild_branch
    };

    struct error_desc_t {
        error_type_t type;
        statement_t orig;
    };

    /**
     * Mapping of each error code (offset by 1) to some metadata to give
     * a better description.
     */
    std::vector<error_desc_t> errors;

    /**
     * Set of unchecked predicate operations.
     */
    typedef std::set<std::string> sset_t;
    sset_t unchecked;
};

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
            /* Synchronize the device, clean up some resources, and try again. */
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

void internal::check_t::check(cudaError_t r) {
    if (r != cudaSuccess) {
        return;
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
            return;
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
            }
        }

        logger::instance().print(buffer.c_str());

        sret = snprintf(buf, sizeof(buf), "Kernel launched by:");
        assert(sret < (int) sizeof(buf) - 1);
        logger::instance().print(buf, bt);
    } /* else: no errors. */
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

    size_t       bound_size;
    backtrace_t  binding;

    typedef std::set<const void *> allocation_vt;
    allocation_vt allocations;
};

internal::texture_t::texture_t() : bound_pointer(NULL), array_bound(false),
    bound_array(NULL) { }
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
    if (stream && checker) {
        cudaError_t ret = callout::cudaEventQuery(event);
        if (ret != cudaErrorNotReady) {
            checker->check(ret);
        }
    }

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

global_context_memcheck::global_context_memcheck() { }

global_context_memcheck::~global_context_memcheck() {
    /**
     * Cleanup instrumentation metadata.
     */
    for (entry_info_map_t::iterator it = entry_info_.begin();
            it != entry_info_.end(); ++it) {
        delete it->second.inst;
    }
}

cuda_context * global_context_memcheck::factory(int device,
        unsigned int flags) const {
    return new cuda_context_memcheck(
        const_cast<global_t *>(this), device, flags);
}

bool cuda_context_memcheck::check_access_device(const void * ptr, size_t len) const {
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
            amap_t::const_iterator jit = device_allocations_.lower_bound(search);
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
            // TODO CHECK FOR RECENTLY FREED
            char msg[128];
            int ret = snprintf(msg, sizeof(msg), "Invalid device read of size %zu\n"
                " Address %p is not malloc'd or (recently) free'd.\n"
                " Possible host pointer?", len, ptr);
            // sizeof(msg) is small, so the cast is safe.
            assert(ret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);

            raise(SIGSEGV);
        } else {
            // Invalid read starts at usrc + offset
            //
            // Technically, we don't know whether the full read of (len - offset)
            // additional bytes is invalid.
            char msg[128];
            int ret = snprintf(msg, sizeof(msg), "Invalid device read of size %zu\n"
                "Address %p is 0 bytes after a block of size %zu alloc'd\n",
                len - offset, uptr + offset, asize);
            // sizeof(msg) is small, so the cast is safe.
            assert(ret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);

            raise(SIGSEGV);
        }

        return false;
    }

    return true;
}

bool cuda_context_memcheck::check_access_host(const void * ptr, size_t len) const {
    return valgrind_ ? (VALGRIND_CHECK_MEM_IS_ADDRESSABLE(ptr, len) == 0) : true;
}

/**
 * TODO:  On setup, setup SIGSEGV signal handler.
 */
cuda_context_memcheck::cuda_context_memcheck(
            global_context_memcheck * g, int device, unsigned int flags) :
        cuda_context(g, device, flags),
        master_(1 << (lg_max_memory - lg_chunk_bytes)),
        chunks_    (1 << (lg_max_memory - lg_chunk_bytes)),
        chunks_aux_(1 << (lg_max_memory - lg_chunk_bytes)), pool_(4, 16),
        error_counts_(max_errors),
        error_buffers_(max_errors),
        valgrind_((unsigned) (RUNNING_ON_VALGRIND)),
        pagesize_((size_t) sysconf(_SC_PAGESIZE)) {
    default_chunk_ = pool_.allocate();
    initialize_chunk(default_chunk_);

    // Initialize master list to point at the default chunk
    metadata_chunk ** const master = master_.host();
    for (size_t i = 0; i < (1u << (lg_max_memory - lg_chunk_bytes)); i++) {
        master[i]  = default_chunk_->gpu();
        chunks_[i] = default_chunk_;
    }

    // Copy master list to GPU
    pool_.to_gpu();
    master_.to_gpu();

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
        if (variable.has_initializer) {
            assert(!(variable.is_array && variable.array_flexible));
            validity_set(ptr, variable.size(), NULL);
        }
    }

    /**
     * Initialize the master lookup symbol.
     * TODO:  Add error checking.
     */
    bool found_somewhere = false;
    for (global_t::module_map_t::iterator it = g->modules_.begin();
            it != g->modules_.end(); ++it) {
        using internal::module_t;
        module_t * const module = it->second;
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
    assert(found_somewhere || g->modules_.size() == 0);

    // Insert the zero stream
    streams_.insert(stream_map_t::value_type(NULL,
        internal::stream_t::stream_zero()));
}

cuda_context_memcheck::~cuda_context_memcheck() {
    /**
     * Cleanup outstanding streams.
     */
    for (stream_map_t::iterator it = streams_.begin();
            it != streams_.end(); ++it) {
        if (it->second->busy() != cudaSuccess) {
            /** TODO:  A backtrace of the original allocation would be nice... */
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
        if (chunks_[i] != default_chunk_) {
            pool_.free(chunks_[i]);
        }
    }

    pool_.free(default_chunk_);
}


void cuda_context_memcheck::initialize_chunk(pool_t::handle_t * handle) const {
    /* Do work on host */
    metadata_chunk * host = handle->host();

    memset(host->a_data, 0,    sizeof(host->a_data));
    memset(host->v_data, 0xFF, sizeof(host->v_data));
}

void global_context_memcheck::instrument(
        void **fatCubinHandle, ptx_t * target) {
    assert(target);

    /* We need sm_11 for atomic instructions. */
    if (target->sm == SM10) {
        target->sm = SM11;
    }

    typedef std::vector<variable_t> variable_vt;
    variable_vt new_globals;

    const size_t tcount = target->textures.size();
    for (size_t i = 0; i < tcount; i++) {
        const size_t ncount = target->textures[i].names.size();
        for (size_t j = 0; j < ncount; j++) {
            variable_t tv;
            tv.space = const_space;
            tv.type  = pointer_type();
            tv.name = __texture_prefix + target->textures[i].names[j];
            tv.has_initializer = true;
            tv.initializer_vector = false;

            variant_t v;
            v.type   = variant_integer;
            v.data.u = 0;

            tv.initializer.push_back(v);
            new_globals.push_back(tv);
        }
    }

    {
        variable_t m;
        m.space     = const_space;
        m.type      = pointer_type();
        m.name      = __master_symbol;
        new_globals.push_back(m);
    }

    { // global_ro
        variable_t m;
        m.space     = const_space;
        m.type      = b64_type;
        m.name      = __global_ro;
        m.is_array  = true;
        m.array_dimensions = 1;
        m.array_size[0] = 2;

        variant_t zero;
        zero.type = variant_integer;
        zero.data.u = 0;
        m.has_initializer = true;
        m.initializer_vector = true;
        m.initializer.push_back(zero);
        m.initializer.push_back(zero);

        new_globals.push_back(m);
    }

    { // global_wo
        variable_t m;
        m.space     = global_space;
        m.type      = pointer_type();
        m.name      = __global_wo;
        m.is_array  = true;
        m.array_dimensions = 1;
        m.array_size[0] = 2;
        new_globals.push_back(m);
    }

    const size_t vcount = target->variables.size();
    for (size_t i = 0; i < vcount; i++) {
        variable_handle_t h(fatCubinHandle, target->variables[i].name);
        variable_data_t d;
        d.ptx            = target->variables[i];
        d.hostVar        = NULL;
        variable_definitions_.insert(variable_definition_map_t::value_type(
            h, d));
    }

    const size_t entries = target->entries.size();
    for (size_t i = 0; i < entries; i++) {
        analyze_entry(target->entries[i]);
    }

    for (size_t i = 0; i < entries; i++) {
        instrument_entry(target->entries[i]);
    }

    target->variables.insert(target->variables.end(),
        new_globals.begin(), new_globals.end());
}

static void analyze_block(size_t * fixed_shared_memory,
        size_t * local_memory, const block_t * block) {
    assert(block);

    const scope_t & scope = *block->scope;
    size_t vn;

    switch (block->block_type) {
        case block_scope:
            /* Scan for shared variables. */
            vn = scope.variables.size();
            for (size_t i = 0; i < vn; i++) {
                const variable_t & v = scope.variables[i];
                if (v.space == shared_space &&
                        !(v.array_flexible)) {
                    *fixed_shared_memory += v.size();
                } else if (v.space == local_space) {
                    *local_memory += v.size();
                }
            }

            /* Recurse. */
            for (scope_t::block_vt::const_iterator it = scope.blocks.begin();
                    it != scope.blocks.end(); ++it) {
                analyze_block(fixed_shared_memory, local_memory, *it);
            }

            break;
        case block_statement:
        case block_label:
        case block_invalid:
            /* Do nothing */
            break;

    }
}

void global_context_memcheck::analyze_entry(function_t * entry) {
    if (entry->linkage == linkage_extern) {
        /* Add to list of external entries. */
        external_entries_.insert(entry->entry_name);
        return;
    }

    if (entry->has_return_value) {
        assert(0 && "Return values are not supported.");
        return;
    }

    if (!(entry->entry)) {
        nonentries_.insert(entry->entry_name);
        return;
    }

    /* Determine size of parameter block. */
    size_t offset = 0;
    const size_t pcount = entry->params.size();
    for (size_t i = 0; i < pcount; i++) {
        const size_t align = std::max(entry->params[i].has_align ?
            entry->params[i].alignment : 1u, entry->params[i].size());
        /* Align offset so far. */
        offset = (offset + align - 1u) & ~(align - 1u);
        /* Add size of parameter. */
        offset += entry->params[i].size();
    }

    /* Note parameter block information. */
    entry_info_t e;
    e.user_params     = pcount;
    e.user_param_size = offset;

    analyze_block(&e.fixed_shared_memory, &e.local_memory, &entry->scope);

    entry_info_.insert(entry_info_map_t::value_type(
        entry->entry_name, e));
}

void global_context_memcheck::instrument_entry(function_t * entry) {
    if (entry->linkage == linkage_extern) {
        return;
    }

    if (!(entry->entry)) {
        assert(0 && "Non-entry methods not supported.");
        return;
    }

    if (entry->no_body) {
        return;
    }

    /* Instrument parameters. */
    const size_t pcount = entry->params.size();
    size_t metadata = 0;

    { /* Dynamic shared memory size */
        param_t ss;
        ss.space = param_space;
        ss.type  = upointer_type();
        ss.name  = __param_shared;

        entry->params.push_back(ss);
        metadata += ss.size();
    }

    { /* Error counter. */
        param_t ec;
        ec.space = param_space;
        ec.type  = pointer_type();
        ec.is_ptr = true;
        ec.name   = __param_error_count;

        entry->params.push_back(ec);
        metadata += ec.size();
    }

    { /* Error output buffer. */
        param_t eb;
        eb.space = param_space;
        eb.type  = pointer_type();
        eb.is_ptr = true;
        eb.name   = __param_error;

        entry->params.push_back(eb);
        metadata += eb.size();
    }

    assert(metadata == metadata_size);

    /* Validity bits for parameters. */
    for (size_t i = 0; i < pcount; i++) {
        param_t vp = entry->params[i];
        switch (vp.type) {
            case b8_type:
            case s8_type:
            case u8_type:
                vp.type = b8_type;
                break;
            case b16_type:
            case f16_type:
            case s16_type:
            case u16_type:
                vp.type = b16_type;
                break;
            case b32_type:
            case f32_type:
            case s32_type:
            case u32_type:
                vp.type = b32_type;
                break;
            case b64_type:
            case f64_type:
            case s64_type:
            case u64_type:
                vp.type = b64_type;
                break;
            case pred_type:
            case texref_type:
            case invalid_type:
                assert(0 && "Unsupported type.");
                break;
        }
        vp.name = make_validity_symbol(vp.name);
        entry->params.push_back(vp);
    }

    internal::instrumentation_t * inst = new
        internal::instrumentation_t();

    instrument_block(&entry->scope, inst);

    entry_info_map_t::iterator it = entry_info_.find(entry->entry_name);
    if (it == entry_info_.end()) {
        assert(0 && "The impossible happened.");
        delete inst;
        return;
    }

    /* Add entry-wide error count initialization after instrumentation
     * pass */
    {
        variable_t errors;
        errors.space = reg_space;
        errors.type  = u32_type;
        errors.name  = __errors_register;
        assert(entry->scope.block_type == block_scope);
        entry->scope.scope->variables.push_back(errors);

        variable_t has_errors;
        has_errors.space = reg_space;
        has_errors.type  = pred_type;
        has_errors.name  = __has_errors;
        entry->scope.scope->variables.push_back(has_errors);

        variable_t global_errors;
        global_errors.space = reg_space;
        global_errors.type  = u32_type;
        global_errors.name  = __reg_global_errors;
        entry->scope.scope->variables.push_back(global_errors);

        variable_t global_errorsw;
        global_errorsw.space = reg_space;
        global_errorsw.type  = pointer_type();
        global_errorsw.name  = __reg_global_errorsw;
        entry->scope.scope->variables.push_back(global_errorsw);

        variable_t error_count;
        error_count.space = reg_space;
        error_count.type  = pointer_type();
        error_count.name  = __reg_error_count;
        entry->scope.scope->variables.push_back(error_count);

        variable_t error_address;
        error_address.space = reg_space;
        error_address.type  = pointer_type();
        error_address.name  = __errors_address;
        entry->scope.scope->variables.push_back(error_address);

        /* Code to initialize errors. */
        {
        block_t * b = new block_t();
        b->block_type   = block_statement;
        b->parent       = &entry->scope;
        b->statement    = new statement_t();
        b->statement->op    = op_mov;
        b->statement->type  = u32_type;

        operand_t dst = operand_t::make_identifier(__errors_register);
        b->statement->operands.push_back(dst);

        operand_t src = operand_t::make_iconstant(0);
        b->statement->operands.push_back(src);

        entry->scope.scope->blocks.push_front(b);
        }

        /* Initialize error_count. */
        {
        block_t * b = new block_t();
        b->block_type   = block_statement;
        b->parent       = &entry->scope;
        b->statement    = new statement_t();
        b->statement->op    = op_ld;
        b->statement->space = param_space;
        b->statement->type  = pointer_type();

        operand_t dst = operand_t::make_identifier(__reg_error_count);
        b->statement->operands.push_back(dst);

        operand_t src = operand_t::make_identifier(__param_error_count);
        b->statement->operands.push_back(src);

        entry->scope.scope->blocks.push_front(b);
        }

        /* Initialize error_address. */
        {
        block_t * b = new block_t();
        b->block_type   = block_statement;
        b->parent       = &entry->scope;
        b->statement    = new statement_t();
        b->statement->op    = op_ld;
        b->statement->space = param_space;
        b->statement->type  = pointer_type();

        operand_t dst = operand_t::make_identifier(__errors_address);
        b->statement->operands.push_back(dst);

        operand_t src = operand_t::make_identifier(__param_error);
        b->statement->operands.push_back(src);

        entry->scope.scope->blocks.push_front(b);
        }

        /* Load shared memory size. */
        {
            scope_t & scope = *entry->scope.scope;
            const type_t uptr = upointer_type();

            variable_t ss;
            ss.space = reg_space;
            ss.type  = uptr;
            ss.name  = __shared_reg;
            scope.variables.push_back(ss);

            const operand_t shared_reg =
                operand_t::make_identifier(__shared_reg);
            const operand_t shared_param =
                operand_t::make_identifier(__param_shared);

            /* We have to do this out of order because we are pushing things
             * onto the *front* of the instruction list. */
            block_t * b;
            if (it->second.fixed_shared_memory > 0) {
                b = new block_t();
                b->block_type   = block_statement;
                b->parent       = &entry->scope;
                b->statement    = new statement_t(
                    make_add(uptr, shared_reg, shared_reg,
                    operand_t::make_iconstant(
                        (int) it->second.fixed_shared_memory)));
                scope.blocks.push_front(b);
            }

            b = new block_t();
            b->block_type   = block_statement;
            b->parent       = &entry->scope;
            b->statement    = new statement_t(
                make_ld(uptr, param_space, shared_reg, shared_param));
            scope.blocks.push_front(b);
        }

        if (it->second.local_memory > 0) {
            /* Load local memory size. */
            scope_t & scope = *entry->scope.scope;
            const type_t uptr = upointer_type();

            variable_t ss;
            ss.space = reg_space;
            ss.type  = uptr;
            ss.name  = __local_reg;
            scope.variables.push_back(ss);

            const operand_t local_reg =
                operand_t::make_identifier(__local_reg);

            block_t * b = new block_t();
            b->block_type   = block_statement;
            b->parent       = &entry->scope;
            b->statement    = new statement_t(
                make_mov(uptr, local_reg, operand_t::make_iconstant(
                    (int) it->second.local_memory)));
            scope.blocks.push_front(b);
        }

        /* Initialize carry flag. */
        {
            scope_t & scope = *entry->scope.scope;

            variable_t vc;
            vc.space = reg_space;
            vc.type  = u32_type;
            vc.name  = __vcarry;

            scope.variables.push_back(vc);

            const operand_t vc_reg =
                operand_t::make_identifier(__vcarry);

            block_t * b = new block_t();
            b->block_type   = block_statement;
            b->parent       = &entry->scope;
            b->statement    = new statement_t(
                make_mov(u32_type, vc_reg, operand_t::make_iconstant(0)));
            scope.blocks.push_front(b);
        }
    }

    /* Copy instrumented function. */
    it->second.function = entry;
    it->second.inst = inst;
}

static statement_t make_and(type_t type, const operand_t & dst,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op = op_and;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_bfe(type_t type, const operand_t & d,
        const operand_t & a, const operand_t & b, const operand_t & c) {
    statement_t ret;
    ret.op   = op_bfe;
    ret.type = type;
    ret.operands.push_back(d);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    ret.operands.push_back(c);
    return ret;
}

static statement_t make_bfi(type_t type, const operand_t & f,
        const operand_t & a, const operand_t & b, const operand_t & c,
        const operand_t & d) {
    statement_t ret;
    ret.op   = op_bfi;
    ret.type = type;
    ret.operands.push_back(f);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    ret.operands.push_back(c);
    ret.operands.push_back(d);
    return ret;
}

static statement_t make_brev(type_t type, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op   = op_brev;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_cvt(type_t type, type_t type2, const operand_t & dst,
        const operand_t & src, bool saturating) {
    statement_t ret;
    ret.op = op_cvt;
    ret.type = type;
    ret.type2 = type2;
    ret.saturating = saturating;
    if (saturating) {
        assert(sizeof_type(type) > sizeof_type(type2));
    }
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_cnot(type_t type, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op   = op_cnot;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_min(type_t type, const operand_t & dst,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op   = op_min;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_mov(type_t type, const operand_t & dst, int64_t src) {
    statement_t ret;
    ret.op   = op_mov;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(operand_t::make_iconstant(src));
    return ret;
}

static statement_t make_neg(type_t type, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op   = op_neg;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_not(type_t type, const operand_t & dst,
        const operand_t & src) {
    statement_t ret;
    ret.op   = op_not;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    return ret;
}

static statement_t make_or(type_t type, const operand_t & d,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op = op_or;
    ret.type = type;
    ret.operands.push_back(d);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_setp(type_t type, op_set_cmp_t cmp,
        const std::string & pred, const operand_t & lhs,
        const operand_t & rhs) {
    statement_t ret;
    ret.op   = op_setp;
    ret.type = type;
    ret.cmp  = cmp;
    ret.has_ppredicate = true;
    ret.ppredicate = pred;
    ret.operands.push_back(lhs);
    ret.operands.push_back(rhs);
    return ret;
}

static statement_t make_setp(type_t type, op_set_cmp_t cmp,
        const std::string & pred, const std::string & qpred,
        const operand_t & lhs, const operand_t & rhs) {
    statement_t ret;
    ret.op   = op_setp;
    ret.type = type;
    ret.cmp  = cmp;
    ret.has_ppredicate = true;
    ret.ppredicate = pred;
    ret.has_qpredicate = true;
    ret.qpredicate = qpred;
    ret.operands.push_back(lhs);
    ret.operands.push_back(rhs);
    return ret;
}

static statement_t make_slct(type_t type, type_t type2,
        const operand_t & d, const operand_t & a,
        const operand_t & b, const operand_t & c) {
    statement_t ret;
    ret.op    = op_slct;
    ret.type  = type;
    ret.type2 = type2;
    ret.operands.push_back(d);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    ret.operands.push_back(c);
    return ret;
}

static statement_t make_shl(type_t type, const operand_t & dst,
        const operand_t & src, const operand_t & shift) {
    statement_t ret;
    ret.op = op_shl;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    ret.operands.push_back(shift);
    return ret;
}

static statement_t make_shr(type_t type, const operand_t & dst,
        const operand_t & src, const operand_t & shift) {
    statement_t ret;
    ret.op = op_shr;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(src);
    ret.operands.push_back(shift);
    return ret;
}

static statement_t make_sub(type_t type, const operand_t & dst,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op = op_sub;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static statement_t make_selp(type_t type, const std::string & pred,
        const operand_t & dst, const operand_t & lhs, const operand_t & rhs) {
    statement_t ret;
    ret.op = op_selp;
    ret.type = type;
    ret.operands.push_back(dst);
    ret.operands.push_back(lhs);
    ret.operands.push_back(rhs);
    ret.operands.push_back(operand_t::make_identifier(pred));
    return ret;
}

static statement_t make_xor(type_t type, const operand_t & d,
        const operand_t & a, const operand_t & b) {
    statement_t ret;
    ret.op = op_xor;
    ret.type = type;
    ret.operands.push_back(d);
    ret.operands.push_back(a);
    ret.operands.push_back(b);
    return ret;
}

static operand_t make_validity_operand(const operand_t & in) {
    operand_t ret;

    size_t ni, nf;
    switch (in.op_type) {
        case operand_indexed:
        case operand_identifier:
        case operand_addressable:
            /**
             * Addressable operands need not have fields.
             */
            ni = in.identifier.size();
            nf = in.field.size();
            assert(ni >= nf);

            ret.op_type = operand_identifier;
            for (size_t i = 0; i < ni; i++) {
                field_t f = i < nf ? in.field[i] : field_none;

                const std::string & id = in.identifier[i];
                bool constant = false;

                const char pm[]  = "pm";
                const char env[] = "envreg";

                if (id == "%tid" || id == "%ntid" || id == "%laneid" ||
                        id == "%warpid" || id == "%nwarpid" || id == "%ctaid" ||
                        id == "%nctaid" || id == "%smid" || id == "%nsmid" ||
                        id == "%gridid" || id == "%lanemask_eq" ||
                        id == "%lanemask_le" || id == "%lanemask_lt" ||
                        id == "%lanemask_ge" || id == "%lanemask_gt" ||
                        id == "%clock" || id == "%clock64" || id == "WARP_SZ") {
                    constant = true;
                } else {
                    if (memcmp(id.c_str(), pm, sizeof(pm) - 1u) == 0) {
                        int s;
                        int p = sscanf(id.c_str() + sizeof(pm) - 1u, "%d", &s);
                        if (p == 1 && s >= 0 && s <= 3) {
                            constant = true;
                        }
                    }

                    if (memcmp(id.c_str(), env, sizeof(env) - 1u) == 0) {
                        int s;
                        int p = sscanf(id.c_str() + sizeof(env) - 1u,
                            "%d", &s);
                        if (p == 1 && s >= 0 && s <= 31) {
                            constant = true;
                        }
                    }
                }

                if (constant) {
                    ret.identifier.push_back("0");
                    ret.field.push_back(field_none);
                } else {
                    ret.identifier.push_back(make_validity_symbol(id));
                    ret.field.push_back(f);
                }
            }

            return ret;
        case operand_constant:
        case operand_float:
        case operand_double:
            /* Constant values are always defined. */
            return operand_t::make_iconstant(0);
        case invalid_operand:
            assert(0 && "Invalid operand type.");
            return in;
    }

    __builtin_unreachable();
    return ret;
}

static std::string make_temp_identifier(type_t type, unsigned id) {
    const char * type_name = NULL;

    switch (type) {
        case b8_type:   type_name = "b8"; break;
        case s8_type:   type_name = "s8"; break;
        case u8_type:   type_name = "u8"; break;
        case b16_type:  type_name = "b16"; break;
        case f16_type:  type_name = "f16"; break;
        case s16_type:  type_name = "s16"; break;
        case u16_type:  type_name = "u16"; break;
        case b32_type:  type_name = "b32"; break;
        case f32_type:  type_name = "f32"; break;
        case s32_type:  type_name = "s32"; break;
        case u32_type:  type_name = "u32"; break;
        case b64_type:  type_name = "b64"; break;
        case f64_type:  type_name = "f64"; break;
        case s64_type:  type_name = "s64"; break;
        case u64_type:  type_name = "u64"; break;
        case pred_type: type_name = "pred"; break;
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            break;
    }

    assert(type_name);

    char buf[24];
    int ret = snprintf(buf, sizeof(buf), "__panoptes_%s_%u",
        type_name, id);
    assert(ret < (int) sizeof(buf));

    return buf;
}

static operand_t make_temp_operand(type_t type, unsigned id) {
    return operand_t::make_identifier(make_temp_identifier(type, id));
}

void global_context_memcheck::instrument_block(block_t * block,
        internal::instrumentation_t * inst) {
    if (block->block_type == block_invalid) {
        assert(0 && "Invalid block type.");
        return;
    } else if (block->block_type == block_label) {
        /* No operations necessary. */
        return;
    } else if (block->block_type == block_statement) {
        /* We do not instrument lone statements. */
        assert(0 && "Lone statements cannot be instrumented.");
    }

    assert(block->block_type == block_scope);
    scope_t * scope = block->scope;
    const size_t vcount = scope->variables.size();
    for (size_t i = 0; i < vcount; i++) {
        /* Note which predicates haven't been checked. */
        const variable_t & v = scope->variables[i];
        if (v.type == pred_type) {
            if (v.has_suffix) {
                const char * name = v.name.c_str();
                const size_t len = v.name.size();
                char buf[256];

                assert(len < sizeof(buf) - 1u);
                const size_t mlen = std::min(len, sizeof(buf) - 1u);
                const size_t rem = sizeof(buf) - mlen;
                memcpy(buf, name, mlen);

                for (int j = 0; j < v.suffix; j++) {
                    int ret = snprintf(buf + mlen, rem, "%d", j);
                    assert(ret < (int) rem);
                    inst->unchecked.insert(buf);
                }
            } else {
                inst->unchecked.insert(v.name);
            }
        }

        variable_t vv = scope->variables[i];
        vv.name = make_validity_symbol(vv.name);
        if (vv.type == f32_type) {
            vv.type = b32_type;
        } else if (vv.type == f64_type) {
            vv.type = b64_type;
        } else if (vv.type == pred_type) {
            vv.type = b16_type;
        }
        scope->variables.push_back(vv);
    }

    // Counts of number of temporary variables needed for Panoptes' purposes
    // tmpb[0] -> b8, tmpb[1] -> b16, tmpb[2] -> b32, tmpb[3] -> b64
    // tmps[0] -> s8, tmps[1] -> s16, tmps[2] -> s32, tmps[3] -> s64
    // tmpu[0] -> u8, tmpu[1] -> u16, tmpu[2] -> u32, tmpu[3] -> u64
    int tmpb[4] = {0, 0, 0, 0};
    int tmps[4] = {0, 0, 0, 0};
    int tmpu[4] = {0, 0, 0, 0};
    int tmp_pred = 0;
    int tmp_ptr = 0;

    const operand_t local_errors =
        operand_t::make_identifier(__errors_register);

    for (scope_t::block_vt::iterator it = scope->blocks.begin();
            it != scope->blocks.end(); ) {
        if ((*it)->block_type != block_statement) {
            instrument_block(*it, inst);
            ++it;
            continue;
        }

        const statement_t & statement = *(*it)->statement;
        std::vector<statement_t> aux;
        bool keep = true;

        /* Check for predicate. */
        if (statement.has_predicate) {
            typedef internal::instrumentation_t inst_t;
            inst_t::sset_t::iterator jit =
                inst->unchecked.find(statement.predicate);
            if (jit != inst->unchecked.end()) {
                /* We haven't checked this predicate lately, insert
                 * a check. */
                const operand_t vp = operand_t::make_identifier(
                    make_validity_symbol(statement.predicate));
                const std::string tmpp = "__panoptes_pred0";
                tmp_pred = std::max(tmp_pred, 1);

                inst_t::error_desc_t desc;
                desc.type = inst_t::wild_branch;
                desc.orig = statement;
                inst->errors.push_back(desc);
                const size_t error_number = inst->errors.size();

                aux.push_back(make_setp(b16_type, cmp_ne, tmpp, vp,
                    operand_t::make_iconstant(0)));
                aux.push_back(make_selp(u32_type, tmpp, local_errors,
                    operand_t::make_iconstant((int64_t) error_number),
                    local_errors));

                inst->unchecked.erase(jit);
            }
        }

        switch (statement.op) {
            case op_abs: {
                int abs_tmps[4] = {0, 0, 0, 0};

                const size_t width  = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);
                const type_t btype  = bitwise_type(statement.type);

                const operand_t & vd = make_validity_operand(
                    statement.operands[0]);
                const operand_t & va = make_validity_operand(
                    statement.operands[1]);

                switch (statement.type) {
                    case s16_type:
                    case s32_type:
                    case s64_type: {
                        /**
                         * Per http://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs,
                         * we can compute abs(x) by:
                         *
                         * const int mask = x >> (sizeof(x) * CHAR_BIT - 1)
                         * abs(x) = (x + mask) ^ mask
                         *
                         * Since we are shifting in the validity bits of the
                         * sign, we have two cases for mask (assuming int32_t):
                         * 0xFFFFFFFF and 0x00000000.  xor with an all
                         * invalid bits value produces a completely invalid
                         * result, so we can simplify our validity tracking to
                         *
                         * vabs(x) = vx | (vx >> (sizeof(vx) * CHAR_BIT - 1))
                         */
                        abs_tmps[lwidth - 1u] = 1;

                        const type_t stype  = statement.type;
                        const operand_t tmp =
                            make_temp_operand(stype, 0);
                        aux.push_back(make_shr(stype, tmp, va,
                            operand_t::make_iconstant(
                            (int) width * CHAR_BIT - 1)));
                        aux.push_back(make_or(bitwise_type(stype),
                            vd, tmp, va));

                        break; }
                    case f32_type:
                    case f64_type:
                        /**
                         * TODO We are ignoring ftz for now.
                         *
                         * The sign bit will always be cleared.
                         */
                        aux.push_back(make_and(btype, vd, va,
                            operand_t::make_iconstant(
                            (1 << (width * CHAR_BIT - 1)) - 1)));
                        break;
                    case s8_type:
                    case b8_type:
                    case u8_type:
                    case b16_type:
                    case f16_type:
                    case u16_type:
                    case b32_type:
                    case u32_type:
                    case b64_type:
                    case u64_type:
                    case pred_type:
                    case texref_type:
                    case invalid_type:
                        assert(0 && "Unsupported type.");
                        break;
                }

                for (unsigned i = 0; i < 4u; i++) {
                    tmps[i] = std::max(tmps[i], abs_tmps[i]);
                }

                break; }
            case op_add:
            case op_addc:
            case op_mul:
            case op_mul24:
            case op_sub:
            case op_subc: {
                assert(statement.operands.size() == 3u);
                const operand_t & a = statement.operands[1];
                const operand_t & b = statement.operands[2];
                const operand_t & d = statement.operands[0];

                const operand_t & vd = make_validity_operand(d);

                int add_tmpu[4] = {0, 0, 0, 0};
                int add_tmps[4] = {0, 0, 0, 0};
                int add_pred = 0;
                int add_ptr  = 0;

                /**
                 * mul24 instructions with types other than u32 and s32 are not
                 * supported by PTX.
                 */
                assert(statement.op != op_mul24 ||
                    (statement.type == s32_type ||
                     statement.type == u32_type));

                const operand_t va = make_validity_operand(a);
                const operand_t vb = make_validity_operand(b);

                const bool carry_in = (statement.op == op_addc) ||
                    (statement.op == op_subc);

                switch (statement.type) {
                    case f32_type:
                    case f64_type: {
                        assert(!(carry_in));

                        /**
                         * If any bits are invalid, assume all are invalid.
                         */
                        const size_t width  = sizeof_type(statement.type);
                        const size_t lwidth = log_size(width * CHAR_BIT);
                        const type_t stype  = signed_of(width);
                        const type_t btype  = bitwise_of(width);

                        const operand_t one = operand_t::make_iconstant(1);

                               if (va.is_constant() && vb.is_constant()) {
                            if (statement.op == op_mul &&
                                    statement.width == width_wide) {
                                /* Destination is wider than source. */
                                aux.push_back(make_mov(bitwise_of(2u * width),
                                    vd, va));
                            } else {
                                aux.push_back(make_mov(btype, vd, va));
                            }
                        } else if (va.is_constant() && !(vb.is_constant())) {
                            add_tmps[lwidth - 1u] = 1;

                            const operand_t tmp =
                                make_temp_operand(stype, 0);
                            aux.push_back(make_cnot(btype, tmp, vb));
                            aux.push_back(make_sub(stype, vd, tmp, one));
                        } else if (!(va.is_constant()) && vb.is_constant()) {
                            add_tmps[lwidth - 1u] = 1;

                            const operand_t tmp =
                                make_temp_operand(stype, 0);
                            aux.push_back(make_cnot(btype, tmp, va));
                            aux.push_back(make_sub(stype, vd, tmp, one));
                        } else {
                            assert(lwidth >= 1u);
                            assert(lwidth <= 4u);
                            add_tmps[lwidth - 1u] = 1;

                            const operand_t tmp =
                                make_temp_operand(stype, 0);

                            aux.push_back(make_or(btype, tmp, va, vb));
                            aux.push_back(make_cnot(btype, tmp, tmp));
                            aux.push_back(make_sub(stype, vd, tmp, one));
                        }

                        break; }
                    case s16_type:
                    case s32_type:
                    case s64_type:
                    case u16_type:
                    case u32_type:
                    case u64_type: {
                        /**
                         * Per "Using Memcheck" (2005), undefinedness is
                         * computed by OR'ing and then propagating to the left.
                         */
                        const size_t width  = sizeof_type(statement.type);
                        const size_t lwidth = log_size(width * CHAR_BIT);
                        const type_t stype  = signed_of(width);
                        const type_t btype  = bitwise_of(width);

                        assert(!(carry_in) || width == 4u);
                        const operand_t vc =
                            operand_t::make_identifier(__vcarry);

                        std::vector<operand_t> vin;
                        if (carry_in) {
                            vin.push_back(vc);
                        }

                        if (!(va.is_constant())) {
                            vin.push_back(va);
                        }

                        if (!(vb.is_constant())) {
                            vin.push_back(vb);
                        }

                        bool constant = false;

                        const operand_t tmp =
                            make_temp_operand(stype, 0);
                        const operand_t u =
                            make_temp_operand(stype, 1);

                        const size_t vsize = vin.size();
                        if (vsize == 0) {
                            aux.push_back(make_mov(btype, vd,
                                operand_t::make_iconstant(0)));
                            constant = true;
                        } else if (vsize == 1 && carry_in) {
                            /* Copy validity bits of carry flag. */
                            aux.push_back(make_mov(btype, vd, vc));
                            break;
                        } else {
                            assert(lwidth >= 1u);
                            assert(lwidth <= 4u);

                            operand_t blended;

                            if (vsize == 1u) {
                                add_tmps[lwidth - 1u] = 1;
                                blended = vin[0];
                            } else {
                                assert(vsize >= 2u);
                                add_tmps[lwidth - 1u] = 2;

                                /* Blend inputs together. */
                                blended = u;
                                aux.push_back(
                                    make_or(btype, u, vin[0], vin[1]));
                                for (size_t i = 2; i < vsize; i++) {
                                    aux.push_back(
                                        make_or(btype, u, u, vin[i]));
                                }
                            }

                            aux.push_back(make_neg(stype, tmp, blended));

                            if (statement.op == op_mul &&
                                    statement.width == width_wide) {
                                /**
                                 * Sign extend result.
                                 */
                                const type_t swtype  = signed_of(width * 2u);
                                add_tmps[lwidth] = 1;

                                aux.push_back(make_or(btype, tmp, tmp, blended));
                                aux.push_back(make_cvt(swtype, stype, vd, tmp,
                                    false));
                            } else {
                                aux.push_back(make_or(btype, vd, tmp, blended));
                            }
                        }

                        /* If instructed to carry-out, store the validity bits
                         * of the carry flag by propagating the high bit. */
                        if (statement.carry_out) {
                            assert(statement.type == u32_type ||
                                   statement.type == s32_type);
                            assert(statement.op   == op_add  ||
                                   statement.op   == op_addc ||
                                   statement.op   == op_sub  ||
                                   statement.op   == op_subc);

                            if (constant) {
                                /* Carry-out is wholly valid. */
                                aux.push_back(make_mov(btype, vc,
                                    operand_t::make_iconstant(0)));
                            } else {
                                aux.push_back(make_shr(stype, vc, vd,
                                    operand_t::make_iconstant(31)));
                            }
                        }

                        break; }
                    case s8_type:
                    case b8_type:
                    case u8_type:
                    case b16_type:
                    case f16_type:
                    case b32_type:
                    case b64_type:
                    case pred_type:
                    case texref_type:
                    case invalid_type:
                        assert(0 && "Unsupported type.");
                        break;
                }

                for (unsigned i = 0; i < 4u; i++) {
                    tmpu[i] = std::max(tmpu[i], add_tmpu[i]);
                    tmps[i] = std::max(tmps[i], add_tmps[i]);
                }
                tmp_pred = std::max(tmp_pred, add_pred);
                tmp_ptr  = std::max(tmp_ptr,  add_ptr);

                break; }
            case op_and: {
                assert(statement.operands.size() == 3u);
                const operand_t & a = statement.operands[1];
                const operand_t & b = statement.operands[2];
                const operand_t & d = statement.operands[0];

                const operand_t va = make_validity_operand(a);
                const operand_t vb = make_validity_operand(b);
                const operand_t vd = make_validity_operand(d);

                int and_tmpb[4] = {0, 0, 0, 0};
                int and_pred = 0;
                int and_ptr  = 0;

                switch (statement.type) {
                    case pred_type:
                        /**
                         * For simplicity, we assume the worst case and OR the
                         * validity bits without regard for the underlying data.
                         */
                        if (va.is_constant() && vb.is_constant()) {
                            aux.push_back(make_mov(b16_type, vd, va));
                        } else if (va.is_constant() && !(vb.is_constant())) {
                            aux.push_back(make_mov(b16_type, vd, vb));
                        } else if (!(va.is_constant()) && vb.is_constant()) {
                            aux.push_back(make_mov(b16_type, vd, va));
                        } else {
                            aux.push_back(make_or(b16_type, vd, va, vb));
                        }

                        assert(d.identifier.size() > 0);
                        inst->unchecked.insert(d.identifier[0]);
                        break;
                    case b16_type:
                    case b32_type:
                    case b64_type: {
                        const size_t width  = sizeof_type(statement.type);
                        const size_t lwidth = log_size(width * CHAR_BIT);
                        const type_t btype  = bitwise_of(width);

                               if (va.is_constant() && vb.is_constant()) {
                            aux.push_back(make_mov(btype, vd, va));
                        } else if (va.is_constant() && !(vb.is_constant())) {
                            and_tmpb[lwidth - 1u] = 2;

                            const operand_t tmp =
                                make_temp_operand(btype, 0);
                            const operand_t tmp2 =
                                make_temp_operand(btype, 1);

                            /**
                             * vd = AND(vb, NOR(b, vb))
                             */
                            aux.push_back(make_or(btype, tmp, b, vb));
                            aux.push_back(make_not(btype, tmp2, tmp));
                            aux.push_back(make_and(btype, vd, tmp, vb));
                        } else if (!(va.is_constant()) && vb.is_constant()) {
                            and_tmpb[lwidth - 1u] = 2;

                            const operand_t tmp =
                                make_temp_operand(btype, 0);
                            const operand_t tmp2 =
                                make_temp_operand(btype, 1);

                            /**
                             * vd = AND(va, NOR(a, va))
                             */
                            aux.push_back(make_or(btype, tmp, a, va));
                            aux.push_back(make_not(btype, tmp2, tmp));
                            aux.push_back(make_and(btype, vd, tmp, va));
                        } else {
                            assert(lwidth >= 1u);
                            assert(lwidth <= 4u);
                            and_tmpb[lwidth - 1u] = 3;

                            const operand_t tmp0 =
                                make_temp_operand(btype, 0);
                            const operand_t tmp1 =
                                make_temp_operand(btype, 1);
                            const operand_t tmp2 =
                                make_temp_operand(btype, 2);

                            aux.push_back(make_or(btype, tmp0, a, va));
                            aux.push_back(make_not(btype, tmp1, tmp0));
                            aux.push_back(make_or(btype, tmp0, b, vb));
                            aux.push_back(make_not(btype, tmp2, tmp0));
                            aux.push_back(make_and(btype, tmp0, tmp1, tmp2));
                            aux.push_back(make_or(btype, tmp1, va, vb));
                            aux.push_back(make_and(btype, vd, tmp0, tmp1));
                        }

                        break; }
                    case f32_type:
                    case f64_type:
                    case s16_type:
                    case s32_type:
                    case s64_type:
                    case u16_type:
                    case u32_type:
                    case u64_type:
                    case s8_type:
                    case b8_type:
                    case u8_type:
                    case f16_type:
                    case texref_type:
                    case invalid_type:
                        assert(0 && "Unsupported type.");
                        break;
                }

                for (unsigned i = 0; i < 4u; i++) {
                    tmpb[i] = std::max(tmpb[i], and_tmpb[i]);
                }
                tmp_pred = std::max(tmp_pred, and_pred);
                tmp_ptr  = std::max(tmp_ptr,  and_ptr);

                break; }
            case op_bar:
                break;
                switch (statement.barrier) {
                    case barrier_sync:
                    case barrier_arrive:
                        /* No-op. */
                        break;
                    case barrier_reduce:
                        assert(0 && "TODO bar.red not supported.");
                        break;
                    case barrier_invalid:
                        assert(0 && "Invalid barrier type.");
                        break;
                }

                break;
            case op_bfe: {
                assert(statement.operands.size() == 4u);

                assert(statement.type == u32_type ||
                       statement.type == u64_type ||
                       statement.type == s32_type ||
                       statement.type == s64_type);

                /**
                 * If either b or c have any invalid bits above their mask
                 * (0xFF), we assume the worst and consider the entire
                 * result invalid.
                 */
                bool immed_constant;
                operand_t immed;

                std::vector<operand_t> source_validity;
                for (unsigned i = 2; i < 4; i++) {
                    const operand_t & op   = statement.operands[i];
                    const operand_t vop = make_validity_operand(op);
                    if (!(vop.is_constant())) {
                        source_validity.push_back(vop);
                    }
                }

                const size_t vargs = source_validity.size();
                const type_t btype = bitwise_type(statement.type);
                const type_t stype = signed_type(statement.type);

                const size_t width  = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);

                const operand_t tmp0 = make_temp_operand(b32_type, 0);

                assert(vargs <= 2);
                switch (vargs) {
                    case 0:
                        /**
                         * No variable arguments, result is defined.
                         */
                        immed = operand_t::make_iconstant(0);
                        immed_constant = true;
                        break;
                    case 1:
                        /**
                         * Propagate from source.
                         */
                        aux.push_back(make_and(b32_type, tmp0,
                            source_validity[0],
                            operand_t::make_iconstant(0xFF)));
                        aux.push_back(make_cnot(b32_type, tmp0, tmp0));
                        aux.push_back(make_sub(s32_type, tmp0, tmp0,
                            operand_t::make_iconstant(1)));

                        immed = tmp0;
                        immed_constant = false;
                        break;
                    case 2:
                        /**
                         * OR validity bits of two results, then propagate.
                         */
                        aux.push_back(make_or(b32_type, tmp0,
                            source_validity[0], source_validity[1]));
                        aux.push_back(make_and(b32_type, tmp0, tmp0,
                            operand_t::make_iconstant(0xFF)));
                        aux.push_back(make_cnot(b32_type, tmp0, tmp0));
                        aux.push_back(make_sub(s32_type, tmp0, tmp0,
                            operand_t::make_iconstant(1)));
                        immed = tmp0;
                        immed_constant = false;
                        break;
                }

                /**
                 * Consider the argument to a.
                 */
                const operand_t & a  = statement.operands[1];
                const operand_t   va = make_validity_operand(a);
                const bool aconstant = va.is_constant();

                const operand_t   vd = make_validity_operand(
                    statement.operands[0]);

                assert(lwidth >= 1u);
                assert(lwidth <= 4u);

                if (aconstant && immed_constant) {
                    /**
                     * Result is constant.  We already have initialized immed
                     * to zero.
                     */
                    aux.push_back(make_mov(btype, vd, immed));
                    break;
                }

                if (aconstant) {
                    /**
                     * We've computed all that we need to validity-bitwise
                     * in computing immed.
                     *
                     * Mark that we are using the tmp variable.
                     */
                    tmpb[3u] = std::max(tmpb[3u], 1);
                    if (width == 8u) {
                        aux.push_back(make_mov(btype, vd, immed));
                    } else if (width == 4u) {
                        /**
                         * Sign extend.
                         */
                        aux.push_back(make_cvt(stype, s32_type, vd, immed,
                            false));
                    } else {
                        assert(0 && "Unsupported width.");
                    }

                    break;
                }

                if (immed_constant) {
                    /**
                     * Extract the validity bits using b and c.
                     *
                     * TODO:  This does not properly take into account the
                     * propagation of the sign bit.
                     */
                    aux.push_back(make_bfe(statement.type, vd, va,
                        statement.operands[2], statement.operands[3]));
                } else {
                    /**
                     * Extract, then OR with worst case validity results
                     * due to b and c.
                     */
                    if (width == 4u) {
                        tmpb[2u] = std::max(tmpb[2u], 2);
                    } else if (width == 8u) {
                        tmpb[2u] = std::max(tmpb[2u], 1);
                        tmpb[3u] = std::max(tmpb[3u], 2);
                    } else {
                        assert(0 && "Unsupported width.");
                    }

                    const operand_t tmp1 = make_temp_operand(btype, 1);
                    const operand_t tmp2 = make_temp_operand(btype, 0);

                    assert(immed != tmp1);

                    /**
                     * Sign extend.
                     */
                    if (width == 8u) {
                        aux.push_back(make_cvt(s64_type, s32_type, tmp2,
                            immed, false));
                        assert(immed != tmp2);
                        immed = tmp2;
                    }

                    aux.push_back(make_bfe(statement.type, tmp1, va,
                        statement.operands[2], statement.operands[3]));
                    aux.push_back(make_or(btype, vd, immed, tmp1));
                }

                break; }
            case op_bfi: {
                assert(statement.operands.size() == 5u);

                assert(statement.type == b32_type ||
                       statement.type == b64_type);

                /**
                 * If either c or d have any invalid bits above their mask
                 * (0xFF), we assume the worst and consider the entire
                 * result invalid.
                 */
                bool immed_constant;
                operand_t immed;

                std::vector<operand_t> source_validity;
                for (unsigned i = 3; i < 5; i++) {
                    const operand_t & op   = statement.operands[i];
                    const operand_t vop = make_validity_operand(op);
                    if (!(vop.is_constant())) {
                        source_validity.push_back(vop);
                    }
                }

                const size_t vargs = source_validity.size();
                const type_t btype = bitwise_type(statement.type);

                const size_t width  = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);

                const operand_t tmp0 = make_temp_operand(b32_type, 0);

                assert(vargs <= 2);
                switch (vargs) {
                    case 0:
                        /**
                         * No variable arguments, result is defined.
                         */
                        immed = operand_t::make_iconstant(0);
                        immed_constant = true;
                        break;
                    case 1:
                        /**
                         * Propagate from source.
                         */
                        aux.push_back(make_and(b32_type, tmp0,
                            source_validity[0],
                            operand_t::make_iconstant(0xFF)));
                        aux.push_back(make_cnot(b32_type, tmp0, tmp0));
                        aux.push_back(make_sub(s32_type, tmp0, tmp0,
                            operand_t::make_iconstant(1)));

                        immed = tmp0;
                        immed_constant = false;
                        break;
                    case 2:
                        /**
                         * OR validity bits of two results, then propagate.
                         */
                        aux.push_back(make_or(b32_type, tmp0,
                            source_validity[0], source_validity[1]));
                        aux.push_back(make_and(b32_type, tmp0, tmp0,
                            operand_t::make_iconstant(0xFF)));
                        aux.push_back(make_cnot(b32_type, tmp0, tmp0));
                        aux.push_back(make_sub(s32_type, tmp0, tmp0,
                            operand_t::make_iconstant(1)));
                        immed = tmp0;
                        immed_constant = false;
                        break;
                }

                /**
                 * Consider the argument to a and b.
                 */
                const operand_t & a  = statement.operands[1];
                const operand_t   va = make_validity_operand(a);
                const bool aconstant = va.is_constant();

                const operand_t & b  = statement.operands[2];
                const operand_t   vb = make_validity_operand(b);
                const bool bconstant = vb.is_constant();

                const operand_t & vd = make_validity_operand(
                    statement.operands[0]);

                assert(lwidth >= 1u);
                assert(lwidth <= 4u);

                const bool data_constant = aconstant && bconstant;

                if (data_constant && immed_constant) {
                    /**
                     * Result is constant.  We already have initialized immed
                     * to zero.
                     */
                    aux.push_back(make_mov(btype, vd, immed));
                    break;
                }

                if (data_constant) {
                    /**
                     * We've computed all that we need to validity-bitwise
                     * in computing immed.
                     *
                     * Mark that we are using the tmp variable and sign extend
                     * to the result, if necessary.
                     */
                    tmpb[2u] = std::max(tmpb[2u], 1);

                    if (width == 4u) {
                        aux.push_back(make_mov(btype, vd, immed));
                    } else if (width == 8u) {
                        aux.push_back(make_cvt(s64_type, s32_type, vd, immed,
                            false));
                    } else {
                        assert(0 && "Unsupported width.");
                    }

                    break;
                }

                if (immed_constant) {
                    /**
                     * Insert the validity bits using c and d.
                     */
                    aux.push_back(make_bfi(statement.type, vd, va, vb,
                        statement.operands[3], statement.operands[4]));
                } else {
                    /**
                     * Insert, then OR with worst case validity results
                     * due to b and c.
                     */
                    const operand_t tmp1 = make_temp_operand(btype, 1);
                    const operand_t tmp2 = make_temp_operand(btype, 0);

                    assert(immed != tmp1);

                    aux.push_back(make_bfi(statement.type, tmp1, va, vb,
                        statement.operands[3], statement.operands[4]));

                    if (width == 4u) {
                        tmpb[2u] = std::max(tmpb[2u], 2);
                        aux.push_back(make_or(btype, vd, immed, tmp1));
                    } else if (width == 8u) {
                        /* Sign extend immed. */
                        tmpb[2u] = std::max(tmpb[2u], 1);
                        tmpb[3u] = std::max(tmpb[3u], 2);
                        assert(tmp2 != immed);

                        aux.push_back(make_cvt(s64_type, s32_type, tmp2,
                            immed, false));
                        immed = tmp2;
                        aux.push_back(make_or(btype, vd, immed, tmp1));
                    } else {
                        assert(0 && "Unsupported width.");
                    }
                }

                break; }
            case op_brev: {
                assert(statement.operands.size() == 2u);
                const operand_t & a = statement.operands[1];
                const operand_t & d = statement.operands[0];

                const operand_t va = make_validity_operand(a);
                const operand_t vd = make_validity_operand(d);

                switch (statement.type) {
                    case b32_type:
                    case b64_type:
                               if (va.is_constant()) {
                            aux.push_back(make_mov(statement.type, vd, va));
                        } else {
                            /**
                             * Bitwise reverse the validity bits.
                             */
                            aux.push_back(make_brev(statement.type, vd, va));
                        }

                        break;
                    case b16_type:
                    case f32_type:
                    case f64_type:
                    case s16_type:
                    case s32_type:
                    case s64_type:
                    case u16_type:
                    case u32_type:
                    case u64_type:
                    case s8_type:
                    case b8_type:
                    case u8_type:
                    case f16_type:
                    case pred_type:
                    case texref_type:
                    case invalid_type:
                        assert(0 && "Unsupported type.");
                        break;
                }

                break; }
            case op_brkpt: /* No-op */ break;
            case op_copysign: {
                assert(statement.operands.size() == 3u);
                assert(statement.type == f32_type ||
                       statement.type == f64_type);
                const operand_t & a = statement.operands[1];
                const operand_t & b = statement.operands[2];
                const operand_t & d = statement.operands[0];

                const operand_t va = make_validity_operand(a);
                const operand_t vb = make_validity_operand(b);
                const operand_t vd = make_validity_operand(d);

                const type_t btype = bitwise_type(statement.type);

                /**
                 * bfi and copysign both appear as part of sm_20 so bfi should
                 * be fair game to use.
                 */
                if (va.is_constant() && vb.is_constant()) {
                    aux.push_back(make_mov(btype, vd, va));
                } else {
                    /* One or both of the validity values is nonconstant,
                     * insert the sign bit into the rest of the data. */
                    aux.push_back(make_bfi(btype, vd, va, vb,
                        operand_t::make_iconstant(
                            (int) sizeof_type(statement.type) * CHAR_BIT - 1),
                        operand_t::make_iconstant(1)));
                }

                break; }
            case op_testp: {
                assert(statement.operands.size() == 2u);
                const operand_t & d = statement.operands[0];
                const operand_t & a = statement.operands[1];

                const operand_t vd = make_validity_operand(d);
                const operand_t va = make_validity_operand(a);

                if (va.is_constant()) {
                    /* Answer is well defined. */
                    aux.push_back(make_mov(b16_type, vd,
                        operand_t::make_iconstant(0)));
                } else {
                    /* Any invalid bits mean the result is invalid. */
                    const size_t width = sizeof_type(statement.type);
                    const size_t lwidth = log_size(width * CHAR_BIT);
                    const type_t btype = bitwise_type(statement.type);
                    const type_t stype = signed_type(statement.type);

                    tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 1);
                    const operand_t tmp = make_temp_operand(btype, 0);

                    aux.push_back(make_cnot(btype, tmp, va));
                    aux.push_back(make_sub(stype, tmp, tmp,
                        operand_t::make_iconstant(1)));
                    aux.push_back(make_cvt(s16_type, stype, vd, tmp, false));
                }

                break; }
            case op_cos:
            case op_ex2:
            case op_lg2:
            case op_rcp:
            case op_rsqrt:
            case op_sin:
            case op_sqrt: { /* Floating point, 1-argument operations. */
                assert((statement.type == f32_type ||
                        statement.type == f64_type) && "Invalid type.");
                assert(statement.operands.size() == 2u);

                const operand_t & a  = statement.operands[1];
                const operand_t   va = make_validity_operand(a);
                const bool constant  = va.is_constant();

                const operand_t & vd = make_validity_operand(
                    statement.operands[0]);
                const type_t btype = bitwise_type(statement.type);
                const type_t stype = signed_type(statement.type);
                const size_t width  = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);

                if (constant) {
                    /* Constant argument, so result is valid. */
                    aux.push_back(make_mov(btype, vd, va));
                } else {
                    assert(lwidth >= 1u);
                    assert(lwidth <= 4u);
                    tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 1);

                    /* Spread invalid bits. */
                    const operand_t tmp =
                        make_temp_operand(btype, 0);

                    aux.push_back(make_cnot(btype, tmp, va));
                    aux.push_back(make_sub(stype, vd, tmp,
                        operand_t::make_iconstant(1)));
                }

                break; }
            case op_cvt: {
                assert(statement.operands.size() == 2u);
                const operand_t & d = statement.operands[0];
                const operand_t & a = statement.operands[1];

                const operand_t vd = make_validity_operand(d);
                const operand_t va = make_validity_operand(a);

                const type_t dtype = statement.type;
                const type_t atype = statement.type2;

                if (!(statement.saturating)) {
                    bool is_signed_d, is_signed_a;
                    bool is_unsigned_d, is_unsigned_a;
                    is_signed_d = is_signed_a = false;
                    is_unsigned_d = is_unsigned_a = false;

                    switch (dtype) {
                        case u8_type:
                        case u16_type:
                        case u32_type:
                        case u64_type:
                            is_unsigned_d = true;
                            break;
                        case s8_type:
                        case s16_type:
                        case s32_type:
                        case s64_type:
                            is_signed_d = true;
                            break;
                        default:
                            break;
                    }

                    switch (atype) {
                        case u8_type:
                        case u16_type:
                        case u32_type:
                        case u64_type:
                            is_unsigned_a = true;
                            break;
                        case s8_type:
                        case s16_type:
                        case s32_type:
                        case s64_type:
                            is_signed_a = true;
                            break;
                        default:
                            break;
                    }

                    assert(!(is_signed_d && is_unsigned_d));
                    assert(!(is_signed_a && is_unsigned_a));
                    if (is_signed_d && is_signed_a) {
                        /* Sign extend the validity bits. */
                        aux.push_back(make_cvt(dtype, atype, vd, va, false));
                        break;
                    } else if (is_unsigned_d && is_unsigned_a) {
                        /* Zero extend the validity bits. */
                        aux.push_back(make_cvt(dtype, atype, vd, va, false));
                        break;
                    }
                }

                /* An invalid bit anywhere propagates across the value. */
                const type_t abtype = bitwise_type(atype);
                const type_t astype = signed_type(atype);
                const size_t awidth = sizeof_type(atype);
                const size_t alwidth = log_size(awidth * CHAR_BIT);
                tmpb[alwidth - 1u] = std::max(tmpb[alwidth - 1u], 1);

                const operand_t tmp = make_temp_operand(abtype, 0);
                aux.push_back(make_cnot(abtype, tmp, va));
                aux.push_back(make_sub(astype, tmp, tmp,
                    operand_t::make_iconstant(1)));
                /* Sign extend across target. */
                const type_t dstype = signed_type(dtype);
                aux.push_back(make_cvt(dstype, astype, vd, tmp, false));

                break; }
            case op_cvta:
                /**
                 * Just pass through invalid bits when loading from registers,
                 * but use mov's machinery for recognizing identifiers when
                 * initializing to a good value.
                 */
            case op_mov: {
                assert(statement.operands.size() == 2u);

                const operand_t & a = statement.operands[1];
                const operand_t & d = statement.operands[0];

                /* Short cut for the simple case, constants/sreg's */
                operand_t va = make_validity_operand(a);
                if (!(va.is_constant())) {
                    const size_t n = va.identifier.size();
                    std::vector<bool> done(n, false);
                    std::vector<bool> fixed(n, false);
                    size_t rem = n;
                    for (size_t i = 0; i < n; i++) {
                        if (va.identifier[i] == "0") {
                            done[i] = true;
                            fixed[i] = true;
                            rem--;
                        }
                    }

                    /* Walk up scopes for identifiers. */
                    const block_t * b = block;
                    const function_t * f = NULL;
                    while (b) {
                        assert(!(b->parent) || !(b->fparent));

                        if (rem == 0) {
                            break;
                        }

                        if (b->block_type == block_scope) {
                            const scope_t * s = b->scope;

                            for (size_t i = 0; i < n; i++) {
                                /* We're done with this identifier. */
                                if (done[i]) { continue; }

                                const size_t vn = s->variables.size();
                                for (size_t vi = 0; vi < vn; vi++) {
                                    const variable_t & v = s->variables[vi];
                                    if (a.identifier[i] == v.name &&
                                            !(v.has_suffix) &&
                                            v.space != reg_space) {
                                        /* This is a fixed symbol. */
                                        done[i] = true;
                                        fixed[i] = true;
                                        rem--;
                                        break;
                                    }

                                    /* TODO: Consider marking off variables
                                     * that we've resolved to registers. */
                                }
                            }
                        }

                        /* Move up. */
                        f = b->fparent;
                        b = b->parent;
                    }

                    if (rem > 0) {
                        assert(f);

                        /* TODO:  Check off parameters. */

                        const ptx_t * p = f->parent;
                        for (size_t i = 0; i < n; i++) {
                            /* We're done with this identifier. */
                            if (done[i]) { continue; }

                            const size_t vn = p->variables.size();
                            for (size_t vi = 0; vi < vn; vi++) {
                                const variable_t & v = p->variables[vi];
                                if (a.identifier[i] == v.name &&
                                        !(v.has_suffix) &&
                                        v.space != reg_space) {
                                    /* This is a fixed symbol. */
                                    done[i] = true;
                                    fixed[i] = true;
                                    rem--;
                                    break;
                                }
                            }
                        }
                    }

                    for (size_t i = 0; i < n; i++) {
                        if (fixed[i])  {
                            va.identifier[i] = "0";
                            va.field[i]      = field_none;
                        }
                    }
                }

                const operand_t vd = make_validity_operand(d);
                statement_t copy = statement;
                copy.op = op_mov;
                copy.type = bitwise_type(copy.type);

                copy.operands.clear();
                copy.operands.push_back(vd);
                copy.operands.push_back(va);
                aux.push_back(copy);

                break; }
            case op_min:
            case op_max:
                /**
                 * This is an imprecise calculation of the validity bits for
                 * min and max.  In principle, if the min/max value would
                 * remain the min/max value (respectively) regardless of the
                 * validity bits, we could safely propagate the validity bits
                 * of the min/max value *alone*.  Since doing so would require
                 * substantially more logic for a (hopefully) fleetingly rare
                 * condition, this imprecise method is preferable.
                 *
                 * Fallthrough.
                 */
            case op_div:
            case op_rem: {
                /**
                 * We work under the assumption that these operations are all
                 * or nothing w.r.t. validity.
                 */
                assert(statement.operands.size() == 3u);

                std::vector<operand_t> source_validity;
                for (unsigned i = 1; i < 3; i++) {
                    const operand_t & op  = statement.operands[i];
                    const operand_t   vop = make_validity_operand(op);
                    if (!(vop.is_constant())) {
                        source_validity.push_back(vop);
                    }
                }

                const size_t vargs = source_validity.size();
                const type_t btype = bitwise_type(statement.type);
                const type_t stype = signed_type(statement.type);

                const size_t width  = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);
                assert(lwidth >= 1u);
                assert(lwidth <= 4u);

                const operand_t & vd = make_validity_operand(
                    statement.operands[0]);

                const operand_t tmp0 = make_temp_operand(btype, 0);

                assert(vargs <= 2);
                switch (vargs) {
                    case 0:
                        /**
                         * No variable arguments, result is defined.
                         */
                        aux.push_back(make_mov(btype, vd,
                            operand_t::make_iconstant(0)));
                        break;
                    case 1:
                        /**
                         * Propagate from source.
                         */
                        tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 1);

                        aux.push_back(make_cnot(btype, tmp0,
                            source_validity[0]));
                        aux.push_back(make_sub(stype, vd, tmp0,
                            operand_t::make_iconstant(1)));
                        break;
                    case 2:
                        /**
                         * OR validity bits of two results, then propagate.
                         */
                        tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 1);

                        aux.push_back(make_or(btype, tmp0,
                            source_validity[0], source_validity[1]));
                        aux.push_back(make_cnot(btype, tmp0, tmp0));
                        aux.push_back(make_sub(stype, vd, tmp0,
                            operand_t::make_iconstant(1)));
                        break;
                }

                break; }
            case op_exit:
            case op_ret: {
                keep = false;

                /* setp.ne.u32 __has_errors, local_errors, 0 */
                statement_t setp;
                setp.op      = op_setp;
                setp.type    = u32_type;
                setp.cmp     = cmp_ne;
                setp.bool_op = bool_none;
                setp.has_ppredicate = true;
                setp.ppredicate = __has_errors;

                setp.operands.push_back(local_errors);

                const operand_t op_zero =
                    operand_t::make_iconstant(0);
                setp.operands.push_back(op_zero);

                aux.push_back(setp);

                /* @__has_errors atom.global.inc.u32 global_errors,
                 *   [reg_error_count], 0xFFFFFFFF */
                statement_t atom;

                atom.has_predicate = true;
                atom.predicate = __has_errors;

                atom.op        = op_atom;
                atom.space     = global_space;
                atom.atomic_op = atom_inc;
                atom.type      = u32_type;

                const operand_t op_ge =
                    operand_t::make_identifier(__reg_global_errors);
                atom.operands.push_back(op_ge);

                const operand_t op_error_count =
                    operand_t::make_identifier(__reg_error_count);
                atom.operands.push_back(op_error_count);

                const operand_t op_max_u32 =
                    operand_t::make_iconstant(0x3FFFFFFF);
                atom.operands.push_back(op_max_u32);

                aux.push_back(atom);

                /**
                 * We want the count to grow without limit rather than hit the
                 * edge case of wrapping around to precisely zero, but we need
                 * to bound the number of values we write out to the size of
                 * our buffer.
                 *
                 * min.u32 global_errors, global_errors, (buffer size)
                 */
                aux.push_back(make_min(u32_type, op_ge, op_ge,
                    operand_t::make_iconstant(
                    (sizeof(error_buffer_t) / sizeof(uint32_t)) - 1)));

                /* shl.b32 global_errors, global_errors, 2 */
                statement_t shl;
                shl.op  = op_shl;
                shl.type = b32_type;
                shl.operands.push_back(op_ge);
                shl.operands.push_back(op_ge);

                const operand_t op_two =
                    operand_t::make_iconstant(2);
                shl.operands.push_back(op_two);

                aux.push_back(shl);

                /* TODO:  This is not necessary when sizeof(void *) == 4 */
                /* cvt.PTRTYPE.u32 global_errorsw, global_errors */
                statement_t cvt;
                cvt.op = op_cvt;
                cvt.type = upointer_type();
                cvt.type2 = u32_type;

                const operand_t op_gew =
                    operand_t::make_identifier(__reg_global_errorsw);

                cvt.operands.push_back(op_gew);
                cvt.operands.push_back(op_ge);

                aux.push_back(cvt);

                /* add.PTRTYPE error_address, error_address, global_errorsw */
                statement_t add;
                add.op = op_add;
                add.type = upointer_type();

                const operand_t op_error_address =
                    operand_t::make_identifier(__errors_address);
                add.operands.push_back(op_error_address);
                add.operands.push_back(op_error_address);
                add.operands.push_back(op_gew);

                aux.push_back(add);

                /* @__has_errors st.global.u32 [error_address],
                 *   local_errors */
                statement_t st;
                st.has_predicate = true;
                st.predicate = __has_errors;
                st.op    = op_st;
                st.space = global_space;
                st.type  = u32_type;
                st.operands.push_back(op_error_address);
                st.operands.push_back(local_errors);
                aux.push_back(st);

                aux.push_back(statement);
                break; }
            case op_fma: { /* Floating point, 3-argument operations. */
                assert((statement.type == f32_type ||
                        statement.type == f64_type) && "Invalid type.");
                assert(statement.operands.size() == 4u);

                /**
                 * If any bits are invalid, assume the worst.
                 */
                std::vector<operand_t> source_validity;
                for (unsigned i = 1; i < 4; i++) {
                    const operand_t & op  = statement.operands[i];
                    const operand_t   vop = make_validity_operand(op);
                    if (!(vop.is_constant())) {
                        source_validity.push_back(vop);
                    }
                }

                const operand_t & vd = make_validity_operand(
                    statement.operands[0]);
                const type_t btype = bitwise_type(statement.type);
                const type_t stype = signed_type(statement.type);
                const size_t width  = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);

                const size_t vargs = source_validity.size();
                if (vargs == 0) {
                    /* No nonconstant arguments, so result is valid. */
                    aux.push_back(make_mov(btype, vd,
                        operand_t::make_iconstant(0)));
                } else {
                    /* One or more nonconstant arguments, fold into a
                     * temporary variable, then spread invalid bits. */
                    assert(lwidth >= 1u);
                    assert(lwidth <= 4u);
                    tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 1);

                    const operand_t tmp =
                        make_temp_operand(btype, 0);
                    if (vargs == 1u) {
                        /* Move directly into temporary variable. */
                        aux.push_back(make_mov(btype, tmp,
                            source_validity[0]));
                    } else {
                        aux.push_back(make_or(btype, tmp,
                            source_validity[0], source_validity[1]));
                        for (size_t i = 2; i < vargs; i++) {
                            aux.push_back(make_or(btype, tmp,
                                tmp, source_validity[i]));
                        }
                    }

                    /* Spread invalid bits. */
                    aux.push_back(make_cnot(btype, tmp, tmp));
                    aux.push_back(make_sub(stype, vd, tmp,
                        operand_t::make_iconstant(1)));
                }

                break; }
            case op_invalid: /* No-op */ break;
            case op_isspacep: {
                assert(statement.operands.size() == 2u);

                /* Narrow a pointer type to a b16. */
                const operand_t & p = statement.operands[0];
                const operand_t & a = statement.operands[1];

                const operand_t vp = make_validity_operand(p);
                assert(!(vp.is_constant()));
                const operand_t va = make_validity_operand(a);

                if (va.is_constant()) {
                    aux.push_back(make_mov(b16_type, vp,
                        operand_t::make_iconstant(0)));
                } else {
                    const operand_t tmpptr = operand_t::make_identifier(
                        "__panoptes_ptr0");
                    const operand_t tmp = make_temp_operand(b16_type, 0);

                    tmpb[1u] = std::max(tmpb[1u], 1);
                    tmp_ptr = std::max(tmp_ptr, 1);

                    /* Map 0 -> 1, (everything) -> 0 */
                    aux.push_back(make_cnot(pointer_type(), tmpptr, va));

                    /* Narrow to b16 */
                    aux.push_back(make_cvt(u16_type, upointer_type(), tmp,
                        tmpptr, false));

                    /* Map 0 -> 0, (everything) -> 0xFFFF */
                    aux.push_back(make_sub(s16_type, vp, tmp,
                        operand_t::make_iconstant(1)));
                }

                break; }
            case op_ld:
            case op_ldu: {
                assert(statement.operands.size() == 2u);
                const operand_t & src = statement.operands[1];
                /**
                 * We convert:
                 *    dst   = *src;
                 *
                 * into:
                 *    const size_t max_chunks =
                 *      (1u << (lg_max_memory - lg_chunk_bytes));
                 *    const size_t chunk_size =
                 *      (1u << lg_chunk_bytes);
                 *          void * global_wo;
                 *    const void * global_ro;
                 *
                 *    uint16_t error = 0;
                 *
                 *    -- TODO Track the max chidx seen per thread before mask,
                 *            warn if exceeded.
                 *
                 *    size_t chidx = (src >> lg_chunk_bytes) &
                 *                   (max_chunks - 1u);
                 *    size_t inidx = (src & (chunk_size - 1u)) >> 3u;
                 *    const metadata_chunk * chunk = __master_symbol[chidx];
                 *
                 *    -- This is appropriately sized for vector+type combo
                 *    typedef FOO read_t;
                 *    read_t  a     = chunk->a_data[inidx];
                 *    uint8_t apass = a ? 1 : 0;
                 *    uint8_t npass = 1 - apass;
                 *    -- BAR is based on the line of PTX
                 *    error          = max(error, npass * BAR);
                 *    -- There are some type casts here.
                 *    void * new_src = src * apass + global_ro * npass;
                 *    *dst  = new_src;
                 *    *vdst = chunk->vdata[inidx];
                 *
                 * TODO:  Check alignment.
                 */
                const operand_t chidx =
                    operand_t::make_identifier("__panoptes_ptr6");
                const operand_t inidx =
                    operand_t::make_identifier("__panoptes_ptr1");
                const operand_t chunk =
                    operand_t::make_identifier("__panoptes_ptr2");
                const operand_t master =
                    operand_t::make_identifier(__master_symbol);
                const operand_t chunk_ptr =
                    operand_t::make_identifier("__panoptes_ptr3");
                const operand_t a_data_ptr =
                    operand_t::make_addressable("__panoptes_ptr3",
                    offsetof(metadata_chunk, a_data));
                const operand_t data_ptr =
                    operand_t::make_identifier("__panoptes_ptr4");
                const operand_t validity_ptr_src =
                    operand_t::make_identifier("__panoptes_ptr5");
                const operand_t validity_ptr =
                    operand_t::make_addressable("__panoptes_ptr5",
                    offsetof(metadata_chunk, v_data));
                const operand_t original_ptr =
                    operand_t::make_identifier("__panoptes_ptr0");
                const operand_t global_ro =
                    operand_t::make_identifier(__global_ro);
                const operand_t global_ro_reg =
                    operand_t::make_identifier("__panoptes_ptr7");
                const operand_t vidx =
                    operand_t::make_identifier("__panoptes_ptr8");
                const std::string valid_pred = "__panoptes_pred0";
                const size_t chunk_size = 1u << lg_chunk_bytes;
                const size_t max_chunks =
                    (1u << (lg_max_memory - lg_chunk_bytes)) - 1u;
                const size_t width = sizeof_type(statement.type) *
                    (unsigned) statement.vector;
                const type_t read_t = unsigned_of(width);
                const operand_t a_data = make_temp_operand(
                    unsigned_of(4u << log_size(width)), 0);

                const type_t wread_t =
                    width <= 8 ? u16_type : read_t;
                const operand_t a_wdata =
                    operand_t::make_identifier("__panoptes_u16_0");
                const operand_t a_data32 =
                    operand_t::make_identifier("__panoptes_u32_0");
                const operand_t a_cdata =
                    width <= 8 ? a_wdata : a_data;

                operand_t new_src, new_vsrc;

                int ld_tmp[4] = {0, 0, 0, 0};
                int ld_pred = 0;
                int ld_ptr  = 0;

                int log_ptr;
                switch (sizeof(void *)) {
                    case 4u: log_ptr = 2; break;
                    case 8u: log_ptr = 3; break;
                }

                bool drop_load = false;
                bool invalidate = false;

                switch (src.op_type) {
                    case operand_identifier:
                    case operand_addressable:
                        if (statement.space == param_space) {
                            assert(src.identifier.size() == 1u);
                            new_src  = src;
                            new_vsrc = operand_t::make_identifier(
                                make_validity_symbol(src.identifier[0]));
                            if (src.op_type == operand_addressable) {
                                new_vsrc.op_type = operand_addressable;
                                new_vsrc.offset  = src.offset;
                            }

                            break;
                        } else if (statement.space == shared_space) {
                            assert(src.identifier.size() == 1u);
                            const std::string & id = src.identifier[0];

                            /* Walk up scopes for identifiers. */
                            const block_t * b = block;
                            const function_t * f = NULL;

                            bool found = false;
                            bool flexible = false;
                            size_t size;

                            while (b && !(found)) {
                                assert(!(b->parent) || !(b->fparent));
                                assert(b->block_type == block_scope);
                                const scope_t * s = b->scope;

                                const size_t vn = s->variables.size();
                                for (size_t vi = 0; vi < vn; vi++) {
                                    const variable_t & v = s->variables[vi];
                                    if (id == v.name && !(v.has_suffix) &&
                                            v.space != reg_space) {
                                        found = true;
                                        size = v.size();
                                        break;
                                    }
                                }

                                /* Move up. */
                                f = b->fparent;
                                b = b->parent;
                            }

                            if (!(found)) {
                                assert(f);

                                const ptx_t * p = f->parent;
                                const size_t vn = p->variables.size();
                                for (size_t vi = 0; vi < vn; vi++) {
                                    const variable_t & v = p->variables[vi];
                                    if (id == v.name && !(v.has_suffix) &&
                                            v.space != reg_space) {
                                        found = true;
                                        flexible = v.array_flexible;
                                        if (!(flexible)) {
                                            size = v.size();
                                        }
                                        break;
                                    }
                                }
                            }

                            if (found && !(flexible)) {
                                /* We found a fixed symbol, verify we do not
                                 * statically overrun it. */
                                if (src.offset < 0) {
                                    std::stringstream ss;
                                    ss << statement;

                                    char msg[256];
                                    int ret = snprintf(msg, sizeof(msg),
                                        "Shared load of %zu bytes at offset "
                                        "%ld will underrun buffer:\n"
                                        "Disassembly: %s\n", width,
                                        src.offset, ss.str().c_str());

                                    assert(ret < (int) sizeof(msg) - 1);
                                    logger::instance().print(msg);

                                    /* Cast off bits into the ether. */
                                    drop_load = true;
                                    invalidate = true;
                                    break;
                                }

                                const size_t end = width + (size_t) src.offset;
                                if (end > size) {
                                    std::stringstream ss;
                                    ss << statement;

                                    char msg[256];
                                    int ret = snprintf(msg, sizeof(msg),
                                        "Shared store of %zu bytes at offset "
                                        "%ld will overrun buffer:\n"
                                        "Disassembly: %s\n", width,
                                        src.offset, ss.str().c_str());

                                    assert(ret < (int) sizeof(msg) - 1);
                                    logger::instance().print(msg);

                                    /* Cast off bits into the ether. */
                                    drop_load = true;
                                    invalidate = true;
                                } else {
                                    /* Map it to the validity symbol. */
                                    new_src  = src;

                                    new_vsrc = src;
                                    new_vsrc.identifier.clear();
                                    new_vsrc.identifier.push_back(
                                        make_validity_symbol(id));
                                    new_vsrc.field.clear();
                                    new_vsrc.field.push_back(field_none);
                                }

                                break;
                            } else {
                                /* Verify address against the shared size
                                 * parameter. TODO:  Verify we don't overrun
                                 * the buffer. */

                                const operand_t limit =
                                    operand_t::make_identifier(__shared_reg);

                                drop_load = true;
                                invalidate = false;

                                ld_pred = 2;
                                ld_ptr = 2;

                                assert(src.identifier.size() == 1u);
                                aux.push_back(make_mov(
                                    pointer_type(), original_ptr,
                                    operand_t::make_identifier(
                                        src.identifier[0])));
                                if (src.op_type == operand_addressable &&
                                        src.offset != 0) {
                                    if (src.offset > 0) {
                                        aux.push_back(make_add(upointer_type(),
                                            original_ptr, original_ptr,
                                            operand_t::make_iconstant(
                                                src.offset)));
                                    } else {
                                        aux.push_back(make_sub(upointer_type(),
                                            original_ptr, original_ptr,
                                            operand_t::make_iconstant(
                                                -src.offset)));
                                    }
                                }

                                const std::string not_valid_pred = "__panoptes_pred1";

                                aux.push_back(make_setp(upointer_type(),
                                    cmp_ge, valid_pred, not_valid_pred, limit,
                                    original_ptr));

                                statement_t new_load = statement;
                                /**
                                 * TODO:  We need to propagate the predicate
                                 * flags of predicated loads.
                                 */
                                assert(!(new_load.has_predicate));
                                new_load.has_predicate = true;
                                new_load.predicate = valid_pred;
                                aux.push_back(new_load);

                                const operand_t vsrc =
                                    operand_t::make_identifier("__panoptes_ptr1");

                                aux.push_back(make_add(upointer_type(),
                                    vsrc, original_ptr, limit));

                                const type_t btype =
                                    bitwise_type(statement.type);

                                const operand_t vdst = make_validity_operand(
                                    statement.operands[0]);

                                statement_t new_vload = statement;
                                new_vload.has_predicate = true;
                                new_vload.predicate = valid_pred;
                                new_vload.type = btype;
                                new_vload.operands[0] = vdst;
                                new_vload.operands[1] = vsrc;
                                aux.push_back(new_vload);

                                /* If we don't do the load, invalidate on the
                                 * not_valid_pred flag. */
                                const size_t ni = vdst.identifier.size();
                                const operand_t negone =
                                    operand_t::make_iconstant(-1);

                                /**
                                 * nvcc does not seem to generate byte-sized
                                 * registers, so while the load is a byte-wide,
                                 * we don't get clued in on the size of our
                                 * target register.  For now, widen the
                                 * register (and hope this behavior doesn't
                                 * change).
                                 */
                                const type_t bwtype = (btype == b8_type) ?
                                    b16_type : btype;

                                for (size_t i = 0; i < ni; i++) {
                                    operand_t o;
                                    o.op_type = vdst.op_type;
                                    o.identifier.push_back(vdst.identifier[i]);
                                    o.field.push_back(vdst.field[i]);

                                    statement_t s;
                                    s.op = op_mov;
                                    s.type = bwtype;
                                    s.has_predicate = true;
                                    s.predicate = not_valid_pred;
                                    s.operands.push_back(o);
                                    s.operands.push_back(negone);
                                    aux.push_back(s);
                                }

                                break;
                            }

                            assert(0 && "Unreachable.");
                            break;
                        } else if (statement.space == local_space) {
                            assert(src.identifier.size() == 1u);
                            const std::string & id = src.identifier[0];

                            /* Verify address against the shared size
                             * parameter. TODO:  Verify we don't overrun
                             * the buffer. */

                            const operand_t limit =
                                operand_t::make_identifier(__local_reg);

                            drop_load = true;
                            invalidate = false;

                            ld_pred = 2;
                            ld_ptr = 2;

                            aux.push_back(make_mov(
                                pointer_type(), original_ptr,
                                operand_t::make_identifier(id)));
                            if (src.op_type == operand_addressable &&
                                    src.offset != 0) {
                                if (src.offset > 0) {
                                    aux.push_back(make_add(upointer_type(),
                                        original_ptr, original_ptr,
                                        operand_t::make_iconstant(
                                            src.offset)));
                                } else {
                                    aux.push_back(make_sub(upointer_type(),
                                        original_ptr, original_ptr,
                                        operand_t::make_iconstant(
                                            -src.offset)));
                                }
                            }

                            const std::string not_valid_pred = "__panoptes_pred1";

                            aux.push_back(make_setp(upointer_type(),
                                cmp_ge, valid_pred, not_valid_pred, limit,
                                original_ptr));

                            statement_t new_load = statement;
                            /**
                             * TODO:  We need to propagate the predicate
                             * flags of predicated loads.
                             */
                            assert(!(new_load.has_predicate));
                            new_load.has_predicate = true;
                            new_load.predicate = valid_pred;
                            aux.push_back(new_load);

                            const operand_t vsrc =
                                operand_t::make_identifier("__panoptes_ptr1");

                            aux.push_back(make_add(upointer_type(),
                                vsrc, original_ptr, limit));

                            const type_t btype =
                                bitwise_type(statement.type);

                            const operand_t vdst = make_validity_operand(
                                statement.operands[0]);

                            statement_t new_vload = statement;
                            new_vload.has_predicate = true;
                            new_vload.predicate = valid_pred;
                            new_vload.type = btype;
                            new_vload.operands[0] = vdst;
                            new_vload.operands[1] = vsrc;
                            aux.push_back(new_vload);

                            /* If we don't do the load, invalidate on the
                             * not_valid_pred flag. */
                            const size_t ni = vdst.identifier.size();
                            const operand_t negone =
                                operand_t::make_iconstant(-1);

                            /**
                             * nvcc does not seem to generate byte-sized
                             * registers, so while the load is a byte-wide,
                             * we don't get clued in on the size of our
                             * target register.  For now, widen the
                             * register (and hope this behavior doesn't
                             * change).
                             */
                            const type_t bwtype = (btype == b8_type) ?
                                b16_type : btype;

                            for (size_t i = 0; i < ni; i++) {
                                operand_t o;
                                o.op_type = vdst.op_type;
                                o.identifier.push_back(vdst.identifier[i]);
                                o.field.push_back(vdst.field[i]);

                                statement_t s;
                                s.op = op_mov;
                                s.type = bwtype;
                                s.has_predicate = true;
                                s.predicate = not_valid_pred;
                                s.operands.push_back(o);
                                s.operands.push_back(negone);
                                aux.push_back(s);
                            }

                            break;
                        }

                        new_src  = data_ptr;
                        new_vsrc = validity_ptr_src;

                        ld_tmp[log_size(width) - 1u]++;
                        if (width <= 8) {
                            ld_tmp[1] = 1;
                            ld_tmp[2] = 1;
                        }
                        ld_pred = 1u;
                        ld_ptr  = 9;

                        aux.push_back(make_mov(pointer_type(), chidx,
                            operand_t::make_identifier(src.identifier[0])));
                        if (src.op_type == operand_addressable &&
                                src.offset != 0) {
                            if (src.offset > 0) {
                                aux.push_back(make_add(upointer_type(), chidx,
                                    chidx,
                                    operand_t::make_iconstant(src.offset)));
                            } else {
                                aux.push_back(make_sub(upointer_type(), chidx,
                                    chidx,
                                    operand_t::make_iconstant(-src.offset)));
                            }
                        }

                        aux.push_back(make_mov(upointer_type(), global_ro_reg,
                            global_ro));
                        aux.push_back(make_mov(pointer_type(), original_ptr,
                            chidx));
                        aux.push_back(make_mov(pointer_type(), inidx, chidx));

                        aux.push_back(make_shr(pointer_type(), chidx,
                            chidx, operand_t::make_iconstant(lg_chunk_bytes)));
                        aux.push_back(make_and(pointer_type(), chidx,
                            chidx, operand_t::make_iconstant(
                                max_chunks - 1)));
                        aux.push_back(make_shl(pointer_type(), chidx,
                            chidx, operand_t::make_iconstant(log_ptr)));

                        aux.push_back(make_and(pointer_type(), inidx, inidx,
                            operand_t::make_iconstant(chunk_size - 1)));
                        aux.push_back(make_mov(pointer_type(), vidx, inidx));
                        aux.push_back(make_shr(pointer_type(), inidx,
                            inidx, operand_t::make_iconstant(3)));

                        aux.push_back(make_ld(pointer_type(), const_space,
                            chunk, master));
                        aux.push_back(make_add(upointer_type(), chunk, chidx,
                            chunk));
                        aux.push_back(make_ld(pointer_type(), global_space,
                            chunk_ptr, chunk));

                        aux.push_back(make_add(upointer_type(),
                            validity_ptr_src, chunk_ptr, vidx));
                        aux.push_back(make_add(upointer_type(), chunk_ptr,
                            chunk_ptr, inidx));
                        aux.push_back(make_ld(read_t, global_space,
                            a_data, a_data_ptr));
                        if (width <= 8) {
                            /* We cannot setp on a u8 */
                            aux.push_back(make_cvt(wread_t, read_t,
                                a_wdata, a_data, false));

                            /**
                             * Since we load a byte at a time (the address bits
                             * for 8 bytes) even if we are performing a load of
                             * less than 8 bytes, we need to shift and mask.
                             *
                             * Grab lower bits of the address and shift
                             * accordingly.  Shifts take u32's as their
                             * shift argument.
                             */
                            const type_t bwread_t = bitwise_type(wread_t);
                            switch (sizeof(void *)) {
                                case 8u:
                                aux.push_back(make_cvt(u32_type,
                                    upointer_type(), a_data32, vidx, false));
                                break;
                                case 4u:
                                aux.push_back(make_mov(u32_type,
                                    a_data32, vidx));
                                break;
                            }
                            aux.push_back(make_and(b32_type,
                                a_data32, a_data32, operand_t::make_iconstant(
                                sizeof(void *) - 1)));
                            aux.push_back(make_shr(wread_t,
                                a_wdata, a_wdata, a_data32));

                            /**
                             * Mask high bits.
                             */
                            aux.push_back(make_and(bwread_t,
                                a_wdata, a_wdata, operand_t::make_iconstant(
                                (1 << width) - 1)));
                        }

                        aux.push_back(make_setp(wread_t, cmp_eq, valid_pred,
                            a_cdata, operand_t::make_iconstant((1 << width) - 1)));
                        aux.push_back(make_selp(pointer_type(), valid_pred,
                            data_ptr, original_ptr, global_ro_reg));

                        if (offsetof(metadata_chunk, v_data)) {
                            aux.push_back(make_add(upointer_type(),
                                validity_ptr_src, validity_ptr_src,
                                operand_t::make_iconstant(
                                offsetof(metadata_chunk, v_data))));
                        }

                        break;
                    case operand_constant:
                        aux.push_back(make_mov(pointer_type(), chidx,
                            ((static_cast<uintptr_t>(src.offset) >>
                                lg_chunk_bytes) &
                             (max_chunks - 1u)) * sizeof(void *)));
                        aux.push_back(make_mov(pointer_type(), inidx,
                            (static_cast<uintptr_t>(src.offset) &
                             chunk_size) >> 3u));
                        aux.push_back(make_ld(pointer_type(), const_space,
                            chunk, master));
                        aux.push_back(make_add(upointer_type(), chunk, chidx,
                            chunk));
                        aux.push_back(make_ld(pointer_type(), global_space,
                            chunk_ptr, chunk));
                        aux.push_back(make_ld(read_t, global_space,
                            a_data, a_data_ptr));
                        aux.push_back(make_setp(read_t, cmp_eq, valid_pred,
                            a_data, operand_t::make_iconstant((1 << width) - 1)));
                        aux.push_back(make_selp(pointer_type(), valid_pred,
                            data_ptr, src, global_ro));
                        aux.push_back(make_add(upointer_type(),
                            validity_ptr_src, chunk_ptr, inidx));

                        new_src  = data_ptr;
                        new_vsrc = validity_ptr;

                        break;
                    default:
                        assert(0 && "Unsupported operand type for load source.");
                        break;
                }

                for (unsigned i = 0; i < 4u; i++) {
                    tmpu[i] = std::max(tmpu[i], ld_tmp[i]);
                }
                tmp_pred = std::max(tmp_pred, ld_pred);
                tmp_ptr  = std::max(tmp_ptr,  ld_ptr);

                keep = false;

                if (invalidate) {
                    /**
                     * TODO:  We may want to actually fill in the data with
                     * some data.
                     *
                     * Mark destinations as invalid.
                     */
                    const operand_t & dst = statement.operands[0];
                    const size_t ni = dst.identifier.size();
                    const type_t btype = bitwise_type(statement.type);
                    const operand_t negone = operand_t::make_iconstant(-1);
                    for (size_t i = 0; i < ni; i++) {
                        aux.push_back(make_mov(btype,
                            operand_t::make_identifier(
                                make_validity_symbol(dst.identifier[i])),
                            negone));
                    }
                }

                if (!(drop_load)) {
                    statement_t new_load = statement;
                    new_load.operands[1] = new_src;
                    aux.push_back(new_load);

                    statement_t new_vload = statement;
                    new_vload.type = bitwise_type(new_vload.type);
                    operand_t & vdst = new_vload.operands[0];
                    assert(vdst.op_type == operand_identifier);
                    assert(vdst.identifier.size() == vdst.field.size());
                    for (size_t i = 0; i < vdst.identifier.size(); i++) {
                        vdst.identifier[i] = make_validity_symbol(
                            vdst.identifier[i]);
                    }
                    new_vload.operands[1] = new_vsrc;
                    aux.push_back(new_vload);
                }

                break; }
            case op_mad:
            case op_mad24:
            case op_madc: {
                assert(statement.operands.size() == 4u);

                std::vector<operand_t> source_validity;
                for (unsigned i = 1; i < 4; i++) {
                    const operand_t & op   = statement.operands[i];
                    const operand_type_t t = op.op_type;
                    switch (t) {
                        case operand_identifier:
                            source_validity.push_back(
                                make_validity_operand(op));
                            break;
                        case operand_addressable:
                        case operand_indexed:
                        case invalid_operand:
                            assert(0 && "Invalid operand type.");
                            break;
                        case operand_constant:
                        case operand_float:
                        case operand_double:
                            /* Skip. */
                            break;
                    }
                }

                const operand_t & vd = make_validity_operand(
                    statement.operands[0]);
                const type_t btype = bitwise_type(statement.type);
                const type_t stype = signed_type(statement.type);
                const size_t width  = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);

                const operand_t tmp = make_temp_operand(btype, 0);
                const operand_t tmp1 = make_temp_operand(btype, 1);

                const operand_t vc = operand_t::make_identifier(__vcarry);

                const size_t vargs = source_validity.size();
                const bool carry_in = (statement.op == op_madc);
                if (vargs == 0) {
                    if (carry_in) {
                        /* Copy carry flag validity bits. */
                        aux.push_back(make_mov(btype, vd, vc));

                        /* Validity of carry-out is validity carry-in, so do
                         * nothing. */
                    } else {
                        /* No nonconstant arguments, so result is valid. */
                        aux.push_back(make_mov(btype, vd,
                            operand_t::make_iconstant(0)));

                        /* Clear carry out. */
                        if (statement.carry_out) {
                            aux.push_back(make_mov(btype, vc,
                                operand_t::make_iconstant(0)));
                        }
                    }

                    break;
                } else {
                    /* One or more nonconstant arguments, fold into a
                     * temporary variable, then spread invalid bits. */
                    assert(lwidth >= 1u);
                    assert(lwidth <= 4u);
                    tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 1);

                    if (vargs == 1u) {
                        /* Move directly into temporary variable. */
                        aux.push_back(make_mov(btype, tmp,
                            source_validity[0]));
                    } else {
                        aux.push_back(make_or(btype, tmp,
                            source_validity[0], source_validity[1]));
                        for (size_t i = 2; i < vargs; i++) {
                            aux.push_back(make_or(btype, tmp,
                                tmp, source_validity[i]));
                        }
                    }

                    /* Fold-in the carry bits. */
                    if (carry_in) {
                        aux.push_back(make_or(btype, tmp, tmp, vc));
                    }
                }

                switch (statement.type) {
                    case u16_type:
                    case u32_type:
                    case u64_type:
                    case s16_type:
                    case s32_type:
                    case s64_type:
                        /**
                         * In the worst case, invalid bits are propagated to
                         * bits as or more significant as the original invalid
                         * bits.
                         */
                        tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 2);

                        /**
                         * Spread left.
                         */
                        aux.push_back(make_neg(stype, tmp1, tmp));
                        aux.push_back(make_or(btype, vd, tmp, tmp1));

                        /**
                         * Carry out.
                         */
                        if (statement.carry_out) {
                            assert(statement.type == u32_type ||
                                   statement.type == s32_type);
                            aux.push_back(make_shr(stype, vc, vd,
                                operand_t::make_iconstant(31)));
                        }

                        break;
                    case f32_type:
                    case f64_type:
                        assert(!(statement.carry_out));

                        /* Spread invalid bits across type. */
                        aux.push_back(make_cnot(btype, tmp, tmp));
                        aux.push_back(make_sub(stype, vd, tmp,
                            operand_t::make_iconstant(1)));
                        break;
                    case b8_type:
                    case b16_type:
                    case b32_type:
                    case b64_type:
                    case u8_type:
                    case s8_type:
                    case f16_type:
                    case pred_type:
                    case texref_type:
                    case invalid_type:
                        assert(0 && "Invalid type.");
                        break;
                }

                break; }
            case op_membar: /* No-op */ break;
            case op_neg: {
                int neg_tmps[4] = {0, 0, 0, 0};

                const size_t width  = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);
                const type_t btype  = bitwise_type(statement.type);

                const operand_t & vd = make_validity_operand(
                    statement.operands[0]);
                const operand_t & va = make_validity_operand(
                    statement.operands[1]);

                switch (statement.type) {
                    case s16_type:
                    case s32_type:
                    case s64_type: {
                        neg_tmps[lwidth - 1u] = 1;

                        const type_t stype  = statement.type;
                        const operand_t tmp =
                            make_temp_operand(stype, 0);
                        aux.push_back(make_neg(stype, tmp, va));
                        aux.push_back(make_or(btype, vd, tmp, va));
                        break; }
                    case f32_type:
                    case f64_type:
                        /**
                         * TODO:  Do not ignore ftz.
                         *
                         * The resulting validity is the same as the source
                         * validity as we are only flipping the sign bit.
                         *
                         * For now, we ignore that NaN's produce unspecified
                         * NaNs (per PTX ISA 3.0).
                         */

                        aux.push_back(make_mov(btype, vd, va));
                        break;
                    case s8_type:
                    case b8_type:
                    case u8_type:
                    case b16_type:
                    case f16_type:
                    case u16_type:
                    case b32_type:
                    case u32_type:
                    case b64_type:
                    case u64_type:
                    case pred_type:
                    case texref_type:
                    case invalid_type:
                        assert(0 && "Unsupported type.");
                        break;
                }

                for (unsigned i = 0; i < 4u; i++) {
                    tmps[i] = std::max(tmps[i], neg_tmps[i]);
                }

                break; }
            case op_not: {
                assert(statement.operands.size() == 2u);
                const operand_t & a = statement.operands[1];
                const operand_t & d = statement.operands[0];

                const operand_t va = make_validity_operand(a);
                const operand_t vd = make_validity_operand(d);

                switch (statement.type) {
                    case pred_type:
                        /**
                         * Validity information is simply transfered.
                         */
                        aux.push_back(make_mov(b16_type, va, vd));

                        assert(d.identifier.size() > 0);
                        inst->unchecked.insert(d.identifier[0]);
                        break;
                    case b16_type:
                    case b32_type:
                    case b64_type:
                        aux.push_back(make_mov(statement.type, vd, va));
                        break;
                    case f32_type:
                    case f64_type:
                    case s16_type:
                    case s32_type:
                    case s64_type:
                    case u16_type:
                    case u32_type:
                    case u64_type:
                    case s8_type:
                    case b8_type:
                    case u8_type:
                    case f16_type:
                    case texref_type:
                    case invalid_type:
                        assert(0 && "Unsupported type.");
                        break;
                }

                break; }
            case op_bra: /* No-op */ break;
            case op_call: /* No-op */ break;
            case op_cnot: {
                /**
                 * We cannot reuse the popc codepath for cnot as it reduces
                 * the width of its output to a u32 regardless of the
                 * instruction datatype.
                 */
                assert(statement.operands.size() == 2u);
                const type_t btype = bitwise_type(statement.type);

                const operand_t & a = statement.operands[1];
                const operand_t & d = statement.operands[0];

                const operand_t & vd = make_validity_operand(d);

                const operand_t va = make_validity_operand(a);
                bool constant = va.is_constant();

                if (constant) {
                    /* Result is completely valid. */
                    aux.push_back(make_mov(btype, vd, va));
                    break;
                }

                const operand_t tmp = make_temp_operand(btype, 0);

                switch (statement.type) {
                    case b16_type:
                        tmpb[1] = std::max(tmpb[1], 1);
                        break;
                    case b32_type:
                        tmpb[2] = std::max(tmpb[2], 1);
                        break;
                    case b64_type:
                        tmpb[3] = std::max(tmpb[3], 1);
                        break;
                    default:
                        assert(0 && "Invalid type.");
                        break;
                }

                /**
                 * Map 0 -> 1, (everything else) -> 0
                 */
                aux.push_back(make_cnot(btype, tmp, va));

                /**
                 * Map 0 -> 0, (everything else) -> 1.
                 */
                aux.push_back(make_xor(btype, vd, tmp,
                    operand_t::make_iconstant(0x1)));

                break; }
            case op_bfind:
            case op_clz:
                /**
                 * This approximation is less precise than it could be, but
                 * it is likely good enough for our purposes.
                 *
                 * Fall through.
                 */
            case op_popc: {
                assert(statement.operands.size() == 2u);
                const type_t btype = bitwise_type(statement.type);

                const operand_t & a = statement.operands[1];
                const operand_t & d = statement.operands[0];

                const operand_t & va = make_validity_operand(a);
                const operand_t & vd = make_validity_operand(d);

                const bool constant = va.is_constant();

                if (constant) {
                    /* Result is completely valid. */
                    aux.push_back(make_mov(btype, vd, va));
                    break;
                }

                /**
                 * Suppose we compute the popc of the valid data bits, e.g.
                 * AND(d, NOT(v)), and label this result "ret."  The invalid
                 * bits make a contribution of 0 to popc(v) to ret to obtain
                 * the popc in the result.  Per our rules for addition, since
                 * the least significant bit is invalid, we must left propagate
                 * it as it could carry to more significant bits.
                 */
                const operand_t tmp = make_temp_operand(btype, 0);
                operand_t ntmp;
                int    mret;

                switch (statement.type) {
                    case b32_type:
                    case s32_type: /* Due to bfind */
                    case u32_type: /* Due to bfind */
                        tmpb[2] = std::max(tmpb[2], 1);
                        ntmp  = tmp;
                        mret  = 32;
                        break;
                    case b64_type:
                    case s64_type: /* Due to bfind */
                    case u64_type: /* Due to bfind */
                        /* The result must be narrowed. */
                        tmpb[2] = std::max(tmpb[2], 1);
                        tmpb[3] = std::max(tmpb[3], 1);
                        ntmp    = make_temp_operand(b32_type, 0);

                        mret = 64;
                        break;
                    default:
                        __builtin_unreachable();
                        assert(0 && "Invalid type.");
                        break;
                }

                /**
                 * Map 0 -> 1, (everything else) -> 0
                 */
                aux.push_back(make_cnot(btype, tmp, va));

                /**
                 * Narrow, if applicable.
                 */
                const size_t width = sizeof_type(statement.type);
                if (width == 8u) {
                    aux.push_back(make_cvt(u32_type, u64_type, ntmp, tmp,
                        false));
                }

                /**
                 * Map 0 -> 0, (everything else) -> 0xFFFFFFFF
                 */
                aux.push_back(make_sub(u32_type, ntmp, ntmp,
                    operand_t::make_iconstant(1)));

                bool has_mask;
                int mask;
                switch (statement.op) {
                    case op_bfind:
                        /**
                         * bfind can return a value between 0 and mret - 1
                         * inclusive or it can return 0xFFFFFFFF.
                         */
                        has_mask = false;
                        break;
                    case op_clz:
                    case op_popc:
                        /* Map 0 -> 0 (modulo 2 * mret). */
                        has_mask = true;
                        mask = 2 * mret - 1;
                        break;
                    default:
                        assert(0 && "Unknown opcode.");
                        break;
                }

                if (has_mask) {
                    /*
                     * The bits above those needed to represent the maximum
                     * result will be zero.
                     */
                    aux.push_back(make_and(b32_type, vd, ntmp,
                        operand_t::make_iconstant(mask)));
                }

                break; }
            case op_pmevent: /* No-op */ break;
            case op_prmt: {
                assert(statement.operands.size() == 4u);
                assert(statement.type == b32_type);

                const operand_t & a = statement.operands[1];
                const operand_t & b = statement.operands[2];
                const operand_t & c = statement.operands[3];
                const operand_t & d = statement.operands[0];

                const operand_t va = make_validity_operand(a);
                const operand_t vb = make_validity_operand(b);
                const operand_t vc = make_validity_operand(c);
                const operand_t vd = make_validity_operand(d);

                assert(!(vd.is_constant()));

                /**
                 * If c is constant, then we can just select the validity
                 * with c from va and vb.
                 */
                if (vc.is_constant()) {
                    if (va.is_constant() && vb.is_constant()) {
                        aux.push_back(make_mov(b32_type, vd, va));
                    } else {
                        statement_t vprmt = statement;
                        vprmt.operands[0] = vd;
                        vprmt.operands[1] = va;
                        vprmt.operands[2] = vb;
                        aux.push_back(vprmt);
                    }

                    break;
                }

                /**
                 * We always need a temporary 32-bit value.
                 */
                const operand_t tmp = make_temp_operand(b32_type, 0);
                int prmt_tmpb[4] = {0, 0, 1, 0};
                operand_t immed;
                bool direct;

                if (va.is_constant() && vb.is_constant()) {
                    /* We can move the results directly into vd */
                    immed = vd;
                    direct = true;
                } else {
                    immed = tmp;
                    direct = false;
                }

                /* An invalid bit anywhere corrupts the output. */
                int64_t mask;
                if (statement.prmt_mode == prmt_default) {
                    mask = 0x0000FFFF;
                } else {
                    mask = 0x00000003;
                }
                aux.push_back(make_and(b32_type, tmp, vc,
                    operand_t::make_iconstant(mask)));
                aux.push_back(make_cnot(b32_type, tmp, tmp));
                aux.push_back(make_sub(s32_type, immed, tmp,
                    operand_t::make_iconstant(1)));

                if (direct) {
                    /* We're done. */
                    for (unsigned i = 0; i < 4; i++) {
                        tmpb[i] = std::max(tmpb[i], prmt_tmpb[i]);
                    }
                    break;
                }

                bool just_convert;
                operand_t immed2;
                operand_t immed3;
                if (va.is_constant() & vb.is_constant()) {
                    /* We're done after we convert. */
                    immed2 = vd;
                    just_convert = true;
                } else {
                    /* We need two more intermediate values, one to hold the
                     * result of selecting and one to hold the converted
                     * contents of immed. */
                    prmt_tmpb[2u] = 2;
                    just_convert = false;

                    immed2 = make_temp_operand(b32_type, 1);
                    immed3 = make_temp_operand(b32_type, 0);
                }

                assert(immed3 == immed);
                std::swap(immed2, immed3);

                if (!(just_convert)) {
                    /* Mix in validity of va and vb via selection. */
                    statement_t vprmt = statement;
                    vprmt.operands[0] = immed3;
                    vprmt.operands[1] = va;
                    vprmt.operands[2] = vb;
                    aux.push_back(vprmt);
                    aux.push_back(make_or(b32_type, vd, immed2, immed3));
                }

                /* We're done. */
                for (unsigned i = 0; i < 4; i++) {
                    tmpb[i] = std::max(tmpb[i], prmt_tmpb[i]);
                }
                break; }
            case op_sad: {
                assert(statement.operands.size() == 4u);

                assert(statement.type == u16_type ||
                       statement.type == u32_type ||
                       statement.type == u64_type ||
                       statement.type == s16_type ||
                       statement.type == s32_type ||
                       statement.type == s64_type);

                /**
                 * We first need to consider the validity bits of the
                 * intermediate abs(a-b).  The invalid bits of a and b
                 * propagate to the left; abs propagates the invalidity of the
                 * sign bit to the right.  Therefore, if any bits of a or b are
                 * invalid, the result abs(a-b) is wholly invalid.
                 *
                 * We place the result of the intermediate validity bits in
                 * the operand immed.
                 */
                bool immed_constant;
                operand_t immed;

                std::vector<operand_t> source_validity;
                for (unsigned i = 1; i < 3; i++) {
                    const operand_t & op  = statement.operands[i];
                    const operand_t   vop = make_validity_operand(op);
                    if (!(vop.is_constant())) {
                        source_validity.push_back(vop);
                    }
                }

                const size_t vargs = source_validity.size();
                const type_t btype = bitwise_type(statement.type);
                const type_t stype = signed_type(statement.type);

                const size_t width  = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);

                const operand_t tmp0 = make_temp_operand(btype, 0);

                assert(vargs <= 2);
                switch (vargs) {
                    case 0:
                        /**
                         * No variable arguments, result is defined.
                         */
                        immed = operand_t::make_iconstant(0);
                        immed_constant = true;
                        break;
                    case 1:
                        /**
                         * Propagate from source.
                         */
                        aux.push_back(make_cnot(btype, tmp0,
                            source_validity[0]));
                        aux.push_back(make_sub(stype, tmp0, tmp0,
                            operand_t::make_iconstant(1)));

                        immed = tmp0;
                        immed_constant = false;
                        break;
                    case 2:
                        /**
                         * OR validity bits of two results, then propagate.
                         */
                        aux.push_back(make_or(btype, tmp0,
                            source_validity[0], source_validity[1]));
                        aux.push_back(make_cnot(btype, tmp0, tmp0));
                        aux.push_back(make_sub(stype, tmp0, tmp0,
                            operand_t::make_iconstant(1)));
                        immed = tmp0;
                        immed_constant = false;
                        break;
                }

                /**
                 * Consider the argument to c.
                 */
                const operand_t & c  = statement.operands[3];
                const operand_t   vc = make_validity_operand(c);
                const bool cconstant = vc.is_constant();

                const operand_t & vd = make_validity_operand(
                    statement.operands[0]);

                assert(lwidth >= 1u);
                assert(lwidth <= 4u);

                if (cconstant && immed_constant) {
                    /**
                     * Result is constant.  We already have initialized immed
                     * to zero.
                     */
                    aux.push_back(make_mov(btype, vd, immed));
                    break;
                }

                if (cconstant) {
                    /**
                     * We've computed all that we need to validity-bitwise
                     * in computing immed.
                     *
                     * Mark that we are using the tmp variable.
                     */
                    tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 1);
                    aux.push_back(make_mov(btype, vd, immed));
                    break;
                }

                if (immed_constant) {
                    /**
                     * We need to left propagate vc.
                     *
                     * Mark our use of the temporary variable.
                     */
                    tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 1);

                    aux.push_back(make_neg(stype, tmp0, vc));
                    aux.push_back(make_or(btype, vd, tmp0, vc));
                } else {
                    /**
                     * OR result and then left propgate.
                     *
                     * We need the temporary variable for immed and another
                     * to compute our left propagation.
                     */
                    tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 2);
                    const operand_t tmp1 = make_temp_operand(btype, 1);

                    assert(immed != tmp1);

                    aux.push_back(make_or(btype, immed, immed, vc));
                    aux.push_back(make_neg(stype, tmp1, immed));
                    aux.push_back(make_or(btype, vd, tmp1, immed));
                }

                break; }
            case op_shl:
            case op_shr: {
                assert(statement.operands.size() == 3u);

                const operand_t & a  = statement.operands[1];
                const operand_t   va = make_validity_operand(a);
                const bool aconstant = va.is_constant();

                const operand_t & b  = statement.operands[2];
                const operand_t   vb = make_validity_operand(b);
                const bool bconstant = vb.is_constant();

                const operand_t & d = statement.operands[0];
                const operand_t vd = make_validity_operand(d);

                /**
                 * We shift the validity bits just as the data bits are shifted
                 * with the provision that if the shift amount is in any way
                 * invalid, the entire result will be invalid.
                 *
                 * PTX clamps shifts greater than the register width to the
                 * register width, so the bits beyond those needed to represent
                 * the width still carry significance (and therefore cannot be
                 * masked out).
                 */
                const type_t btype = bitwise_type(statement.type);
                const type_t stype = signed_type(statement.type);
                if (aconstant && bconstant) {
                    /* Result is completely valid. */
                    aux.push_back(make_mov(btype, vd,
                        operand_t::make_iconstant(0)));
                } else if (aconstant) {
                    /* Result is dependant on the validity bits of b.  b is
                     * always a 32-bit value. */
                    const size_t width = sizeof_type(statement.type);

                    tmpb[2u] = std::max(tmpb[2u], 1);
                    const operand_t tmp = make_temp_operand(b32_type, 0);

                    aux.push_back(make_cnot(b32_type, tmp, vb));
                    if (width == 4u) {
                        /* We can write the result directly. */
                        aux.push_back(make_sub(s32_type, vd, tmp,
                            operand_t::make_iconstant(1)));
                    } else {
                        /* Compute into temporary, the convert. */
                        aux.push_back(make_sub(s32_type, tmp, tmp,
                            operand_t::make_iconstant(1)));
                        aux.push_back(make_cvt(stype, s32_type, vd, tmp,
                            false));
                    }
                } else if (bconstant) {
                    /* Shift the validity bits of a according to b. */
                    statement_t sh;
                    sh.op   = statement.op;
                    sh.type = statement.type;
                    sh.operands.push_back(vd);
                    sh.operands.push_back(va);
                    sh.operands.push_back(b);
                    aux.push_back(sh);
                } else {
                    /* Join sources of validity bits. */
                    const size_t width = sizeof_type(statement.type);
                    const size_t lwidth = log_size(width * CHAR_BIT);

                    const operand_t tmp0    = make_temp_operand(btype, 0);

                    statement_t sh;
                    sh.op   = statement.op;
                    sh.type = statement.type;
                    sh.operands.push_back(tmp0);
                    sh.operands.push_back(va);
                    sh.operands.push_back(b);
                    aux.push_back(sh);

                    operand_t intermediate;
                    if (width == 4u) {
                        tmpb[2u] = std::max(tmpb[2u], 2);
                        const operand_t tmp1 = make_temp_operand(b32_type, 1);

                        aux.push_back(make_cnot(b32_type, tmp1, vb));
                        aux.push_back(make_sub(s32_type, tmp1, tmp1,
                            operand_t::make_iconstant(1)));
                        intermediate = tmp1;
                    } else {
                        assert(lwidth != 3u);
                        tmpb[2u] = std::max(tmpb[2u], 1);
                        tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 2);

                        assert(btype != b32_type);
                        const operand_t tmp32_0 = make_temp_operand(b32_type, 0);
                        assert(tmp0 != tmp32_0);
                        const operand_t tmp1 = make_temp_operand(btype, 1);

                        aux.push_back(make_cnot(b32_type, tmp32_0, vb));
                        aux.push_back(make_sub(s32_type, tmp32_0, tmp32_0,
                            operand_t::make_iconstant(1)));
                        aux.push_back(make_cvt(stype, s32_type, tmp1, tmp32_0,
                            false));

                        intermediate = tmp1;
                    }

                    /* OR the result from shifting and from spreading b's
                     * invalid bits. */
                    aux.push_back(make_or(btype, vd, tmp0, intermediate));
                }

                break;
                }
            case op_selp: {
                assert(statement.operands.size() == 4u);

                const operand_t & d = statement.operands[0];
                const operand_t & a = statement.operands[1];
                const operand_t & b = statement.operands[2];
                const operand_t & c = statement.operands[3];
                assert(!(c.is_constant()));

                const operand_t vd = make_validity_operand(d);
                const operand_t va = make_validity_operand(a);
                const operand_t vb = make_validity_operand(b);
                const operand_t vc = make_validity_operand(c);

                int selp_tmpb[4] = {0, 0, 0, 0};
                int selp_tmps[4] = {0, 0, 0, 0};

                /**
                 * Sign extend validity bits associated with c to the width of
                 * the operation.
                 */
                const size_t width = sizeof_type(statement.type);
                const size_t lwidth = log_size(width * CHAR_BIT);
                const type_t stype = signed_type(statement.type);
                const operand_t tmp = make_temp_operand(stype, 0);
                operand_t immed;
                if (stype == s16_type) {
                    immed = vc;
                } else {
                    immed = tmp;
                    selp_tmps[lwidth - 1u]++;
                    aux.push_back(make_cvt(stype, s16_type, tmp, vc, false));
                }

                const type_t btype = bitwise_type(statement.type);
                if (!(va.is_constant()) || !(vb.is_constant())) {
                    /* Mix validity bits from a and b according to c. */
                    const operand_t btmp = make_temp_operand(btype, 0);
                    selp_tmpb[lwidth - 1u]++;

                    statement_t selp;
                    selp.op = op_selp;
                    selp.type = btype;
                    selp.operands.push_back(btmp);
                    selp.operands.push_back(va);
                    selp.operands.push_back(vb);
                    selp.operands.push_back(c);
                    aux.push_back(selp);

                    aux.push_back(make_or(btype, vd, btmp, immed));
                } else {
                    /* Move results of vc into vd. */
                    aux.push_back(make_mov(btype, vd, immed));
                }

                for (unsigned i = 0; i < 4; i++) {
                    tmpb[i] = std::max(tmpb[i], selp_tmpb[i]);
                    tmps[i] = std::max(tmps[i], selp_tmps[i]);
                }

                break; }
            case op_set: {
                assert(statement.operands.size() >= 3u &&
                       statement.operands.size() <= 4u);

                const operand_t & d = statement.operands[0];
                const operand_t & a = statement.operands[1];
                const operand_t & b = statement.operands[2];

                const operand_t vd = make_validity_operand(d);
                const operand_t va = make_validity_operand(a);
                const operand_t vb = make_validity_operand(b);

                const type_t bdtype = bitwise_type(statement.type);
                const type_t sdtype = signed_type(statement.type);
                const size_t dwidth = sizeof_type(statement.type);
                const size_t ldwidth = log_size(dwidth * CHAR_BIT);

                operand_t immed;
                if (va.is_constant() && vb.is_constant()) {
                    immed = va;
                } else {
                    const type_t bstype = bitwise_type(statement.type2);
                    const type_t sstype = signed_type(statement.type2);
                    const size_t swidth = sizeof_type(statement.type2);
                    const size_t lswidth = log_size(swidth * CHAR_BIT);

                    const operand_t tmp = make_temp_operand(bstype, 0);

                    operand_t tmpd;
                    if (swidth == dwidth) {
                        tmpb[ldwidth - 1u] = std::max(tmpb[ldwidth - 1u], 1);
                        tmpd = tmp;
                    } else {
                        tmpb[ldwidth - 1u] = std::max(tmpb[ldwidth - 1u], 1);
                        tmpb[lswidth - 1u] = std::max(tmpb[lswidth - 1u], 1);

                        tmpd = make_temp_operand(bdtype, 0);
                    }

                    if (!(va.is_constant()) && vb.is_constant()) {
                        immed = va;
                    } else if (va.is_constant() && !(vb.is_constant())) {
                        immed = vb;
                    } else {
                        /* Merge invalid bits. */
                        immed = tmp;
                        aux.push_back(make_or(bstype, tmp, va, vb));
                    }

                    /* 0 -> 1, (anything) -> 0 */
                    aux.push_back(make_cnot(bstype, tmp, immed));

                    if (swidth != dwidth) {
                        aux.push_back(make_cvt(sdtype, sstype, tmpd, tmp,
                            false));
                    }

                    /* 0 -> 0, (anything) -> -1 */
                    aux.push_back(make_sub(sdtype, tmpd, tmpd,
                        operand_t::make_iconstant(1)));

                    immed = tmpd;
                }

                /**
                 * Fold in validity bits from c, if provided.
                 */
                if (statement.operands.size() == 4u) {
                    const operand_t & c = statement.operands[2];
                    const operand_t vc = make_validity_operand(c);

                    if (immed.is_constant()) {
                        immed = vc;
                    } else {
                        tmpb[ldwidth - 1u] = std::max(tmpb[ldwidth - 1u], 2);
                        const operand_t tmp2 =
                            make_temp_operand(bdtype, 1);

                        aux.push_back(make_cvt(sdtype, s16_type, tmp2, vc,
                            false));
                        aux.push_back(make_or(bdtype, immed, immed, tmp2));
                    }
                }

                aux.push_back(make_mov(bdtype, vd, immed));

                break; }
            case op_setp: {
                assert(statement.operands.size() >= 2u &&
                       statement.operands.size() <= 3u);

                const operand_t & a = statement.operands[0];
                const operand_t & b = statement.operands[1];
                const operand_t va = make_validity_operand(a);
                const operand_t vb = make_validity_operand(b);

                operand_t immed;
                if (va.is_constant() && vb.is_constant()) {
                    immed = va;
                } else {
                    const type_t btype = bitwise_type(statement.type);
                    const type_t stype = signed_type(statement.type);
                    const size_t width = sizeof_type(statement.type);
                    const size_t lwidth = log_size(width * CHAR_BIT);

                    const operand_t tmp = make_temp_operand(btype, 0);

                    operand_t tmp16;
                    if (width == 2u) {
                        tmpb[1u] = std::max(tmpb[1u], 1);
                        tmp16 = tmp;
                    } else {
                        assert(lwidth != 1u);
                        tmpb[1u] = std::max(tmpb[1u], 1);
                        tmpb[lwidth - 1u] = std::max(tmpb[lwidth - 1u], 1);

                        tmp16 = make_temp_operand(b16_type, 0);
                    }

                    if (!(va.is_constant()) && vb.is_constant()) {
                        immed = va;
                    } else if (va.is_constant() && !(vb.is_constant())) {
                        immed = vb;
                    } else {
                        /* Merge invalid bits. */
                        immed = tmp;
                        aux.push_back(make_or(btype, tmp, va, vb));
                    }

                    /* 0 -> 1, (anything) -> 0 */
                    aux.push_back(make_cnot(btype, tmp, immed));

                    if (width != 2u) {
                        /* Convert to 16 bits */
                        aux.push_back(make_cvt(s16_type, stype, tmp16, tmp,
                            false));
                    }

                    /* 0 -> 0, (anything) -> 0xFFFF */
                    aux.push_back(make_sub(s16_type, tmp16, tmp16,
                        operand_t::make_iconstant(1)));

                    immed = tmp16;
                }

                /**
                 * Fold in validity bits from c, if provided.
                 */
                if (statement.operands.size() == 3u) {
                    const operand_t & c = statement.operands[2];
                    const operand_t vc = make_validity_operand(c);

                    if (immed.is_constant()) {
                        immed = vc;
                    } else {
                        aux.push_back(make_or(b16_type, immed, immed, vc));
                    }
                }

                assert(statement.has_ppredicate);
                const operand_t vp = operand_t::make_identifier(
                    make_validity_symbol(statement.ppredicate));
                operand_t vq;
                if (statement.has_qpredicate) {
                    vq = operand_t::make_identifier(
                        make_validity_symbol(statement.qpredicate));
                }

                aux.push_back(make_mov(b16_type, vp, immed));
                inst->unchecked.insert(statement.ppredicate);
                if (statement.has_qpredicate) {
                    aux.push_back(make_mov(b16_type, vq, immed));
                    inst->unchecked.insert(statement.qpredicate);
                }

                break; }
            case op_slct: {
                assert(statement.operands.size() == 4u);
                assert(statement.type2 == f32_type ||
                       statement.type2 == s32_type);

                const operand_t & a = statement.operands[1];
                const operand_t & b = statement.operands[2];
                const operand_t & c = statement.operands[3];
                const operand_t & d = statement.operands[0];

                const operand_t va = make_validity_operand(a);
                const operand_t vb = make_validity_operand(b);
                const operand_t vc = make_validity_operand(c);
                const operand_t vd = make_validity_operand(d);

                assert(!(vd.is_constant()));

                /**
                 * If c is constant, then we can just select the validity
                 * with c from va and vb.
                 */
                const type_t btype = bitwise_type(statement.type);
                if (vc.is_constant()) {
                    if (va.is_constant() && vb.is_constant()) {
                        aux.push_back(make_mov(btype, vd, va));
                    } else {
                        aux.push_back(make_slct(btype, statement.type2,
                            vd, va, vb, c));
                    }

                    break;
                }

                /**
                 * We always need a temporary 32-bit value.
                 */
                const operand_t tmp = make_temp_operand(b32_type, 0);
                int slct_tmpb[4] = {0, 0, 1, 0};

                const size_t width = sizeof_type(statement.type);
                operand_t immed;
                bool direct;
                if (va.is_constant() && vb.is_constant() && width == 4u) {
                    /* We can move the results directly into vd */
                    immed = vd;
                    direct = true;
                } else {
                    immed = tmp;
                    direct = false;
                }

                /**
                 * We spread the invalid bits of c according to the type
                 * of the operation, we then narrow to the destination type.
                 */
                if (statement.type2 == f32_type) {
                    /* An invalid bit anywhere corrupts the output. */
                    aux.push_back(make_cnot(b32_type, tmp, vc));
                    aux.push_back(make_sub(s32_type, immed, tmp,
                        operand_t::make_iconstant(1)));
                } else if (statement.type2 == s32_type) {
                    /* The sign bit governs whether the output is
                     * invalid */
                    aux.push_back(make_shr(s32_type, immed, vd,
                        operand_t::make_iconstant(31)));
                }

                if (direct) {
                    /* We're done. */
                    for (unsigned i = 0; i < 4; i++) {
                        tmpb[i] = std::max(tmpb[i], slct_tmpb[i]);
                    }
                    break;
                }

                bool just_convert;
                operand_t immed2;
                operand_t immed3;
                if (va.is_constant() & vb.is_constant()) {
                    /* We're done after we convert. */
                    immed2 = vd;
                    just_convert = true;
                } else {
                    /* We need two more intermediate values, one to hold the
                     * result of selecting and one to hold the converted
                     * contents of immed. */
                    const size_t lwidth = log_size(width * CHAR_BIT);
                    assert(lwidth >= 1u);
                    assert(lwidth <= 4u);
                    slct_tmpb[lwidth - 1u] = 2;
                    just_convert = false;

                    immed2 = make_temp_operand(btype, 1);
                    immed3 = make_temp_operand(btype, 0);
                }

                if (width != 4u) {
                    /* Our width didn't match.  Convert. */
                    assert(immed2 != immed);
                    aux.push_back(make_cvt(signed_type(statement.type),
                        s32_type, immed2, immed, true));
                } else {
                    assert(immed3 == immed);
                    std::swap(immed2, immed3);
                }

                if (!(just_convert)) {
                    /* Mix in validity of va and vb via selection. */
                    aux.push_back(make_slct(btype, statement.type2,
                        immed3, va, vb, c));
                    aux.push_back(make_or(btype, vd, immed2, immed3));
                }

                /* We're done. */
                for (unsigned i = 0; i < 4; i++) {
                    tmpb[i] = std::max(tmpb[i], slct_tmpb[i]);
                }
                break; }
            case op_st: {
                assert(statement.operands.size() == 2u);
                const operand_t & dst = statement.operands[0];
                /*
                 * This is similar to op_ld
                 */
                const operand_t chidx =
                    operand_t::make_identifier("__panoptes_ptr6");
                const operand_t inidx =
                    operand_t::make_identifier("__panoptes_ptr1");
                const operand_t chunk =
                    operand_t::make_identifier("__panoptes_ptr2");
                const operand_t master =
                    operand_t::make_identifier(__master_symbol);
                const operand_t chunk_ptr =
                    operand_t::make_identifier("__panoptes_ptr3");
                const operand_t a_data_ptr =
                    operand_t::make_addressable("__panoptes_ptr3",
                    offsetof(metadata_chunk, a_data));
                const operand_t data_ptr =
                    operand_t::make_identifier("__panoptes_ptr4");
                const operand_t validity_ptr_dst =
                    operand_t::make_identifier("__panoptes_ptr5");
                const operand_t validity_ptr =
                    operand_t::make_addressable("__panoptes_ptr5",
                    offsetof(metadata_chunk, v_data));
                const operand_t original_ptr =
                    operand_t::make_identifier("__panoptes_ptr0");
                const operand_t global_ro =
                    operand_t::make_identifier(__global_ro);
                const operand_t global_wo =
                    operand_t::make_identifier(__global_wo);
                const operand_t global_wo_reg =
                    operand_t::make_identifier("__panoptes_ptr7");
                const operand_t vidx =
                    operand_t::make_identifier("__panoptes_ptr8");
                const std::string valid_pred = "__panoptes_pred0";
                const size_t chunk_size = 1u << lg_chunk_bytes;
                const size_t max_chunks =
                    (1u << (lg_max_memory - lg_chunk_bytes)) - 1u;
                const size_t width = sizeof_type(statement.type) *
                    (unsigned) statement.vector;
                const type_t read_t = unsigned_of(width);
                const operand_t a_data = make_temp_operand(
                    unsigned_of(4u << log_size(width)), 0);

                const type_t wread_t =
                    width <= 8 ? u16_type : read_t;
                const operand_t a_wdata =
                    operand_t::make_identifier("__panoptes_u16_0");
                const operand_t a_cdata =
                    width <= 8 ? a_wdata : a_data;
                const operand_t a_data32 =
                    operand_t::make_identifier("__panoptes_u32_0");

                operand_t new_dst, new_vdst;

                int st_tmp[4] = {0, 0, 0, 0};
                int st_pred = 0;
                int st_ptr  = 0;

                int log_ptr;
                switch (sizeof(void *)) {
                    case 4u: log_ptr = 2; break;
                    case 8u: log_ptr = 3; break;
                }

                bool drop_store = false;

                switch (dst.op_type) {
                    case operand_identifier:
                    case operand_addressable:
                        if (statement.space == param_space) {
                            assert(dst.identifier.size() == 1u);
                            new_dst  = dst;
                            new_vdst = operand_t::make_identifier(
                                make_validity_symbol(dst.identifier[0]));
                            break;
                        } else if (statement.space == shared_space) {
                            assert(dst.identifier.size() == 1u);
                            const std::string & id = dst.identifier[0];

                            /* Walk up scopes for identifiers. */
                            const block_t * b = block;
                            const function_t * f = NULL;

                            bool found = false;
                            bool flexible = false;
                            size_t size;

                            while (b && !(found)) {
                                assert(!(b->parent) || !(b->fparent));
                                assert(b->block_type == block_scope);
                                const scope_t * s = b->scope;

                                const size_t vn = s->variables.size();
                                for (size_t vi = 0; vi < vn; vi++) {
                                    const variable_t & v = s->variables[vi];
                                    if (id == v.name && !(v.has_suffix) &&
                                            v.space != reg_space) {
                                        found = true;
                                        size = v.size();
                                        break;
                                    }
                                }

                                /* Move up. */
                                f = b->fparent;
                                b = b->parent;
                            }

                            if (!(found)) {
                                assert(f);

                                const ptx_t * p = f->parent;
                                const size_t vn = p->variables.size();
                                for (size_t vi = 0; vi < vn; vi++) {
                                    const variable_t & v = p->variables[vi];
                                    if (id == v.name && !(v.has_suffix) &&
                                            v.space != reg_space) {
                                        found = true;
                                        flexible = v.array_flexible;
                                        if (!(flexible)) {
                                            size = v.size();
                                        }
                                        break;
                                    }
                                }
                            }

                            if (found && !(flexible)) {
                                /* We found a fixed symbol, verify we do not
                                 * statically overrun it. */
                                if (dst.offset < 0) {
                                    std::stringstream ss;
                                    ss << statement;

                                    char msg[256];
                                    int ret = snprintf(msg, sizeof(msg),
                                        "Shared store of %zu bytes at offset "
                                        "%ld will underrun buffer:\n"
                                        "Disassembly: %s\n", width,
                                        dst.offset, ss.str().c_str());

                                    assert(ret < (int) sizeof(msg) - 1);
                                    logger::instance().print(msg);

                                    /* Cast off bits into the ether. */
                                    drop_store = true;
                                    break;
                                }

                                const size_t end = width + (size_t) dst.offset;
                                if (end > size) {
                                    std::stringstream ss;
                                    ss << statement;

                                    char msg[256];
                                    int ret = snprintf(msg, sizeof(msg),
                                        "Shared store of %zu bytes at offset "
                                        "%ld will overrun buffer:\n"
                                        "Disassembly: %s\n", width,
                                        dst.offset, ss.str().c_str());

                                    assert(ret < (int) sizeof(msg) - 1);
                                    logger::instance().print(msg);

                                    /* Cast off bits into the ether. */
                                    drop_store = true;
                                } else {
                                    /* Map it to the validity symbol. */
                                    new_dst  = dst;

                                    new_vdst = dst;
                                    new_vdst.identifier.clear();
                                    new_vdst.identifier.push_back(
                                        make_validity_symbol(id));
                                    new_vdst.field.clear();
                                    new_vdst.field.push_back(field_none);
                                }

                                break;
                            } else {
                                /* Verify address against the shared size
                                 * parameter. TODO:  Verify we don't overrun
                                 * the buffer. */

                                const operand_t limit =
                                    operand_t::make_identifier(__shared_reg);

                                drop_store = true;
                                st_pred = 1;
                                st_ptr = 2;

                                aux.push_back(make_mov(
                                    pointer_type(), original_ptr,
                                    operand_t::make_identifier(
                                        dst.identifier[0])));
                                if (dst.op_type == operand_addressable &&
                                        dst.offset != 0) {
                                    if (dst.offset > 0) {
                                        aux.push_back(make_add(upointer_type(),
                                            original_ptr, original_ptr,
                                            operand_t::make_iconstant(dst.offset)));
                                    } else {
                                        aux.push_back(make_sub(upointer_type(),
                                            original_ptr, original_ptr,
                                            operand_t::make_iconstant(-dst.offset)));
                                    }
                                }

                                aux.push_back(make_setp(upointer_type(),
                                    cmp_ge, valid_pred, limit, original_ptr));

                                statement_t new_store = statement;
                                /**
                                 * TODO:  We need to propagate the predicate
                                 * flags of predicated stores.
                                 */
                                assert(!(new_store.has_predicate));
                                new_store.has_predicate = true;
                                new_store.predicate = valid_pred;
                                aux.push_back(new_store);

                                const operand_t vdst =
                                    operand_t::make_identifier("__panoptes_ptr1");

                                aux.push_back(make_add(upointer_type(),
                                    vdst, original_ptr, limit));

                                statement_t new_vstore = statement;
                                new_vstore.has_predicate = true;
                                new_vstore.predicate = valid_pred;
                                new_vstore.type = bitwise_type(
                                    new_vstore.type);
                                new_vstore.operands[0] = vdst;
                                new_vstore.operands[1] =
                                    make_validity_operand(
                                    new_vstore.operands[1]);
                                aux.push_back(new_vstore);
                                break;
                            }

                            assert(0 && "Unreachable.");
                            break;
                        } else if (statement.space == local_space) {
                            assert(dst.identifier.size() == 1u);
                            const std::string & id = dst.identifier[0];

                            /* Verify address against the shared size
                             * parameter. TODO:  Verify we don't overrun
                             * the buffer. */

                            const operand_t limit =
                                operand_t::make_identifier(__local_reg);

                            drop_store = true;

                            st_pred = 1;
                            st_ptr = 2;

                            aux.push_back(make_mov(
                                pointer_type(), original_ptr,
                                operand_t::make_identifier(id)));
                            if (dst.op_type == operand_addressable &&
                                    dst.offset != 0) {
                                if (dst.offset > 0) {
                                    aux.push_back(make_add(upointer_type(),
                                        original_ptr, original_ptr,
                                        operand_t::make_iconstant(dst.offset)));
                                } else {
                                    aux.push_back(make_sub(upointer_type(),
                                        original_ptr, original_ptr,
                                        operand_t::make_iconstant(-dst.offset)));
                                }
                            }

                            aux.push_back(make_setp(upointer_type(),
                                cmp_ge, valid_pred, limit, original_ptr));

                            statement_t new_store = statement;
                            /**
                             * TODO:  We need to propagate the predicate
                             * flags of predicated stores.
                             */
                            assert(!(new_store.has_predicate));
                            new_store.has_predicate = true;
                            new_store.predicate = valid_pred;
                            aux.push_back(new_store);

                            const operand_t vdst =
                                operand_t::make_identifier("__panoptes_ptr1");

                            aux.push_back(make_add(upointer_type(),
                                vdst, original_ptr, limit));

                            statement_t new_vstore = statement;
                            new_vstore.has_predicate = true;
                            new_vstore.predicate = valid_pred;
                            new_vstore.type = bitwise_type(
                                new_vstore.type);
                            new_vstore.operands[0] = vdst;
                            new_vstore.operands[1] =
                                make_validity_operand(
                                new_vstore.operands[1]);
                            aux.push_back(new_vstore);
                            break;
                        }

                        aux.push_back(make_mov(upointer_type(), global_wo_reg,
                            global_wo));

                        aux.push_back(make_mov(pointer_type(), chidx,
                            operand_t::make_identifier(dst.identifier[0])));
                        if (dst.op_type == operand_addressable &&
                                dst.offset != 0) {
                            if (dst.offset > 0) {
                                aux.push_back(make_add(upointer_type(), chidx,
                                    chidx,
                                    operand_t::make_iconstant(dst.offset)));
                            } else {
                                aux.push_back(make_sub(upointer_type(), chidx,
                                    chidx,
                                    operand_t::make_iconstant(-dst.offset)));
                            }
                        }

                        aux.push_back(make_mov(pointer_type(), original_ptr,
                            chidx));
                        aux.push_back(make_mov(pointer_type(), inidx, chidx));

                        aux.push_back(make_shr(pointer_type(), chidx,
                            chidx, operand_t::make_iconstant(lg_chunk_bytes)));
                        aux.push_back(make_and(pointer_type(), chidx,
                            chidx, operand_t::make_iconstant(
                                max_chunks - 1)));
                        aux.push_back(make_shl(pointer_type(), chidx,
                            chidx, operand_t::make_iconstant(log_ptr)));

                        aux.push_back(make_and(pointer_type(), inidx, inidx,
                            operand_t::make_iconstant(chunk_size - 1)));
                        aux.push_back(make_mov(pointer_type(), vidx, inidx));
                        aux.push_back(make_shr(pointer_type(), inidx,
                            inidx, operand_t::make_iconstant(3)));

                        aux.push_back(make_ld(pointer_type(), const_space,
                            chunk, master));
                        aux.push_back(make_add(upointer_type(), chunk, chidx,
                            chunk));
                        aux.push_back(make_ld(pointer_type(), global_space,
                            chunk_ptr, chunk));

                        aux.push_back(make_add(upointer_type(),
                            validity_ptr_dst, chunk_ptr, vidx));

                        aux.push_back(make_add(upointer_type(), chunk_ptr,
                            chunk_ptr, inidx));
                        aux.push_back(make_ld(read_t, global_space,
                            a_data, a_data_ptr));
                        if (width <= 8) {
                            /* We cannot setp on a u8 */
                            aux.push_back(make_cvt(wread_t, read_t,
                                a_wdata, a_data, false));

                            /* See explaination under op_ld.
                             *
                             * Grab lower bits of the address and shift
                             * accordingly.  Shifts take u32's as their
                             * shift argument.
                             */
                            const type_t bwread_t = bitwise_type(wread_t);
                            switch (sizeof(void *)) {
                                case 8u:
                                aux.push_back(make_cvt(u32_type,
                                    upointer_type(), a_data32, vidx,
                                    false));
                                break;
                                case 4u:
                                aux.push_back(make_mov(u32_type,
                                    a_data32, vidx));
                                break;
                            }
                            aux.push_back(make_and(b32_type,
                                a_data32, a_data32, operand_t::make_iconstant(
                                sizeof(void *) - 1)));
                            aux.push_back(make_shr(wread_t,
                                a_wdata, a_wdata, a_data32));

                            /**
                             * Mask high bits.
                             */
                            aux.push_back(make_and(bwread_t,
                                a_wdata, a_wdata, operand_t::make_iconstant(
                                (1 << width) - 1)));
                        }

                        aux.push_back(make_setp(wread_t, cmp_eq, valid_pred,
                            a_cdata, operand_t::make_iconstant((1 << width) - 1)));

                        aux.push_back(make_selp(pointer_type(), valid_pred,
                            data_ptr, original_ptr, global_wo_reg));
                        if (offsetof(metadata_chunk, v_data)) {
                            aux.push_back(make_add(upointer_type(),
                                validity_ptr_dst, validity_ptr_dst,
                                operand_t::make_iconstant(
                                offsetof(metadata_chunk, v_data))));
                        }
                        aux.push_back(make_selp(pointer_type(), valid_pred,
                            validity_ptr_dst, validity_ptr_dst,
                            global_wo_reg));

                        new_dst  = data_ptr;
                        new_vdst = validity_ptr_dst;

                        st_tmp[log_size(width) - 1u]++;
                        if (width <= 8) {
                            st_tmp[1] = 1;
                            st_tmp[2] = 1;
                        }
                        st_pred = 1u;
                        st_ptr  = 9u;

                        break;
                    case operand_constant:
                        aux.push_back(make_mov(upointer_type(), global_wo_reg,
                            global_wo));

                        aux.push_back(make_mov(pointer_type(), chidx,
                            ((static_cast<uintptr_t>(dst.offset) >>
                                lg_chunk_bytes) &
                             (max_chunks - 1u)) * sizeof(void *)));
                        aux.push_back(make_mov(pointer_type(), inidx,
                            (static_cast<uintptr_t>(dst.offset) &
                             chunk_size) >> 3u));
                        aux.push_back(make_ld(pointer_type(), global_space,
                            chunk, master));
                        aux.push_back(make_add(upointer_type(), chunk, chidx,
                            chunk));
                        aux.push_back(make_ld(pointer_type(), global_space,
                            chunk_ptr, chunk));
                        aux.push_back(make_ld(read_t, global_space,
                            a_data, a_data_ptr));
                        aux.push_back(make_setp(read_t, cmp_eq, valid_pred,
                            a_data, operand_t::make_iconstant((1 << width) - 1)));
                        aux.push_back(make_selp(pointer_type(), valid_pred,
                            data_ptr, dst, global_wo));
                        aux.push_back(make_add(upointer_type(),
                            validity_ptr_dst, chunk_ptr, inidx));
                        aux.push_back(make_selp(pointer_type(), valid_pred,
                            validity_ptr_dst, validity_ptr_dst, global_wo));

                        new_dst  = data_ptr;
                        new_vdst = validity_ptr;

                        st_tmp[log_size(width) - 1u]++;
                        st_pred = 1u;
                        st_ptr  = 9u;

                        break;
                    default:
                        assert(0 && "Unsupported operand type for store destination.");
                        break;
                }

                for (unsigned i = 0; i < 4u; i++) {
                    tmpu[i] = std::max(tmpu[i], st_tmp[i]);
                }
                tmp_pred = std::max(tmp_pred, st_pred);
                tmp_ptr  = std::max(tmp_ptr,  st_ptr);

                keep = false;

                statement_t new_st = statement;
                new_st.operands[0] = new_dst;

                statement_t new_vst = statement;
                new_vst.type = bitwise_type(new_vst.type);
                new_vst.operands[0] = new_vdst;
                operand_t & vsrc = new_vst.operands[1];
                assert(vsrc.op_type == operand_identifier);
                assert(vsrc.identifier.size() == vsrc.field.size());
                for (size_t i = 0; i < vsrc.identifier.size(); i++) {
                    vsrc.identifier[i] = make_validity_symbol(
                        vsrc.identifier[i]);
                }

                if (!(drop_store)) {
                    aux.push_back(new_st);
                    aux.push_back(new_vst);
                }

                break; }
            case op_trap: /* noop */ break;
            case op_or:
            case op_xor: {
                assert(statement.operands.size() == 3u);
                const operand_t & a = statement.operands[1];
                const operand_t & b = statement.operands[2];
                const operand_t & d = statement.operands[0];

                const operand_t va = make_validity_operand(a);
                const operand_t vb = make_validity_operand(b);
                const operand_t vd = make_validity_operand(d);

                switch (statement.type) {
                    case pred_type:
                        /**
                         * For simplicity, we assume the worst case and OR the
                         * validity bits without regard for the underlying data.
                         */
                        if (va.is_constant() && vb.is_constant()) {
                            aux.push_back(make_mov(b16_type, vd, va));
                        } else if (va.is_constant() && !(vb.is_constant())) {
                            aux.push_back(make_mov(b16_type, vd, vb));
                        } else if (!(va.is_constant()) && vb.is_constant()) {
                            aux.push_back(make_mov(b16_type, vd, va));
                        } else {
                            aux.push_back(make_or(b16_type, vd, va, vb));
                        }

                        assert(d.identifier.size() > 0);
                        inst->unchecked.insert(d.identifier[0]);
                        break;
                    case b16_type:
                    case b32_type:
                    case b64_type:
                               if (va.is_constant() && vb.is_constant()) {
                            aux.push_back(make_mov(statement.type, vd, va));
                        } else if (va.is_constant() && !(vb.is_constant())) {
                            aux.push_back(make_mov(statement.type, vd, vb));
                        } else if (!(va.is_constant()) && vb.is_constant()) {
                            aux.push_back(make_mov(statement.type, vd, va));
                        } else {
                            aux.push_back(make_or(statement.type, vd, va, vb));
                        }

                        break;
                    case f32_type:
                    case f64_type:
                    case s16_type:
                    case s32_type:
                    case s64_type:
                    case u16_type:
                    case u32_type:
                    case u64_type:
                    case s8_type:
                    case b8_type:
                    case u8_type:
                    case f16_type:
                    case texref_type:
                    case invalid_type:
                        assert(0 && "Unsupported type.");
                        break;
                }

                break; }
        }

        scope_t::block_vt::iterator tmp_it = it;
        scope_t::block_vt::iterator old_it = it;

        ++tmp_it;
        const size_t acount = aux.size();
        for (size_t i = 0; i < acount; i++) {
            block_t * b = new block_t();
            b->block_type   = block_statement;
            b->parent       = (*it);
            b->statement    = new statement_t(aux[i]);

            it = scope->blocks.insert(it, b);
            ++it;
        }

        if (!(keep)) {
            delete *old_it;
            scope->blocks.erase(old_it);
        }
        it = tmp_it;
    }

    for (unsigned i = 0; i < 4; i++) {
        if (tmpb[i] == 0) {
            continue;
        }

        variable_t v;

        v.space = reg_space;
        v.type = bitwise_of(1u << i);

        char sym[16];
        int ret = snprintf(sym, sizeof(sym), "__panoptes_b%d_",
            8u << i);
        assert(ret < (int) sizeof(sym));

        v.name = sym;
        v.has_suffix = true;
        v.suffix = tmpb[i];

        scope->variables.push_back(v);
    }

    for (unsigned i = 0; i < 4; i++) {
        if (tmps[i] == 0) {
            continue;
        }

        variable_t v;

        v.space = reg_space;
        v.type = signed_of(1u << i);

        char sym[16];
        int ret = snprintf(sym, sizeof(sym), "__panoptes_s%d_",
            8u << i);
        assert(ret < (int) sizeof(sym));

        v.name = sym;
        v.has_suffix = true;
        v.suffix = tmps[i];

        scope->variables.push_back(v);
    }

    for (unsigned i = 0; i < 4; i++) {
        if (tmpu[i] == 0) {
            continue;
        }

        variable_t v;

        v.space = reg_space;
        v.type = unsigned_of(8u << i);

        char sym[16];
        int ret = snprintf(sym, sizeof(sym), "__panoptes_u%d_",
            8u << i);
        assert(ret < (int) sizeof(sym));

        v.name = sym;
        v.has_suffix = true;
        v.suffix = tmpu[i];

        scope->variables.push_back(v);
    }

    if (tmp_pred > 0) {
        variable_t v;
        v.space = reg_space;
        v.type = pred_type;
        v.name = "__panoptes_pred";
        v.has_suffix = true;
        v.suffix = tmp_pred;
        scope->variables.push_back(v);
    }

    if (tmp_ptr > 0) {
        variable_t v;
        v.space = reg_space;
        v.type  = upointer_type();
        v.name = "__panoptes_ptr";
        v.has_suffix = true;
        v.suffix = tmp_ptr;
        scope->variables.push_back(v);
    }
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
         * CUDA segfaults here.
         */
        char msg[128];
        int ret = snprintf(msg, sizeof(msg),
            "cudaEventRecord called on invalid event.\n");
        // sizeof(msg) is small, so the cast is safe.
        assert(ret < (int) sizeof(msg) - 1);
        logger::instance().print(msg);

        raise(SIGSEGV);
        return cudaErrorInvalidValue;
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

            pret = snprintf(msg, sizeof(msg), "Texture most recently bound by:");
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

cudaError_t cuda_context_memcheck::cudaGetDeviceProperties(struct cudaDeviceProp *prop,
        int device) {
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
    prop_.unifiedAddressing = 0;
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

cudaError_t cuda_context_memcheck::remove_device_allocation(const void * device_ptr) {
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
        assert(0 && "The impossible happened.  Inconsistent upper/lower bounds");
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

    uintptr_t ptr = reinterpret_cast<uintptr_t>(device_ptr);

    const size_t chunk_bytes = 1 << lg_chunk_bytes;
    const size_t first_chunk =  ptr         >> lg_chunk_bytes;
    const size_t last_chunk  = (ptr + size) >> lg_chunk_bytes;
    bool master_dirty        = false;
    bool synchronized        = false;
    metadata_chunk ** const master = master_.host();

    for (size_t i = first_chunk; i <= last_chunk; i++) {
        /* The host is authoritative for addressability data */
        assert(chunks_[i] != default_chunk_);

        /* Decrement usage */
        assert(chunks_aux_[i].allocations > 0);
        chunks_aux_[i].allocations--;

        if (chunks_aux_[i].allocations == 0) {
            /* Free */

            /* There cannot be anything touching the device */
            if (!(synchronized)) {
                cudaError_t sret = callout::cudaDeviceSynchronize();
                if (sret != cudaSuccess) {
                    return sret;
                }

                synchronized = true;
            }

            assert(chunks_[i] != default_chunk_);
            pool_.free(chunks_[i]);
            chunks_[i]   = default_chunk_;
            master[i]    = default_chunk_->gpu();
            master_dirty = true;
        }

        /* Set addressability bits...
         *
         * Starting at (ptr       ) or  i      * chunk_bytes,     whichever is greater.
         * Ending at   (ptr + size) or (i + 1) * chunk_bytes - 1, whichever is less.
         */
        uintptr_t start = std::max(ptr,         i * chunk_bytes)     & (chunk_bytes - 1);
        uintptr_t end   = std::min(ptr + size - i * chunk_bytes, chunk_bytes);

        /* These form the ranges for applying memset to a_data.  They may differ from
         * start and end because the last byte on each end needs to be adjusted. */
        uintptr_t astart, aend;

        if ((start & (CHAR_BIT - 1)) != 0) {
            /* This write isn't a whole byte */
            uint8_t mask    = static_cast<uint8_t>(0xFF << (start & (CHAR_BIT - 1)));
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
            chunks_[i]->host()->a_data[start / CHAR_BIT] &=
                static_cast<uint8_t>(~mask);
        } else {
            astart = start;
        }

        if (end > astart && (end & (CHAR_BIT - 1)) != 0) {
            uint8_t mask = static_cast<uint8_t>(~(0xFF <<
                (end & (CHAR_BIT - 1))));
            chunks_[i]->host()->a_data[end / CHAR_BIT] &=
                static_cast<uint8_t>(~mask);
            aend = end & ~((uintptr_t) CHAR_BIT - 1);
        } else {
            aend = end;
        }

        if (aend > astart) {
            memset(static_cast<uint8_t *>(chunks_[i]->host()->a_data) +
                astart / CHAR_BIT, 0x0, (aend - astart) / CHAR_BIT);
        }

        /* Push to the device */
        cudaError_t tret = callout::cudaMemcpy(chunks_[i]->gpu()->a_data,
            chunks_[i]->host()->a_data, sizeof(chunks_[i]->host()->a_data),
            cudaMemcpyHostToDevice);
        if (tret != cudaSuccess) {
            return tret;
        }
    }

    /* If any chunks were freed, update the master list */
    if (master_dirty) {
        master_.to_gpu();
    }

    if (valgrind_ && must_free) {
        VALGRIND_FREELIKE_BLOCK(device_ptr, 0);
    }

    return cudaSuccess;
}

void cuda_context_memcheck::clear() {
    if (!(valgrind_)) {
        /* None of the operations below have any impact without Valgrind. */
        return;
    }

    for(amap_t::const_iterator it = device_allocations_.begin();
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
                ptr + size < reinterpret_cast<uintptr_t>(lit->first))) {
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

    /* The authoritative copy of master_ is always kept on the host, so we can
     * make any needed modifications and then push it.
     */
    const size_t chunk_bytes = 1 << lg_chunk_bytes;
    const size_t first_chunk =  ptr         >> lg_chunk_bytes;
    const size_t last_chunk  = (ptr + size) >> lg_chunk_bytes;
    bool master_dirty        = false;
    metadata_chunk ** const master = master_.host();

    /* bool existing_dirtied    = false; */

    for (size_t i = first_chunk; i <= last_chunk; i++) {
        if (chunks_[i] == default_chunk_) {
            /* Allocate another chunk */
            pool_t::handle_t * c = pool_.allocate();
            chunks_[i] = c;
            master [i] = c->gpu();
            master_dirty = true;
        } /* else {
            existing_dirtied = true;
        } */
    }

    /**
     * TODO:  Consult existing_dirtied flag here and conditionally download
     * the state of the buffers.  We upload whole blocks, but this can have
     * the adverse side effect of not downloading whole blocks when we freshly
     * allocate a chunk out of an existing block.
     */
    /* if (existing_dirtied) { */
        pool_.to_host(&chunks_[first_chunk], &chunks_[last_chunk + 1]);
    /* } */

    for (size_t i = first_chunk; i <= last_chunk; i++) {
        assert(chunks_[i] != default_chunk_);
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
            chunks_[i]->host()->a_data[start / CHAR_BIT] = mask;
        } else {
            astart = start;
        }

        if (end > astart && (end & (CHAR_BIT - 1)) != 0) {
            uint8_t mask = static_cast<uint8_t>(~(0xFF <<
                (end & (CHAR_BIT - 1))));
            chunks_[i]->host()->a_data[end / CHAR_BIT] = mask;
            aend = end & ~((uintptr_t) CHAR_BIT - 1);
        } else {
            aend = end;
        }

        if (aend > astart) {
            memset(static_cast<uint8_t *>(chunks_[i]->host()->a_data) +
                astart / CHAR_BIT, 0xFF, (aend - astart) / CHAR_BIT);
        }

        if (valgrind_) {
            uint8_t * vptr = reinterpret_cast<uint8_t *>(
                chunks_[i]->host()->v_data) + start;
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

    pool_.to_gpu(&chunks_[first_chunk], &chunks_[last_chunk + 1]);
    if (master_dirty) {
        master_.to_gpu();
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
        struct cudaExtent extent) : desc(_desc), x(0), y(0), z(0), validity(NULL) {
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

    cudaError_t ret = cuda_context::cudaMallocPitch(devPtr, pitch, width, height);
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
        } /* else: p >= lb */

        /**
         * Advance for length of containing block beyond p.
         */
        checked += it->second - (size_t)
            (static_cast<const uint8_t *>(p) -
             static_cast<const uint8_t *>(lb));
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

               if ( ddst &&  dsrc) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (!ddst &&  dsrc) {
            kind = cudaMemcpyDeviceToHost;
        } else if ( ddst && !dsrc) {
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
        if (check_access_device(dst, size) &&
                check_access_device(src, size)) {
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
        check_access_device(src, size);

        if (!(check_access_host(dst, size))) {
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
        check_access_device(dst, size);

        if (!(check_access_host(src, size))) {
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
             * CUDA 4.1 is quite adept at performing seemingly impossible memory
             * copies without segfault or error.  In an ideal world, we would use
             * our own memcpy here (as host-to-host predates CUDA) but we can't
             * perform impossible copies (NULL to NULL with size > 0 segfaults for
             * mere mortals).
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

cudaError_t cuda_context_memcheck::cudaMemset(void *devPtr, int value, size_t count) {
    if (count == 0) {
        // No-op
        return setLastError(cudaSuccess);
    } else if (!(check_access_device(devPtr, count))) {
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
    } else if (!(check_access_device(devPtr, count))) {
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

void global_context_memcheck::cudaRegisterVar(void **fatCubinHandle,
        char *hostVar, char *deviceAddress, const char *deviceName,
        int ext, int size, int constant, int global) {
    global_context::cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress,
        deviceName, ext, size, constant, global);

    /* Examine the variable definition and check whether it was initialized. */
    scoped_lock lock(mx_);
    variable_handle_t h(fatCubinHandle, deviceName);
    variable_definition_map_t::iterator it =
        variable_definitions_.find(h);
    if (it == variable_definitions_.end()) {
        /* Attempting to register a variable that does not exist.
         * This should have been caught at the cuda_context level, so return
         * quietly. */
        return;
    }

    it->second.hostVar = hostVar;
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

    *pStream = reinterpret_cast<cudaStream_t>(handle);
    return ret;
}

cudaError_t cuda_context_memcheck::cudaStreamDestroy(cudaStream_t stream) {
    void ** handle = reinterpret_cast<void **>(stream);

    scoped_lock lock(mx_);
    stream_map_t::iterator it = streams_.find(handle);
    if (it == streams_.end()) {
        /**
         * CUDA segfaults here.
         */
        raise(SIGSEGV);
        return cudaErrorInvalidResourceHandle;
    }

    internal::stream_t * stream_metadata = it->second;
    streams_.erase(it);

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
        /**
         * CUDA segfaults here.
         */
        raise(SIGSEGV);
        return cudaErrorInvalidResourceHandle;
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
        /**
         * CUDA segfaults here.
         */
        raise(SIGSEGV);
        return cudaErrorInvalidResourceHandle;
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
    const size_t first_chunk =  upptr        >> lg_chunk_bytes;
    const size_t last_chunk  = (upptr + len) >> lg_chunk_bytes;

    for (size_t i = first_chunk, voffset = 0; i <= last_chunk; i++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(chunks_[i] != default_chunk_);

        uintptr_t start = std::max(upptr,        i * chunk_bytes) & (chunk_bytes - 1);
        uintptr_t end   = std::min(upptr + len - i * chunk_bytes, chunk_bytes);
        assert(end >= start);
        uintptr_t diff  = end - start;

        /* Copy validity bits onto device from buffer */
        uint8_t * const vptr =
            reinterpret_cast<uint8_t *>(chunks_[i]->gpu()->v_data) + start;

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
    const size_t first_chunk =  upsrc        >> lg_chunk_bytes;
    const size_t last_chunk  = (upsrc + len) >> lg_chunk_bytes;

    for (size_t srcidx = first_chunk, voffset = 0; srcidx <= last_chunk; srcidx++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(chunks_[srcidx] != default_chunk_);

        uintptr_t sstart = std::max(upsrc,        srcidx * chunk_bytes) & (chunk_bytes - 1);
        uintptr_t send   = std::min(upsrc + len - srcidx * chunk_bytes, chunk_bytes);
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
        assert(chunks_[dstidx] != default_chunk_);
        const uintptr_t dstoff = updst & (chunk_bytes - 1);
        /* dstrem: number of bytes remaining in this chunk (dstidx) */
        const uintptr_t dstrem = chunk_bytes - dstoff;

        if (dstrem >= srcrem) {
            /* Single copy. */
            uint8_t * vdst =
                reinterpret_cast<uint8_t *>(chunks_[dstidx]->gpu()->v_data);
            uint8_t * vsrc =
                reinterpret_cast<uint8_t *>(chunks_[srcidx]->gpu()->v_data);

            callout::cudaMemcpyAsync(vdst + dstoff, vsrc + sstart, srcrem,
                cudaMemcpyDeviceToDevice, cs);
        } else {
            /* First copy (remainder of first destination chunk) */
            uint8_t * vdst =
                reinterpret_cast<uint8_t *>(chunks_[dstidx]->gpu()->v_data);
            uint8_t * vsrc =
                reinterpret_cast<uint8_t *>(chunks_[srcidx]->gpu()->v_data);

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
            assert(chunks_[nxtidx] != default_chunk_);
            callout::cudaMemcpyAsync(chunks_[nxtidx]->gpu()->v_data,
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
    }

    const uintptr_t upgpu    = reinterpret_cast<uintptr_t>(gpu);
    const size_t first_chunk =  upgpu        >> lg_chunk_bytes;
    const size_t last_chunk  = (upgpu + len) >> lg_chunk_bytes;

    /** TODO:  Pipeline */
    for (size_t i = first_chunk, voffset = 0; i <= last_chunk; i++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(chunks_[i] != default_chunk_);

        uintptr_t start = std::max(upgpu,        i * chunk_bytes) & (chunk_bytes - 1);
        uintptr_t end   = std::min(upgpu + len - i * chunk_bytes, chunk_bytes);
        assert(end >= start);
        uintptr_t diff  = end - start;

        /* Copy validity bits off device into buffer */
        uint8_t * const vptr =
            reinterpret_cast<uint8_t *>(chunks_[i]->gpu()->v_data) + start;

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
        unsigned set = VALGRIND_SET_VBITS(static_cast<uint8_t *>(host) + voffset,
            buffer, diff);
        if (set != 1) {
            assert(0 && "Valgrind failed to set validity bits.");
            return false;
        }

        voffset += diff;
    }

    {
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
    const size_t first_chunk =  upptr        >> lg_chunk_bytes;
    const size_t last_chunk  = (upptr + len) >> lg_chunk_bytes;

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

    for (size_t i = first_chunk, voffset = 0; i <= last_chunk; i++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(chunks_[i] != default_chunk_);

        uintptr_t start = std::max(upptr,        i * chunk_bytes) & (chunk_bytes - 1);
        uintptr_t end   = std::min(upptr + len - i * chunk_bytes, chunk_bytes);
        assert(end >= start);
        uintptr_t diff  = end - start;

        /* Write validity bits onto device */
        uint8_t * const chunk_ptr =
            reinterpret_cast<uint8_t *>(chunks_[i]->gpu()->v_data) + start;
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
    }

    const uintptr_t upgpu    = reinterpret_cast<uintptr_t>(gpu);
    const size_t first_chunk =  upgpu        >> lg_chunk_bytes;
    const size_t last_chunk  = (upgpu + len) >> lg_chunk_bytes;

    for (size_t i = first_chunk, voffset = 0; i <= last_chunk; i++) {
        /* Since we got this far, no chunk should point at the
         * default chunk */
        assert(chunks_[i] != default_chunk_);

        uintptr_t start = std::max(upgpu,        i * chunk_bytes) & (chunk_bytes - 1);
        uintptr_t end   = std::min(upgpu + len - i * chunk_bytes, chunk_bytes);
        assert(end >= start);
        uintptr_t diff  = end - start;

        /* Transfer bits out of Valgrind */
        const uint8_t * host_chunk = static_cast<const uint8_t *>(host) + voffset;
        (void) VALGRIND_GET_VBITS(host_chunk, buffer, diff);
        memset(static_cast<uint8_t *>(buffer) + chunk_bytes - diff, 0, chunk_bytes - diff);

        /* Copy validity bits onto device from buffer */
        uint8_t * const chunk_ptr =
            reinterpret_cast<uint8_t *>(chunks_[i]->gpu()->v_data) + start;

        cudaError_t ret;
        if (stream) {
            callout::cudaMemcpyAsync(chunk_ptr, buffer, end - start,
                cudaMemcpyHostToDevice, stream->stream);

            /**
             * We're going to overwrite buffer shortly (and/or deallocate it),
             * so a sync is required.
             */
            ret = stream->synchronize();
        } else {
            ret = callout::cudaMemcpy(chunk_ptr,
                buffer, end - start, cudaMemcpyHostToDevice);
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
        cudaError_t ret = callout::cudaFreeHost(buffer);
        if (ret != cudaSuccess) {
            return false;
        }
    }

    return true;
}

cudaError_t cuda_context_memcheck::cudaLaunch(const char *entry) {
    /**
     * If the entry name is invalid, fail.
     */
    if (entry == NULL) {
        return cudaErrorUnknown;
    }

    scoped_lock lock(mx_);
    internal::check_t * checker = new internal::check_t(this,
        get_entry_name(entry));
    /**
     * Go over call stack, pass in validity information and other auxillary information.
     */
    const char * entry_name = get_entry_name(entry);
    global_t::entry_info_map_t::const_iterator it;
    global_t * g = global();

    if (entry_name == NULL) {
        /* Try the now depreciated interpretation of entry as the entry name
         * itself. */
        it = g->entry_info_.find(entry);
    } else {
        it = g->entry_info_.find(entry_name);
    }

    if (it == g->entry_info_.end()) {
        delete checker;
        return setLastError(cudaErrorInvalidDeviceFunction);
    }

    if (call_stack().size() == 0) {
        /**
         * This isn't a specified return value in the CUDA 4.x documentation
         * for cudaLaunch, but this has been experimentally validated.
         */
        delete checker;
        return cudaErrorInvalidConfiguration;
    }

    const size_t pcount = it->second.user_params;
    size_t offset = it->second.user_param_size;

    internal::call_t * call = call_stack().top();

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

cudaError_t cuda_context_memcheck::cudaBindTexture(size_t *offset,
        const struct textureReference *texref, const void *devPtr,
        const struct cudaChannelFormatDesc *desc, size_t size) {
    cudaError_t ret = cuda_context::cudaBindTexture(
        offset, texref, devPtr, desc, size);
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

    it->second->bound_pointer   = devPtr;
    it->second->bound_size      = size;
    it->second->binding         = backtrace_t::instance();

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
            // TODO CHECK FOR RECENTLY FREED
            char msg[256];
            int pret = snprintf(msg, sizeof(msg), "Invalid device pointer "
                " of size %zu bound to texture.\n"
                " Address %p is not malloc'd or (recently) free'd.\n"
                " Possible host pointer?", size, devPtr);
            // sizeof(msg) is small, so the cast is safe.
            assert(pret < (int) sizeof(msg) - 1);
            logger::instance().print(msg);
        } else {
            // Invalid read starts at usrc + coffset
            //
            // Technically, we don't know whether the full read of (len - coffset)
            // additional bytes is invalid.
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

    return ret;
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

    if (internal_array) {
        internal_array->bindings.insert(texref);
    }

    return ret;
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
