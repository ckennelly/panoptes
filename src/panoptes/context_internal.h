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

#ifndef __PANOPTES__CONTEXT_INTERNAL_H__
#define __PANOPTES__CONTEXT_INTERNAL_H__

#include <boost/unordered_map.hpp>
#include <boost/utility.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <ptx_io/ptx_ir.h>
#include <vector>

namespace panoptes {
namespace internal {

struct module_t : boost::noncopyable {
    module_t();
    ~module_t();

    bool     module_set;
    CUmodule module;
    ptx_t    ptx;

    struct function_t {
        CUfunction function;

        char *deviceFun;
        const char *deviceName;
        int thread_limit;
        uint3 *tid;
        uint3 *bid;
        dim3 *bDim;
        dim3 *gDim;
        int *wSize;
    };

    struct variable_t {
        char *hostVar;
        char *deviceAddress;
        const char *deviceName;
        int ext;
        int user_size;
        size_t size;
        int constant;
        int global;
    };

    struct texture_t : boost::noncopyable {
        texture_t() : has_texref(false), bound(false) { }

        ~texture_t() { }

        bool has_texref;
        CUtexref texref;
        const struct textureReference *hostVar;
        const void **deviceAddress;
        const char *deviceName;
        int dim;
        int norm;
        int ext;

        bool bound;
        size_t offset;
    };

    // Map host function pointers to function information
    typedef boost::unordered_map<const void *, function_t> function_map_t;
    function_map_t functions;

    // Map host variable names to variable information
    typedef boost::unordered_map<const void *, variable_t> variable_map_t;
    variable_map_t variables;

    // Map host texture names to texture information
    typedef boost::unordered_map<const struct textureReference *,
        texture_t *> texture_map_t;
    texture_map_t textures;

    bool handle_owned;
};

struct modules_t {
    ~modules_t() {
        const size_t n = modules.size();
        for (size_t i = 0; i < n; i++) {
            delete modules[i];
        }
    }

    typedef std::vector<module_t *> module_vt;
    module_vt modules;

    typedef boost::unordered_map<const void *,
        internal::module_t *> function_map_t;
    function_map_t functions;

    typedef boost::unordered_map<const void *, internal::module_t *>
        variable_map_t;
    variable_map_t variables;

    typedef boost::unordered_map<const struct textureReference *,
        internal::module_t *> texture_map_t;
    texture_map_t textures;
};

/**
 * Provides a container for copies of kernel arguments.  CUDA does not specify
 * whether we can repeatedly pass in the same pointer and get correct results.
 * Empirically, it appears that CUDA copies the argument.
 */
struct arg_t : boost::noncopyable {
    arg_t() : arg(NULL), size(0), offset(0) { }
    arg_t(const void * arg_, size_t size_, size_t offset_) :
            arg(new uint8_t[size_]), size(size_), offset(offset_) {
        memcpy(const_cast<void *>(arg), arg_, size);
    }

    ~arg_t() { delete[] static_cast<const uint8_t *>(arg); }

    const void * arg;
    const size_t size;
    const size_t offset;
};

struct call_t {
    call_t() { }
    ~call_t() {
        for (size_t i = 0; i < args.size(); i++) {
            delete args[i];
        }
    }

    dim3 gridDim;
    dim3 blockDim;
    uint32_t sharedMem;
    cudaStream_t stream;

    std::vector<arg_t *> args;
};

/**
 * For user-exposed handles, we return a specialized opaque type.
 */
void** create_handle();
void free_handle(void **);

} // end namespace internal
} // end namespace panoptes

#endif // __PANOPTES__CONTEXT_INTERNAL_H__
