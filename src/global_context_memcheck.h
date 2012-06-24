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

#ifndef __PANOPTES__GLOBAL_CONTEXT_MEMCHECK_H__
#define __PANOPTES__GLOBAL_CONTEXT_MEMCHECK_H__

#include "global_context.h"
#include "global_memcheck_state.h"
#include "ptx_ir.h"

namespace panoptes {

/* Forward declarations. */
namespace internal {
    struct check_t;
    struct instrumentation_t;
}

class global_context_memcheck : public global_context {
public:
    global_context_memcheck();
    ~global_context_memcheck();

    virtual cuda_context * factory(int device, unsigned int flags) const;

    virtual cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
    virtual cudaError_t cudaDeviceEnablePeerAccess(int peerDevice,
        unsigned int flags);
    virtual void cudaRegisterVar(void **fatCubinHandle,char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global);

    state_ptr_t state();
protected:
    /**
     * Instruments the given PTX program in-place for validity and
     * addressability checks.
     */
    virtual void instrument(void **fatCubinHandle, ptx_t * target);

    struct entry_info_t;

    void analyze_entry(function_t * entry);
    void instrument_entry(function_t * entry);
    void instrument_block(block_t * block, internal::instrumentation_t * inst,
        const entry_info_t & e);
private:
    friend class cuda_context_memcheck;
    friend struct internal::check_t;

    state_ptr_t state_;

    typedef std::pair<void **, std::string> variable_handle_t;
    struct variable_data_t {
        variable_t ptx;
        char * hostVar;
    };
    typedef boost::unordered_map<variable_handle_t, variable_data_t>
        variable_definition_map_t;
    variable_definition_map_t variable_definitions_;

    /**
     * Instrumentation information.
     */
    typedef std::set<std::string> string_set_t;
    string_set_t external_entries_;
    string_set_t nonentries_;
protected:
    struct entry_info_t {
        entry_info_t() : function(NULL), inst(NULL),
            fixed_shared_memory(0), local_memory(0) { }

        function_t * function;
        internal::instrumentation_t * inst;
        size_t user_params;
        size_t user_param_size;
        size_t fixed_shared_memory;
        size_t local_memory;
    };
private:
    typedef std::map<std::string, entry_info_t> entry_info_map_t;
    entry_info_map_t entry_info_;
};
}

#endif // __PANOPTES__GLOBAL_CONTEXT_MEMCHECK_H__