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

#include <panoptes/global_context.h>
#include <panoptes/memcheck/global_memcheck_state.h>
#include <ptx_io/ptx_ir.h>

namespace panoptes {

/* Forward declarations. */
namespace internal {
    class auxillary_t;
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
    typedef std::vector<statement_t> statement_vt;

    void instrument_abs(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_add(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_and(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_atom(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_bar(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_bfe(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_bfi(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_bit1(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_brev(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_cnot(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_copysign(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_cvt(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_fp1(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_fp3(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_isspacep(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_ld(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_mad(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_math2(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_mov(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_neg(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_not(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_or(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_ret(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_prefetch(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_prmt(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_sad(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_selp(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_set(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_setp(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_shift(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_slct(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_st(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_testp(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_tex(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_tld4(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);
    void instrument_vote(const statement_t & statement,
        statement_vt * instrumentation, bool * keep,
        internal::auxillary_t * auxillary);

    friend class cuda_context_memcheck;
    friend struct internal::check_t;

    state_ptr_t state_;

    typedef std::pair<void **, std::string> variable_handle_t;
    struct variable_data_t {
        variable_t ptx;
        char * hostVar;
        const ptx_t * parent_ptx;
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
