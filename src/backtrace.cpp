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
#include <boost/thread.hpp>
#include <cassert>
#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <syscall.h>
#include <unistd.h>
#include <valgrind/memcheck.h>

using namespace panoptes;

namespace {
    pid_t gettid() {
        return static_cast<pid_t>(syscall(__NR_gettid));
    }
}

backtrace_t::backtrace_t() {
    refresh();
}

backtrace_t::backtrace_t(const backtrace_t & rhs) :
        bt_size_(rhs.bt_size_) {
    assert(bt_size_ <= max_backtrace);
    memcpy(bt_, rhs.bt_, static_cast<size_t>(bt_size_) * sizeof(*bt_));
}

backtrace_t & backtrace_t::operator=(const backtrace_t & rhs) {
    bt_size_    = rhs.bt_size_;
    assert(bt_size_ <= max_backtrace);
    memcpy(bt_, rhs.bt_, static_cast<size_t>(bt_size_) * sizeof(*bt_));

    return *this;
}

void backtrace_t::print() const {
    const int tid = static_cast<int>(gettid());

    /* This backtrace is styled much like Valgrind's.
     *
     * TODO:  Block out bits of backtrace internal to Panoptes.
     */
    char **bt_syms = backtrace_symbols(bt_, bt_size_);
    if (bt_syms) {
        /**
         * i = 1 is chosen to skip logger::print in the backtrace.
         */
        const int start  = 1;
        const char ats[] = "at";
        const char bys[] = "by";

        for (int i = start; i < bt_size_; i++) {
            /**
             * Attempt to pattern match for libc's provided components.  For
             * example:
             *
             * /usr/lib64/libgtest.so.0(_ZN7testing4Test3RunEv+0xaa) [0x7fc59066e3da]
             *
             * This strategy will break if some parenthesis show up in the
             * filename. Parsing from the other direction might help?
             */
            char * mangled_start = NULL;
            char * offset_mark   = NULL;
            char * mangled_end   = NULL;
            char * address_start = NULL;
            char * address_end   = NULL;

            for (char * c = bt_syms[i]; *c; c++) {
                      if (!(mangled_start) && *c == '(') {
                    mangled_start = c;
               } else if (mangled_start  && *c == '+') {
                    offset_mark   = c;
               } else if (mangled_start  && *c == ')') {
                    mangled_end   = c;
               } else if (mangled_end    && *c == '[') {
                    address_start = c;
               } else if (address_start  && *c == ']') {
                    address_end   = c;
               }
            }

            const char * attr = (i == start) ? ats : bys;

            if (address_end) {
                /* The others except possibly offset_mark being non-NULL
                 * follows from this. */
                const char * filename = bt_syms[i];
                *mangled_start = '\0';  // Chop all characters starting at the (

                const char * name     = mangled_start + 1;
                const char * offset;
                if (offset_mark) {
                    /* Chop all characters starting at the + */
                    *offset_mark      = '\0';
                    offset            = offset_mark + 1;
                } else {
                   /* This will end up being an empty string after the next
                    * chop */
                   offset             = mangled_end;
                }

                *mangled_end = '\0';  // Chop all characters starting at the )

                /* Check up on the address as a way of validating our parsing.
                 *
                 * 4 is a very, very crude upperbound for log_{2}{10}.
                 */
                #ifndef NDEBUG
                const char * address  = address_start + 1;
                *address_end = '\0';  // Chop the ]

                char tmp[sizeof(bt_[i]) * 4];
                const int tmp_ret = snprintf(tmp, sizeof(tmp), "%p", bt_[i]);
                assert(tmp_ret < (int) (sizeof(tmp) - 1));
                assert(strcmp(tmp, address) == 0);
                #endif

                int status;
                char * demangled = abi::__cxa_demangle(name /* mangled name */,
                    NULL /* output_buffer */, NULL /* length */, &status);
                if (status == 0) {
                    fprintf(stderr, "==%d== %s %p: %s+%s (%s)\n", tid, attr,
                        bt_[i], demangled, offset, filename);
                    free(demangled);
                } else if (*offset) {
                    fprintf(stderr, "==%d== %s %p: %s+%s (%s)\n", tid, attr,
                        bt_[i], name,      offset, filename);
                } else {
                    fprintf(stderr, "==%d== %s %p: %s\n", tid, attr,
                        bt_[i], filename);
                }
            } else {
                // Unable to parse
                fprintf(stderr, "==%d== %s %s\n", tid, attr, bt_syms[i]);
            }
        }

        fprintf(stderr, "==%d==\n", tid);
    }

    free(bt_syms);
}

void backtrace_t::refresh() {
    (void) VALGRIND_MAKE_MEM_UNDEFINED(bt_, sizeof(bt_));
    bt_size_ = backtrace(bt_, max_backtrace);
}

backtrace_t & backtrace_t::instance() {
    static boost::thread_specific_ptr<backtrace_t> instances;
    if (instances.get() == NULL) {
        instances.reset(new backtrace_t());
    }

    return *instances;
}
