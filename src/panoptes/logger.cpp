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
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include "logger.h"
#include <string.h>
#include <syscall.h>
#include <unistd.h>

using namespace panoptes;

namespace {
    pid_t gettid() {
        return static_cast<pid_t>(syscall(__NR_gettid));
    }
}

logger & logger::instance() {
    static logger inst;
    return inst;
}

logger::logger() : stack_depth(20) { }

void logger::print(const char * msg) {
    print(msg, backtrace_t::instance());
}

void logger::print(const char * msg, const backtrace_t & bt) {
    const int tid = static_cast<int>(gettid());

    const char * cur = msg;
    while (true) {
        const char * next = strchr(cur, '\n');

        if (next == NULL) {
            // Print remainder of string
            fprintf(stderr, "==%d== %s\n==%d==\n", tid, cur, tid);
            break;
        } else {
            int length = static_cast<int>(next - cur);
            assert(length >= 0);

            fprintf(stderr, "==%d== %.*s\n",       tid, length, cur);

            cur = next + 1;
        }
    }

    bt.print();
}
