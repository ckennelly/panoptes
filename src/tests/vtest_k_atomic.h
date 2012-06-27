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

#ifndef __PANOPTES__TESTS__VTEST_K_ATOMIC_H__
#define __PANOPTES__TESTS__VTEST_K_ATOMIC_H__

#include <algorithm>
#include <stdint.h>

/**
 * For all atomic operations, the () operator computes the result of the
 * operation that will be stored in the target address.
 */
struct atom_add {
    template<typename T>
    T operator()(T r, T s) {
        return r + s;
    }
};

struct atom_and {
    template<typename T>
    T operator()(T r, T s) {
        return r & s;
    }
};

struct atom_dec {
    template<typename T>
    T operator()(T r, T s) {
        if (r == 0 || r > s) {
            return s;
        } else {
            return r - 1;
        }
    }
};

struct atom_exch {
    template<typename T>
    T operator()(T r, T s) {
        (void) r;
        return s;
    }
};

struct atom_inc {
    template<typename T>
    T operator()(T r, T s) {
        if (r >= s) {
            return 0;
        } else {
            return r + 1;
        }
    }
};

struct atom_max {
    template<typename T>
    T operator()(T r, T s) {
        return std::max(r, s);
    }
};

struct atom_min {
    template<typename T>
    T operator()(T r, T s) {
        return std::min(r, s);
    }
};

struct atom_or {
    template<typename T>
    T operator()(T r, T s) {
        return r | s;
    }
};

struct atom_xor {
    template<typename T>
    T operator()(T r, T s) {
        return r ^ s;
    }
};

template<typename T>
bool launch_atomic_global(int threads, uint32_t * d, uint32_t * a, uint32_t b);

bool launch_atomic_global_cas(int threads, uint32_t * d, uint32_t * a,
    uint32_t b, uint32_t c);

cudaError_t launch_red_global_overrun(uint32_t * a);

template<typename T>
bool launch_atomic_shared(int threads, uint32_t * d, uint32_t * a, uint32_t b);

bool launch_atomic_shared_cas(int threads, uint32_t * d, uint32_t * a,
    uint32_t b, uint32_t c);

cudaError_t launch_red_shared_overrun(uint32_t offset);

#endif // __PANOPTES__TESTS__VTEST_K_ATOMIC_H__
