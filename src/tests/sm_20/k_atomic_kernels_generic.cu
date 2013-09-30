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

#include "vtest_k_atomic_generic.h"

template<typename OP>
static __device__ uint32_t atomic_generic(uint32_t * a, uint32_t b) {
    BOOST_STATIC_ASSERT(sizeof(OP) == 0);
    return 0;
}

template<>
static __device__ uint32_t atomic_generic<atom_add>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.add.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_generic<atom_and>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.and.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_generic<atom_dec>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.dec.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_generic<atom_exch>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.exch.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_generic<atom_inc>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.inc.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_generic<atom_max>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.max.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_generic<atom_min>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.min.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_generic<atom_or>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.or.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_generic<atom_xor>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.xor.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<typename OP>
static __global__ void k_generic(uint32_t * d, uint32_t * a, uint32_t b) {
    d[threadIdx.x] = atomic_generic<OP>(a, b);
}

static __global__ void k_generic_cas(uint32_t * d, uint32_t * a, uint32_t b,
        uint32_t c) {
    uint32_t out;
    asm volatile("atom.cas.b32 %0, [%1], %2, %3;" :
        "=r"(out) : "l"(a), "r"(b), "r"(c));
    d[threadIdx.x] = out;
}

template<typename T>
bool launch_atomic_generic(int threads, uint32_t * d, uint32_t * a,
        uint32_t b) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return false;
    }

    k_generic<T><<<1, threads, 0, stream>>>(d, a, b);

    ret = cudaStreamSynchronize(stream);
    if (ret != cudaSuccess) {
        return false;
    }

    ret = cudaStreamDestroy(stream);
    if (ret != cudaSuccess) {
        return false;
    }

    return true;
}

bool launch_atomic_generic_cas(int threads, uint32_t * d, uint32_t * a,
        uint32_t b, uint32_t c) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return false;
    }

    k_generic_cas<<<1, threads, 0, stream>>>(d, a, b, c);

    ret = cudaStreamSynchronize(stream);
    if (ret != cudaSuccess) {
        return false;
    }

    ret = cudaStreamDestroy(stream);
    if (ret != cudaSuccess) {
        return false;
    }

    return true;
}

template bool launch_atomic_generic<atom_add> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_generic<atom_and> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_generic<atom_dec> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_generic<atom_exch>(int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_generic<atom_inc> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_generic<atom_max> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_generic<atom_min> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_generic<atom_or>  (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_generic<atom_xor> (int, uint32_t *, uint32_t *, uint32_t);

static __global__ void k_generic_overrun(uint32_t * a) {
    asm volatile("red.xor.b32 [%0], 1;\n" : : "l"(a));
}

cudaError_t launch_red_generic_overrun(uint32_t * a) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return ret;
    }

    k_generic_overrun<<<1, 1, 0, stream>>>(a);

    ret = cudaStreamSynchronize(stream);
    if (ret != cudaSuccess) {
        return ret;
    }

    return cudaStreamDestroy(stream);
}
