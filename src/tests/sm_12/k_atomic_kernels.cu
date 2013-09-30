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

#include "vtest_k_atomic.h"

template<typename OP>
static __device__ uint32_t atomic_global(uint32_t * a, uint32_t b) {
    BOOST_STATIC_ASSERT(sizeof(OP) == 0);
    return 0;
}

template<>
static __device__ uint32_t atomic_global<atom_add>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.global.add.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_global<atom_and>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.global.and.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_global<atom_dec>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.global.dec.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_global<atom_exch>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.global.exch.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_global<atom_inc>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.global.inc.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_global<atom_max>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.global.max.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_global<atom_min>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.global.min.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_global<atom_or>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.global.or.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_global<atom_xor>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.global.xor.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<typename OP>
static __device__ uint32_t atomic_global_const1(uint32_t * a) {
    BOOST_STATIC_ASSERT(sizeof(OP) == 0);
    return 0;
}

template<>
static __device__ uint32_t atomic_global_const1<atom_add>(uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.add.u32 %0, [%1], 1;" : "=r"(out) : "l"(a));
    return out;
}

template<>
static __device__ uint32_t atomic_global_const1<atom_and>(uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.and.b32 %0, [%1], 1;" : "=r"(out) : "l"(a));
    return out;
}

template<>
static __device__ uint32_t atomic_global_const1<atom_dec>(uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.dec.u32 %0, [%1], 1;" : "=r"(out) : "l"(a));
    return out;
}

template<>
static __device__ uint32_t atomic_global_const1<atom_exch>(uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.exch.b32 %0, [%1], 1;" : "=r"(out) : "l"(a));
    return out;
}

template<>
static __device__ uint32_t atomic_global_const1<atom_inc>(uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.inc.u32 %0, [%1], 1;" : "=r"(out) : "l"(a));
    return out;
}

template<>
static __device__ uint32_t atomic_global_const1<atom_max>(uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.max.u32 %0, [%1], 1;" : "=r"(out) : "l"(a));
    return out;
}

template<>
static __device__ uint32_t atomic_global_const1<atom_min>(uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.min.u32 %0, [%1], 1;" : "=r"(out) : "l"(a));
    return out;
}

template<>
static __device__ uint32_t atomic_global_const1<atom_or>(uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.or.b32 %0, [%1], 1;" : "=r"(out) : "l"(a));
    return out;
}

template<>
static __device__ uint32_t atomic_global_const1<atom_xor>(uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.xor.b32 %0, [%1], 1;" : "=r"(out) : "l"(a));
    return out;
}

template<typename OP>
static __device__ uint32_t atomic_shared(uint32_t * a, uint32_t b) {
    BOOST_STATIC_ASSERT(sizeof(OP) == 0);
    return 0;
}

template<>
static __device__ uint32_t atomic_shared<atom_add>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.shared.add.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_shared<atom_and>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.shared.and.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_shared<atom_dec>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.shared.dec.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_shared<atom_exch>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.shared.exch.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_shared<atom_inc>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.shared.inc.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_shared<atom_max>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.shared.max.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_shared<atom_min>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.shared.min.u32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_shared<atom_or>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.shared.or.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<>
static __device__ uint32_t atomic_shared<atom_xor>(uint32_t * a, uint32_t b) {
    uint32_t out;
    asm volatile("atom.shared.xor.b32 %0, [%1], %2;" :
        "=r"(out) : "l"(a), "r"(b));
    return out;
}

template<typename OP>
static __global__ void k_ag(uint32_t * d, uint32_t * a, uint32_t b) {
    d[threadIdx.x] = atomic_global<OP>(a, b);
}

template<typename OP>
static __global__ void k_ag_const1(uint32_t * d, uint32_t * a) {
    d[threadIdx.x] = atomic_global_const1<OP>(a);
}

static __global__ void k_ag_cas(uint32_t * d, uint32_t * a, uint32_t b,
        uint32_t c) {
    uint32_t out;
    asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" :
        "=r"(out) : "l"(a), "r"(b), "r"(c));
    d[threadIdx.x] = out;
}

static __global__ void k_ag_cas_const1(uint32_t * d, uint32_t * a, uint32_t c) {
    uint32_t out;
    asm volatile("atom.global.cas.b32 %0, [%1], 1, %2;" :
        "=r"(out) : "l"(a), "r"(c));
    d[threadIdx.x] = out;
}

static __global__ void k_ag_cas_const2(uint32_t * d, uint32_t * a) {
    uint32_t out;
    asm volatile("atom.global.cas.b32 %0, [%1], 1, 5;" : "=r"(out) : "l"(a));
    d[threadIdx.x] = out;
}

template<typename OP>
static __global__ void k_as(uint32_t * d, uint32_t * a, uint32_t b) {
    __shared__ uint32_t l[1];
    l[0] = *a;
    __syncthreads();
    d[threadIdx.x] = atomic_shared<OP>(l, b);
    __syncthreads();
    *a = l[0];
}

static __global__ void k_as_cas(uint32_t * d, uint32_t * a, uint32_t b,
        uint32_t c) {
    __shared__ uint32_t l[1];
    l[0] = *a;
    __syncthreads();
    uint32_t out;
    asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" :
        "=r"(out) : "l"(l), "r"(b), "r"(c));
    __syncthreads();
    *a = l[0];
    d[threadIdx.x] = out;
}

template<typename T>
bool launch_atomic_global(int threads, uint32_t * d, uint32_t * a,
        uint32_t b) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return false;
    }

    k_ag<T><<<1, threads, 0, stream>>>(d, a, b);

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

bool launch_atomic_global_cas(int threads, uint32_t * d, uint32_t * a,
        uint32_t b, uint32_t c) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return false;
    }

    k_ag_cas<<<1, threads, 0, stream>>>(d, a, b, c);

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

bool launch_atomic_global_cas_const1(int threads, uint32_t * d, uint32_t * a,
        uint32_t * b, uint32_t c) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return false;
    }

    k_ag_cas_const1<<<1, threads, 0, stream>>>(d, a, c);
    *b = 1u;

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

bool launch_atomic_global_cas_const2(int threads, uint32_t * d, uint32_t * a,
        uint32_t * b, uint32_t * c) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return false;
    }

    k_ag_cas_const2<<<1, threads, 0, stream>>>(d, a);
    *b = 1;
    *c = 5;

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

template<typename T>
bool launch_atomic_shared(int threads, uint32_t * d, uint32_t * a,
        uint32_t b) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return false;
    }

    k_as<T><<<1, threads, 0, stream>>>(d, a, b);

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

bool launch_atomic_shared_cas(int threads, uint32_t * d, uint32_t * a,
        uint32_t b, uint32_t c) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return false;
    }

    k_as_cas<<<1, threads, 0, stream>>>(d, a, b, c);

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

template<typename T>
bool launch_atomic_global_const1(int threads, uint32_t * d, uint32_t * a) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return false;
    }

    k_ag_const1<T><<<1, threads, 0, stream>>>(d, a);

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

/**
 * Instantiate templates.
 */
template bool launch_atomic_global<atom_add> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_global<atom_and> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_global<atom_dec> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_global<atom_exch>(int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_global<atom_inc> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_global<atom_max> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_global<atom_min> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_global<atom_or>  (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_global<atom_xor> (int, uint32_t *, uint32_t *, uint32_t);

template bool launch_atomic_global_const1<atom_add> (int, uint32_t *, uint32_t *);
template bool launch_atomic_global_const1<atom_and> (int, uint32_t *, uint32_t *);
template bool launch_atomic_global_const1<atom_dec> (int, uint32_t *, uint32_t *);
template bool launch_atomic_global_const1<atom_exch>(int, uint32_t *, uint32_t *);
template bool launch_atomic_global_const1<atom_inc> (int, uint32_t *, uint32_t *);
template bool launch_atomic_global_const1<atom_max> (int, uint32_t *, uint32_t *);
template bool launch_atomic_global_const1<atom_min> (int, uint32_t *, uint32_t *);
template bool launch_atomic_global_const1<atom_or>  (int, uint32_t *, uint32_t *);
template bool launch_atomic_global_const1<atom_xor> (int, uint32_t *, uint32_t *);

template bool launch_atomic_shared<atom_add> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_shared<atom_and> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_shared<atom_dec> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_shared<atom_exch>(int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_shared<atom_inc> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_shared<atom_max> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_shared<atom_min> (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_shared<atom_or>  (int, uint32_t *, uint32_t *, uint32_t);
template bool launch_atomic_shared<atom_xor> (int, uint32_t *, uint32_t *, uint32_t);

static __global__ void k_global_overrun(uint32_t * a) {
    asm volatile("red.global.xor.b32 [%0], 1;\n" : : "l"(a));
}

cudaError_t launch_red_global_overrun(uint32_t * a) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return ret;
    }

    k_global_overrun<<<1, 1, 0, stream>>>(a);

    ret = cudaStreamSynchronize(stream);
    if (ret != cudaSuccess) {
        return ret;
    }

    return cudaStreamDestroy(stream);
}

static __global__ void k_shared_overrun(uint32_t offset) {
    __shared__ uint32_t s[1];
    asm volatile("red.shared.xor.b32 [%0], 1;\n" : : "l"(s + offset));
}

cudaError_t launch_red_shared_overrun(uint32_t offset) {
    cudaError_t ret;
    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    if (ret != cudaSuccess) {
        return ret;
    }

    k_shared_overrun<<<1, 1, 0, stream>>>(offset);

    ret = cudaStreamSynchronize(stream);
    if (ret != cudaSuccess) {
        return ret;
    }

    return cudaStreamDestroy(stream);
}
