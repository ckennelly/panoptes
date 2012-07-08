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

#include <boost/static_assert.hpp>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

/**
 * nvcc has issues with compiling the extensive bits of C++ used by Google
 * Test.  As a result, kernels are compiled separately and launched through
 * a plain interface that can be exposed to g++.
 */
#include "vtest_k_atomic.h"

using testing::Types;

typedef Types<atom_add, atom_and, atom_dec, atom_exch, atom_inc,
    atom_min, atom_max, atom_or, atom_xor> AtomicTypes;

template<typename T>
class AtomicGlobalFixture : public ::testing::Test {
public:
    typedef T _TypeParam;
protected:
    AtomicGlobalFixture() { }
    ~AtomicGlobalFixture() { }
};

TYPED_TEST_CASE(AtomicGlobalFixture, AtomicTypes);

TYPED_TEST(AtomicGlobalFixture, NoInitialization) {
    cudaError_t ret;

    uint32_t * a;
    ret = cudaMalloc((void **) &a, sizeof(*a));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t * d;
    ret = cudaMalloc((void **) &d, 2 * sizeof(*d));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t b = 0xDEADBEEF;
    VALGRIND_MAKE_MEM_UNDEFINED(&b, sizeof(b));

    const bool bret = launch_atomic_global<TypeParam>(2, d, a, b);
    EXPECT_TRUE(bret);

    uint32_t ha, hd[2];

    ret = cudaMemcpy(&ha, a, sizeof(ha), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(d);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(a);
    EXPECT_EQ(cudaSuccess, ret);

    /* We do not expect anything except for ha and hd to be uninitialized. */
    uint32_t va, vd[2];
    const bool vret = (VALGRIND_GET_VBITS(&ha, &va, sizeof(ha)) == 1) &&
                      (VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd)) == 1);
    if (vret) {
        EXPECT_EQ(0xFFFFFFFF, va);
        EXPECT_EQ(0xFFFFFFFF, vd[0]);
        EXPECT_EQ(0xFFFFFFFF, vd[1]);
    }
}

TYPED_TEST(AtomicGlobalFixture, MixInInitialized) {
    cudaError_t ret;

    uint32_t * a;
    ret = cudaMalloc((void **) &a, sizeof(*a));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t * d;
    ret = cudaMalloc((void **) &d, 2 * sizeof(*d));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t b = 0xDEADBEEF;

    const bool bret = launch_atomic_global<TypeParam>(2, d, a, b);
    EXPECT_TRUE(bret);

    uint32_t ha, hd[2];

    ret = cudaMemcpy(&ha, a, sizeof(ha), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(d);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(a);
    EXPECT_EQ(cudaSuccess, ret);

    /* We do not expect anything except for ha and hd to be uninitialized. */
    uint32_t va, vd[2];
    const bool vret = (VALGRIND_GET_VBITS(&ha, &va, sizeof(ha)) == 1) &&
                      (VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd)) == 1);
    if (vret) {
        EXPECT_EQ(0xFFFFFFFF, va);
        EXPECT_EQ(0xFFFFFFFF, vd[0]);
        EXPECT_EQ(0xFFFFFFFF, vd[1]);
    }
}

TYPED_TEST(AtomicGlobalFixture, Initialized) {
    cudaError_t ret;

    uint32_t * a;
    ret = cudaMalloc((void **) &a, sizeof(*a));
    EXPECT_EQ(cudaSuccess, ret);

    const uint32_t a_init = 0;
    ret = cudaMemcpy(a, &a_init, sizeof(*a), cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t * d;
    ret = cudaMalloc((void **) &d, 2 * sizeof(*d));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t b = 0xDEADBEEF;

    const bool bret = launch_atomic_global<TypeParam>(2, d, a, b);
    EXPECT_TRUE(bret);

    uint32_t ha, hd[2];

    ret = cudaMemcpy(&ha, a, sizeof(ha), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    /**
     * Since a and b were initalized, we can reason about the values of *a
     * and *d.
     */
    TypeParam op;
    const uint32_t expected0 = op(a_init, b);
    const uint32_t expected1 = op(expected0, b);

    EXPECT_EQ(expected1, ha);

    uint32_t expected[2] = {a_init, expected0};
    /* We are not particularly concerned with the order that the values come
     * out. */
    std::sort(hd, hd + 1);
    std::sort(expected, expected + 1);

    EXPECT_EQ(expected[0], hd[0]);
    EXPECT_EQ(expected[1], hd[1]);

    ret = cudaFree(d);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(a);
    EXPECT_EQ(cudaSuccess, ret);

    /* We do not expect anything except for ha and hd to be uninitialized. */
    uint32_t va, vd[2];
    const bool vret = (VALGRIND_GET_VBITS(&ha, &va, sizeof(ha)) == 1) &&
                      (VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd)) == 1);
    if (vret) {
        EXPECT_EQ(0x0, va);
        EXPECT_EQ(0x0, vd[0]);
        EXPECT_EQ(0x0, vd[1]);
    }
}

TYPED_TEST(AtomicGlobalFixture, AddressInitialized) {
    cudaError_t ret;

    uint32_t * a;
    ret = cudaMalloc((void **) &a, sizeof(*a));
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemset(a, 0, sizeof(*a));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t * d;
    ret = cudaMalloc((void **) &d, 2 * sizeof(*d));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t b = 0xDEADBEEF;
    VALGRIND_MAKE_MEM_UNDEFINED(&b, sizeof(b));

    const bool bret = launch_atomic_global<TypeParam>(2, d, a, b);
    EXPECT_TRUE(bret);

    uint32_t ha, hd[2];

    ret = cudaMemcpy(&ha, a, sizeof(ha), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(d);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(a);
    EXPECT_EQ(cudaSuccess, ret);

    /* We do not expect anything except for ha and hd to be uninitialized. */
    uint32_t va, vd[2];
    const bool vret = (VALGRIND_GET_VBITS(&ha, &va, sizeof(ha)) == 1) &&
                      (VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd)) == 1);
    if (vret) {
        EXPECT_EQ(0xFFFFFFFF, va);

        std::sort(vd, vd + 1);
        EXPECT_EQ(0x0,        vd[0]);
        EXPECT_EQ(0xFFFFFFFF, vd[1]);
    }
}

TYPED_TEST(AtomicGlobalFixture, ConstantArgument) {
    cudaError_t ret;

    uint32_t * a;
    ret = cudaMalloc((void **) &a, sizeof(*a));
    EXPECT_EQ(cudaSuccess, ret);

    const uint32_t a_init = 0;
    ret = cudaMemcpy(a, &a_init, sizeof(a_init), cudaMemcpyHostToDevice);
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t * a_invalid;
    ret = cudaMalloc((void **) &a_invalid, sizeof(*a_invalid));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t * d;
    ret = cudaMalloc((void **) &d, 2 * sizeof(*d));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t * d_invalid;
    ret = cudaMalloc((void **) &d_invalid, 2 * sizeof(*d_invalid));
    EXPECT_EQ(cudaSuccess, ret);

    bool bret = launch_atomic_global_const1<TypeParam>(2, d, a);
    EXPECT_TRUE(bret);

    bret = launch_atomic_global_const1<TypeParam>(2, d_invalid, a_invalid);
    EXPECT_TRUE(bret);

    uint32_t ha, hd[2];

    ret = cudaMemcpy(&ha, a, sizeof(ha), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(d);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(a);
    EXPECT_EQ(cudaSuccess, ret);

    TypeParam op;
    const uint32_t expected0 = op(a_init, 1u);
    const uint32_t expected1 = op(expected0, 1u);

    EXPECT_EQ(expected1, ha);

    uint32_t expected[2] = {a_init, expected0};
    /* We are not particularly concerned with the order that the values come
     * out. */
    std::sort(hd, hd + 1);
    std::sort(expected, expected + 1);

    EXPECT_EQ(expected[0], hd[0]);
    EXPECT_EQ(expected[1], hd[1]);

    uint32_t va, vd[2];
    const bool vret = (VALGRIND_GET_VBITS(&ha, &va, sizeof(ha)) == 1) &&
                      (VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd)) == 1);
    if (vret) {
        EXPECT_EQ(0x0, va);

        std::sort(vd, vd + 1);
        EXPECT_EQ(0x0, vd[0]);
        EXPECT_EQ(0x0, vd[1]);
    }

    ret = cudaMemcpy(&ha, a_invalid, sizeof(ha), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&hd, d_invalid, sizeof(hd), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(d_invalid);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(a_invalid);
    EXPECT_EQ(cudaSuccess, ret);

    /* These values will be uninitialized. */
    if (vret) {
        bool vret2 = (VALGRIND_GET_VBITS(&ha, &va, sizeof(ha)) == 1) &&
                     (VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd)) == 1);
        EXPECT_TRUE(vret2);

        EXPECT_EQ(0xFFFFFFFF, va);
        EXPECT_EQ(0xFFFFFFFF, vd[0]);
        EXPECT_EQ(0xFFFFFFFF, vd[1]);
    }
}

/**
 * param & 0x1 => initialize target address
 * param & 0x2 => initialize b
 * param & 0x4 => initialize c
 */
class AtomicGlobalFixtureCAS : public ::testing::TestWithParam<unsigned> {
public:
    AtomicGlobalFixtureCAS() { }
    ~AtomicGlobalFixtureCAS() { }
};

TEST_P(AtomicGlobalFixtureCAS, DoTest) {
    const unsigned p = GetParam();

    cudaError_t ret;

    uint32_t * a;
    ret = cudaMalloc((void **) &a, sizeof(*a));
    EXPECT_EQ(cudaSuccess, ret);

    const uint32_t a_init = 0;
    if (p & 0x1) {
        ret = cudaMemcpy(a, &a_init, sizeof(*a), cudaMemcpyHostToDevice);
        EXPECT_EQ(cudaSuccess, ret);
    }

    uint32_t * d;
    ret = cudaMalloc((void **) &d, 2 * sizeof(*d));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t b = 0x00000000;
    if (!(p & 0x2)) {
        VALGRIND_MAKE_MEM_UNDEFINED(&b, sizeof(b));
    }

    uint32_t c = 0xDEADBEEF;
    if (!(p & 0x4)) {
        VALGRIND_MAKE_MEM_UNDEFINED(&c, sizeof(c) / 2);
    }

    const bool bret = launch_atomic_global_cas(2, d, a, b, c);
    EXPECT_TRUE(bret);

    uint32_t ha, hd[2];

    ret = cudaMemcpy(&ha, a, sizeof(ha), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(d);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(a);
    EXPECT_EQ(cudaSuccess, ret);

    /* We do not expect anything except for ha and hd to be uninitialized. */
    uint32_t va, vd[2];
    const bool vret = (VALGRIND_GET_VBITS(&ha, &va, sizeof(ha)) == 1) &&
                      (VALGRIND_GET_VBITS(&hd, &vd, sizeof(hd)) == 1);
    if (vret) {
        uint32_t eva, evd0, evd1;
        switch (p) {
            case 0:
            case 2:
            case 4:
            case 6:
                eva = evd0 = evd1 = 0xFFFFFFFF;
                break;
            case 1:
            case 5:
                /* The first thread to encounter the address will see a valid
                 * value.  The second will see the invalid value left by the
                 * first. */
                evd0 = 0x0;
                eva = evd1 = 0xFFFFFFFF;
                break;
            case 3:
                /**
                 * Similar to cases 1 and 5.
                 */
                evd0 = 0x0;
                eva = evd1 = 0x0000FFFF;
                break;
            case 7:
                eva = evd0 = evd1 = 0x0;
                break;
        }

        EXPECT_EQ(eva, va);

        std::sort(vd, vd + 1);
        EXPECT_EQ(evd0, vd[0]);
        EXPECT_EQ(evd1, vd[1]);

        VALGRIND_MAKE_MEM_DEFINED(&ha, sizeof(ha));
        VALGRIND_MAKE_MEM_DEFINED(hd, sizeof(hd));
    }

    if (p & 1) {
        if (!(p & 0x4)) {
            VALGRIND_MAKE_MEM_DEFINED(&c, sizeof(c));
        }

        std::sort(hd, hd + 1);

        EXPECT_EQ(a_init, hd[0]);
        EXPECT_EQ(c, hd[1]);
    }
}

TEST_P(AtomicGlobalFixtureCAS, ConstantArguments) {
    const unsigned p = GetParam();

    cudaError_t ret;

    uint32_t *a0, *a1;
    ret = cudaMalloc((void **) &a0, sizeof(*a0));
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &a1, sizeof(*a1));
    EXPECT_EQ(cudaSuccess, ret);

    const uint32_t a_init = 0;
    if (p & 0x1) {
        ret = cudaMemcpy(a0, &a_init, sizeof(*a0), cudaMemcpyHostToDevice);
        EXPECT_EQ(cudaSuccess, ret);

        ret = cudaMemcpy(a1, &a_init, sizeof(*a1), cudaMemcpyHostToDevice);
        EXPECT_EQ(cudaSuccess, ret);
    }

    uint32_t *d0, *d1;
    ret = cudaMalloc((void **) &d0, 2 * sizeof(*d0));
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaMalloc((void **) &d1, 2 * sizeof(*d1));
    EXPECT_EQ(cudaSuccess, ret);

    uint32_t b0, c0, b1, c1;

    c0 = 0xDEADBEEF;
    if (!(p & 0x4)) {
        VALGRIND_MAKE_MEM_UNDEFINED(&c0, sizeof(c0) / 2);
    }

    const bool bret0 = launch_atomic_global_cas_const1(2, d0, a0, &b0,  c0);
    const bool bret1 = launch_atomic_global_cas_const2(2, d1, a1, &b1, &c1);
    EXPECT_TRUE(bret0);
    EXPECT_TRUE(bret1);

    uint32_t ha0, hd0[2], ha1, hd1[2];

    ret = cudaMemcpy(&ha0, a0, sizeof(ha0), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);
    ret = cudaMemcpy(&hd0, d0, sizeof(hd0), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);
    ret = cudaMemcpy(&ha1, a1, sizeof(ha1), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);
    ret = cudaMemcpy(&hd1, d1, sizeof(hd1), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cudaSuccess, ret);

    ret = cudaFree(d0);
    EXPECT_EQ(cudaSuccess, ret);
    ret = cudaFree(a0);
    EXPECT_EQ(cudaSuccess, ret);
    ret = cudaFree(d1);
    EXPECT_EQ(cudaSuccess, ret);
    ret = cudaFree(a1);
    EXPECT_EQ(cudaSuccess, ret);

    /* We do not expect anything except for ha and hd to be uninitialized. */
    uint32_t va0, vd0[2], va1, vd1[2];
    const bool vret = (VALGRIND_GET_VBITS(&ha0, &va0, sizeof(ha0)) == 1) &&
                      (VALGRIND_GET_VBITS(&hd0, &vd0, sizeof(hd0)) == 1) &&
                      (VALGRIND_GET_VBITS(&ha1, &va1, sizeof(ha1)) == 1) &&
                      (VALGRIND_GET_VBITS(&hd1, &vd1, sizeof(hd1)) == 1);
    if (vret) {
        uint32_t eva0, evd0[2], eva1, evd1[2];
        switch (p) {
            case 0:
            case 2:
            case 4:
            case 6:
                eva0 = evd0[0] = evd0[1] = 0xFFFFFFFF;
                eva1 = evd1[0] = evd1[1] = 0xFFFFFFFF;
                break;
            case 1:
            case 3:
                /* The first thread to encounter the address will see a valid
                 * value.  The second will see the invalid value left by the
                 * first. */
                evd0[0] = evd1[0] = eva1 = evd1[1] = 0x0;
                eva0 = evd0[1] = 0x0000FFFF;
                break;
            case 5:
            case 7:
                eva0 = evd0[0] = evd0[1] = eva1 = evd1[0] = evd1[1] = 0x0;
                break;
        }

        EXPECT_EQ(eva0, va0);

        std::sort(vd0, vd0 + 1);
        EXPECT_EQ(evd0[0], vd0[0]);
        EXPECT_EQ(evd0[1], vd0[1]);

        EXPECT_EQ(eva1, va1);

        std::sort(vd1, vd1 + 1);
        EXPECT_EQ(evd1[0], vd1[0]);
        EXPECT_EQ(evd1[1], vd1[1]);

        VALGRIND_MAKE_MEM_DEFINED(&ha0, sizeof(ha0));
        VALGRIND_MAKE_MEM_DEFINED( hd0, sizeof(hd0));
        VALGRIND_MAKE_MEM_DEFINED(&ha1, sizeof(ha1));
        VALGRIND_MAKE_MEM_DEFINED( hd1, sizeof(hd1));
    }

    if (p & 1) {
        if (!(p & 0x4)) {
            VALGRIND_MAKE_MEM_DEFINED(&c0, sizeof(c0));
        }

        std::sort(hd0, hd0 + 1);
        std::sort(hd1, hd1 + 1);

        EXPECT_EQ(a_init, hd0[0]);
        if (a_init == b0) {
            EXPECT_EQ(c0, hd0[1]);
        } else {
            EXPECT_EQ(a_init, hd0[1]);
        }

        EXPECT_EQ(a_init, hd0[0]);
        if (a_init == b0) {
            EXPECT_EQ(c1, hd1[1]);
        } else {
            EXPECT_EQ(a_init, hd0[1]);
        }
    }
}

INSTANTIATE_TEST_CASE_P(GlobalCASTests, AtomicGlobalFixtureCAS,
    ::testing::Range(0u, 8u));

TEST(GlobalOverrun, DoTest) {
    const size_t N = 13;
    uint32_t * a;
    cudaError_t ret = cudaMalloc((void **) &a, N * sizeof(*a));
    ASSERT_EQ(cudaSuccess, ret);

    for (size_t i = 0; i < N; i++) {
        ret = launch_red_global_overrun(a);
        EXPECT_EQ(cudaSuccess, ret);
    }

    /**
     * We need a large offset to cause an actual launch failure.
     */
    ret = launch_red_global_overrun(a + (1 << 18));
    EXPECT_EQ(cudaErrorLaunchFailure, ret);

    ret = cudaDeviceReset();
    EXPECT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
