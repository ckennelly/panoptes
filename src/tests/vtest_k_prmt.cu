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

#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <valgrind/memcheck.h>

__global__ void k_prmt(uint32_t * d, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a), "r"(b),
        "r"(c));
    *d = ret;
}

__global__ void k_prmt_f4e(uint32_t * d, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm volatile("prmt.b32.f4e %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a), "r"(b),
        "r"(c));
    *d = ret;
}

__global__ void k_prmt_b4e(uint32_t * d, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm volatile("prmt.b32.b4e %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a), "r"(b),
        "r"(c));
    *d = ret;
}

__global__ void k_prmt_rc8(uint32_t * d, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm volatile("prmt.b32.b4e %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a), "r"(b),
        "r"(c));
    *d = ret;
}

__global__ void k_prmt_ecl(uint32_t * d, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm volatile("prmt.b32.ecl %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a), "r"(b),
        "r"(c));
    *d = ret;
}

__global__ void k_prmt_ecr(uint32_t * d, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm volatile("prmt.b32.ecr %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a), "r"(b),
        "r"(c));
    *d = ret;
}

__global__ void k_prmt_rc16(uint32_t * d, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm volatile("prmt.b32.rc16 %0, %1, %2, %3;\n" : "=r"(ret) : "r"(a), "r"(b),
        "r"(c));
    *d = ret;
}

TEST(Permute, Single) {
    cudaError_t ret;
    cudaStream_t stream;

    uint32_t * d;
    ret = cudaMalloc((void **) &d, 3u * sizeof(*d));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t a = 0x809122B3;
    uint32_t b = 0xC455E6FF;
    uint32_t c = 0x00008F94;

    /**
     * Apply a set of validity bits to a and b.
     */
    uint32_t va = 0xDEADBEEF;
    uint32_t vb = 0xFFFE7F7E;
    const int valgrind = VALGRIND_SET_VBITS(&a, &va, sizeof(a));
    assert(valgrind == 0 || valgrind == 1);
    (void) VALGRIND_SET_VBITS(&b, &vb, sizeof(b));

    uint32_t expected = 0;
    for (unsigned i = 0; i < 4u; i++) {
        uint8_t select = (c >> (4u * i)) & 0xF;
        const bool sign = select & 0x8;
        select &= 0x7;

        const uint64_t tmp64 = (((uint64_t) b) << 32) | a;

        union {
            uint8_t u;
            int8_t i;
        } b;

        b.u = (tmp64 >> (CHAR_BIT * select)) & 0xFF;
        if (sign) {
            b.i = b.i >> (CHAR_BIT - 1);
        }

        expected |= b.u << (8u * i);
    }

    k_prmt<<<1, 1, 0, stream>>>(d + 0, a, b, c);

    /**
     * If we're running under Valgrind, set the validity bits of c and
     * measure the outcome.
     */
    if (valgrind) {
        /* This should have no impact on the validity bits. */
        const uint32_t vchi = 0xFFFF0000;
        (void) VALGRIND_SET_VBITS(&c, &vchi, sizeof(c));
        k_prmt<<<1, 1, 0, stream>>>(d + 1, a, b, c);

        /**
         * This should have a catastrophic impact on the validity bits.
         * TODO:  If/when Panoptes supports byte-level invalidity propagation,
         *        this test will need to be updated.
         */
        const uint32_t vclo = 0x00001000;
        (void) VALGRIND_SET_VBITS(&c, &vclo, sizeof(c));
        k_prmt<<<1, 1, 0, stream>>>(d + 2, a, b, c);
    }

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hd[3];
    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * If we're running under Valgrind, check the validity bits then
     * mark hd as entirely valid so our comparison does not produce spurioius
     * warnings from Valgrind.
     */
    if (valgrind) {
        uint32_t vhd[3];
        (void) VALGRIND_GET_VBITS(&hd, &vhd, sizeof(hd));

        uint32_t vexpected;
        (void) VALGRIND_GET_VBITS(&expected, &vexpected, sizeof(expected));

        uint32_t vcatastrophic = 0xFFFFFFFF;
        /**
         * The first two values had fully valid bytes for c, so the expected
         * validity bits should match.  The third value had an invalid bit in
         * c, so the output should be completely invalid.
         */
        EXPECT_EQ(vexpected,     vhd[0]);
        EXPECT_EQ(vexpected,     vhd[1]);
        EXPECT_EQ(vcatastrophic, vhd[2]);

        (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&expected,
            sizeof(expected));
        (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&hd, sizeof(hd));
    }

    EXPECT_EQ(expected, hd[0]);

    if (valgrind) {
        EXPECT_EQ(expected, hd[1]);
        EXPECT_EQ(expected, hd[2]);
    }

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

TEST(Permute, SingleMode) {
    cudaError_t ret;
    cudaStream_t stream;

    uint32_t * d;
    ret = cudaMalloc((void **) &d, 18u * sizeof(*d));
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t a = 0x809122B3;
    uint32_t b = 0xC455E6FF;
    uint32_t c = 0x00008F94;

    /**
     * Apply a set of validity bits to a and b.
     */
    uint32_t va = 0xDEADBEEF;
    uint32_t vb = 0xFFFE7F7E;
    const int valgrind = VALGRIND_SET_VBITS(&a, &va, sizeof(a));
    (void) VALGRIND_SET_VBITS(&b, &vb, sizeof(b));

    const uint32_t  expected[6] = {0x809122B3, 0xE655C4B3, 0xE655C4B3,
                                   0x809122B3, 0xB3B3B3B3, 0x22B322B3};
    const uint32_t vexpected[6] = {0xDEADBEEF, 0x7FFEFFEF, 0x7FFEFFEF,
                                   0xDEADBEEF, 0xEFEFEFEF, 0xBEEFBEEF};

    k_prmt_f4e <<<1, 1, 0, stream>>>(d + 0, a, b, c);
    k_prmt_b4e <<<1, 1, 0, stream>>>(d + 1, a, b, c);
    k_prmt_rc8 <<<1, 1, 0, stream>>>(d + 2, a, b, c);
    k_prmt_ecl <<<1, 1, 0, stream>>>(d + 3, a, b, c);
    k_prmt_ecr <<<1, 1, 0, stream>>>(d + 4, a, b, c);
    k_prmt_rc16<<<1, 1, 0, stream>>>(d + 5, a, b, c);

    /**
     * If we're running under Valgrind, set the validity bits of c and
     * measure the outcome.
     */
    if (valgrind) {
        /* This should have no impact on the validity bits. */
        const uint32_t vchi = 0xFFFFFFFC;
        (void) VALGRIND_SET_VBITS(&c, &vchi, sizeof(c));
        k_prmt_f4e <<<1, 1, 0, stream>>>(d +  6, a, b, c);
        k_prmt_b4e <<<1, 1, 0, stream>>>(d +  7, a, b, c);
        k_prmt_rc8 <<<1, 1, 0, stream>>>(d +  8, a, b, c);
        k_prmt_ecl <<<1, 1, 0, stream>>>(d +  9, a, b, c);
        k_prmt_ecr <<<1, 1, 0, stream>>>(d + 10, a, b, c);
        k_prmt_rc16<<<1, 1, 0, stream>>>(d + 11, a, b, c);

        /**
         * This should have a catastrophic impact on the validity bits.
         * TODO:  If/when Panoptes supports byte-level invalidity propagation,
         *        this test will need to be updated.
         */
        const uint32_t vclo = 0x00000001;
        (void) VALGRIND_SET_VBITS(&c, &vclo, sizeof(c));
        k_prmt_f4e <<<1, 1, 0, stream>>>(d + 12, a, b, c);
        k_prmt_b4e <<<1, 1, 0, stream>>>(d + 13, a, b, c);
        k_prmt_rc8 <<<1, 1, 0, stream>>>(d + 14, a, b, c);
        k_prmt_ecl <<<1, 1, 0, stream>>>(d + 15, a, b, c);
        k_prmt_ecr <<<1, 1, 0, stream>>>(d + 16, a, b, c);
        k_prmt_rc16<<<1, 1, 0, stream>>>(d + 17, a, b, c);
    }

    ret = cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaSuccess, ret);

    ret = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, ret);

    uint32_t hd[18];
    ret = cudaMemcpy(&hd, d, sizeof(hd), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, ret);

    /**
     * If we're running under Valgrind, check the validity bits then
     * mark hd as entirely valid so our comparison does not produce spurioius
     * warnings from Valgrind.
     */
    if (valgrind) {
        uint32_t vhd[18];
        (void) VALGRIND_GET_VBITS(&hd, &vhd, sizeof(hd));

        const uint32_t vcatastrophic = 0xFFFFFFFF;
        /**
         * The first two values had fully valid bytes for c, so the expected
         * validity bits should match.  The third value had an invalid bit in
         * c, so the output should be completely invalid.
         */
        EXPECT_EQ(vexpected[0],  vhd[ 0]);
        EXPECT_EQ(vexpected[1],  vhd[ 1]);
        EXPECT_EQ(vexpected[2],  vhd[ 2]);
        EXPECT_EQ(vexpected[3],  vhd[ 3]);
        EXPECT_EQ(vexpected[4],  vhd[ 4]);
        EXPECT_EQ(vexpected[5],  vhd[ 5]);
        EXPECT_EQ(vexpected[0],  vhd[ 6]);
        EXPECT_EQ(vexpected[1],  vhd[ 7]);
        EXPECT_EQ(vexpected[2],  vhd[ 8]);
        EXPECT_EQ(vexpected[3],  vhd[ 9]);
        EXPECT_EQ(vexpected[4],  vhd[10]);
        EXPECT_EQ(vexpected[5],  vhd[11]);

        EXPECT_EQ(vcatastrophic, vhd[12]);
        EXPECT_EQ(vcatastrophic, vhd[13]);
        EXPECT_EQ(vcatastrophic, vhd[14]);
        EXPECT_EQ(vcatastrophic, vhd[15]);
        EXPECT_EQ(vcatastrophic, vhd[16]);
        EXPECT_EQ(vcatastrophic, vhd[17]);

        (void) VALGRIND_MAKE_MEM_DEFINED_IF_ADDRESSABLE(&hd, sizeof(hd));
    }

    EXPECT_EQ(expected[0], hd[ 0]);
    EXPECT_EQ(expected[1], hd[ 1]);
    EXPECT_EQ(expected[2], hd[ 2]);
    EXPECT_EQ(expected[3], hd[ 3]);
    EXPECT_EQ(expected[4], hd[ 4]);
    EXPECT_EQ(expected[5], hd[ 5]);

    if (valgrind) {
        EXPECT_EQ(expected[0], hd[ 6]);
        EXPECT_EQ(expected[1], hd[ 7]);
        EXPECT_EQ(expected[2], hd[ 8]);
        EXPECT_EQ(expected[3], hd[ 9]);
        EXPECT_EQ(expected[4], hd[10]);
        EXPECT_EQ(expected[5], hd[11]);

        EXPECT_EQ(expected[0], hd[12]);
        EXPECT_EQ(expected[1], hd[13]);
        EXPECT_EQ(expected[2], hd[14]);
        EXPECT_EQ(expected[3], hd[15]);
        EXPECT_EQ(expected[4], hd[16]);
        EXPECT_EQ(expected[5], hd[17]);
    }

    ret = cudaFree(d);
    ASSERT_EQ(cudaSuccess, ret);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
