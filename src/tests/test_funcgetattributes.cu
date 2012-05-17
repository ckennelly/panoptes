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

#include <boost/scoped_array.hpp>
#include <boost/static_assert.hpp>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <sys/mman.h>
#include <valgrind/memcheck.h>
#include <cstdio>

class FuncGetAttributesFixture : public ::testing::Test {
public:
    virtual void SetUp() {
        const long page_size_ = sysconf(_SC_PAGESIZE);
        ASSERT_LT(0, page_size_);

        page_size = page_size_;

        buffer = mmap(NULL, 2u * page_size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
        EXPECT_FALSE(buffer == (void *) -1);
    }

    virtual void TearDown() {
        (void) munmap(buffer, 2u * page_size);
    }

    void clear() {
        memset(buffer, 0, 2u * page_size);
        (void) VALGRIND_MAKE_MEM_UNDEFINED(buffer, 2u * page_size);
    }

    void protect() {
        int ret = mprotect(static_cast<char *>(buffer) + page_size,
            page_size, PROT_NONE);
        EXPECT_EQ(0, ret);
    }

    void unprotect() {
        int ret = mprotect(static_cast<char *>(buffer) + page_size,
            page_size, PROT_READ | PROT_WRITE);
        EXPECT_EQ(0, ret);
    }

    size_t page_size;
    void * buffer;
};

TEST_F(FuncGetAttributesFixture, UnaddressablePointer) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    struct cudaFuncAttributes attr;
    cudaError_t ret;

    protect();

    for (unsigned i = 0; i < 13; i++) {
        size_t offset;
        if (i == 0) {
            offset = 0;
        } else {
            offset = 1u << (i - 1);
        }

        const char * new_buffer = static_cast<const char *>(buffer) +
            page_size - offset;

        if (offset > 0) {
            ret = cudaFuncGetAttributes(&attr, new_buffer);
            EXPECT_EQ(cudaErrorInvalidDeviceFunction, ret);
        } else {
            EXPECT_EXIT(
                cudaFuncGetAttributes(&attr, new_buffer),
                ::testing::KilledBySignal(SIGSEGV), "");
        }
    }

    unprotect();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
