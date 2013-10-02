/**
 * Panoptes - A Binary Translation Framework for CUDA
 * (c) 2011-2013 Chris Kennelly <chris@ckennelly.com>
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

#include <__cudaFatFormat.h>
#include <boost/lexical_cast.hpp>
#include <cassert>
#include <cstdio>
#include <cuda.h>
#if CUDA_VERSION >= 5000
#include <fatbinary.h>
#endif
#include <panoptes/compress.h>
#include <panoptes/fat_binary.h>

using namespace panoptes;

fat_binary_exception::fat_binary_exception(const std::string & str) :
    str_(str) { }

fat_binary_exception::~fat_binary_exception() throw() { }

const char *fat_binary_exception::what() const throw() {
    return str_.c_str();
}

fat_binary::fat_binary(void *fatCubin) {
    /**
     * Fat binaries contain an integer magic cookie.  Versions are
     * distinguished by differing values.  See also GPU Ocelot's implementation
     * of cuda::FatBinaryContext::FatBinaryContext, on which this is based.
     */
    const int   magic  = *(reinterpret_cast<int *>(fatCubin));

    if (magic == __cudaFatMAGIC) {
        /* This parsing strategy follows from __cudaFatFormat.h */

        const __cudaFatCudaBinary * handle =
            static_cast<const __cudaFatCudaBinary *>(fatCubin);
        /* TODO Handle gracefully */
        assert(handle->ptx);
        assert(handle->ptx[0].ptx);

        unsigned best_version = 0;
        const char *_ptx = NULL;

        for (unsigned i = 0; ; i++) {
            if (!(handle->ptx[i].ptx)) {
                break;
            }

            /* Grab compute capability in the form "compute_xy" */
            std::string profile_name = handle->ptx[i].gpuProfileName;
            std::string string_version(
                profile_name.begin() + sizeof("compute_") - 1,
                profile_name.end());

            if (profile_name.size() > 10) {
                char msg[128];
                snprintf(msg, sizeof(msg),
                    "Compute mode is too long (%zu bytes).",
                    profile_name.size());
                throw fat_binary_exception(msg);
            }

            unsigned numeric_version;
            try {
                numeric_version = boost::lexical_cast<unsigned>(
                    string_version);
            } catch (boost::bad_lexical_cast) {
                char msg[128];
                snprintf(msg, sizeof(msg), "Unable to parse compute mode '%s'.",
                    handle->ptx[i].gpuProfileName);
                throw fat_binary_exception(msg);
            }

            if (numeric_version > best_version) {
                best_version    = numeric_version;
                _ptx            = handle->ptx[i].ptx;
            }
        }

        if (_ptx) {
            ptx_ = _ptx;
        } else {
            assert(0);
        }
    } else if (magic == __cudaFatMAGIC2) {
        /* This follows from GPU Ocelot */
        const __cudaFatCudaBinary2 * handle =
            static_cast<const __cudaFatCudaBinary2 *>(fatCubin);
        const __cudaFatCudaBinary2Header * header =
            reinterpret_cast<const __cudaFatCudaBinary2Header *>(
                handle->fatbinData);

        const char * base = reinterpret_cast<const char *>(header + 1);
        unsigned long long offset = 0;

        const __cudaFatCudaBinary2Entry * entry =
            reinterpret_cast<const __cudaFatCudaBinary2Entry *>(base);
        while (!(entry->type & FATBIN_2_PTX) && offset < header->length) {
            entry   = reinterpret_cast<const __cudaFatCudaBinary2Entry *>(
                base + offset);
            offset  = entry->binary + entry->binarySize;
        }

        const char *data =
            reinterpret_cast<const char *>(entry) + entry->binary;
        #if CUDA_VERSION >= 5000
        if (entry->flags & FATBIN_FLAG_COMPRESS) {
            ptx_.resize(entry->uncompressedSize);

            CompressZlib z;
            size_t out_size = ptx_.size();
            if (z.decompress(data, entry->binarySize, &ptx_[0], &out_size)) {
                ptx_.resize(out_size);
            } else {
                const char msg[] = "Unable to decompress binary data.";
                throw fat_binary_exception(msg);
            }
        } else {
        #endif
            ptx_ = data;
        #if CUDA_VERSION >= 5000
        }
        #endif
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "Unknown cubin magic number '%08X'.", magic);
        throw fat_binary_exception(msg);
    }
}

fat_binary::~fat_binary() { }

const std::string & fat_binary::ptx() const {
    return ptx_;
}
