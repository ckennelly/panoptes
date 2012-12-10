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

#ifndef __PANOPTES__GPU_POOL_H_
#define __PANOPTES__GPU_POOL_H_

#include <boost/utility.hpp>
#include <cassert>
#include "host_gpu_vector.h"
#include <limits>
#include <map>
#include <set>
#include <valgrind/valgrind.h>

namespace panoptes {

template<class T>
class gpu_pool {
    typedef host_gpu_vector<T> storage_type;
public:
    /**
     * Handle for receiving a GPU/Host pair of pointers of type T
     */
    class handle_t : boost::noncopyable {
        friend class gpu_pool<T>;
    public:
        T * host() { return host_; }
        const T * host() const { return host_; }
        T * gpu() { return gpu_; }
        const T * gpu() const { return gpu_; }

        handle_t(T* host__, T* gpu__) : host_(host__), gpu_(gpu__),
            cb_(NULL) { }
    protected:
        handle_t(T* host__, T* gpu__, storage_type * cb) :
            host_(host__), gpu_(gpu__), cb_(cb) { }

        storage_type * containing_block() const { return cb_; }
    private:
        T * const host_;
        T * const gpu_;
        storage_type * const cb_;
    };
public:
    /**
     * Obtains a handle.
     */
    handle_t * allocate() {
        /* Check free list.  If nothing is available, allocate */
        if (free_.size() == 0) {
            /* Actually allocate */
            storage_type * nblock = new storage_type(block_size_);
            VALGRIND_CREATE_MEMPOOL(nblock->host(), 0, 0);

            /* Add to free list */
            for (size_t i = 0; i < block_size_; ++i) {
                free_.insert(typename free_map_t::value_type(nblock, i));
            }

            blocks_.insert(nblock);
        }
        assert(free_.size() > 0);

        /* Take first T */
        typename free_map_t::iterator it = free_.begin();

        storage_type *  block = it->first;
        size_t          index = it->second;
        free_.erase(it);

        /* Get pointers */
        T * host    = block->host() + index;
        T * gpu     = block->gpu()  + index;

        /* Add to used list */
        VALGRIND_MEMPOOL_ALLOC(block->host(), host, sizeof(*host));
        in_use_.insert(typename used_map_t::value_type(host, block));

        /* Make handle and return */
        return new handle_t(host, gpu, block);
    }

    void free(handle_t * handle) {
        /* Retrieve key, e.g. the host pointer */
        T * host = handle->host();
        delete handle;

        /* Find block and compute index */
        typename used_map_t::iterator it = in_use_.find(host);
        assert(it != in_use_.end());
        storage_type * block = it->second;
        in_use_.erase(it);

        VALGRIND_MEMPOOL_FREE(block->host(), host);

        off_t          oindex = host - block->host();
        assert(oindex >= 0);
        assert(oindex < (off_t) block_size_);
        // Now it's safe to cast
        size_t         index = (size_t) oindex;

        /*
         * Consider deallocation if we will still have some free chunks to give
         * out and all of the other parts of this block have been returned to
         * the free list.
         */
        if (free_.size() + 1 >= block_size_ + max_free_ &&
                free_.count(block) == block_size_ - 1) {
            /* Remove from free list */
            free_.erase(block);
            /* Deallocate */
            blocks_.erase(block);
            VALGRIND_DESTROY_MEMPOOL(block->host());
            delete block;
        } else {
            /* Add back to free list */
            free_.insert(typename free_map_t::value_type(block, index));
        }
    }

    /**
     * Copies all host-based data to the GPU.
     */
    void to_gpu() {
        for (typename block_set_t::iterator it = blocks_.begin();
                it != blocks_.end(); ++it) {
            (*it)->to_gpu();
        }
    }

    template<typename Iter>
    void to_gpu(Iter first, Iter last) {
        block_set_t blocks;
        for (Iter it = first; it != last; ++it) {
            storage_type * container = (*it)->containing_block();
            if (container) {
                blocks.insert(container);
            } else {
                callout::cudaMemcpy((*it)->gpu(), (*it)->host(), sizeof(T),
                    cudaMemcpyHostToDevice);
            }
        }

        for (typename block_set_t::iterator it = blocks.begin();
            it != blocks.end(); ++it) {
            assert(blocks_.find(*it) != blocks_.end());
            (*it)->to_gpu();
        }
    }

    /**
     * Copies all GPU-based data to the host.
     */
    void to_host() {
        for (typename block_set_t::iterator it = blocks_.begin();
                it != blocks_.end(); ++it) {
            (*it)->to_host();
        }
    }

    template<typename Iter>
    void to_host(Iter first, Iter last) {
        block_set_t blocks;
        for (Iter it = first; it != last; ++it) {
            storage_type * container = (*it)->containing_block();
            if (container) {
                blocks.insert(container);
            } else {
                callout::cudaMemcpy((*it)->host(), (*it)->gpu(), sizeof(T),
                    cudaMemcpyDeviceToHost);
            }
        }

        for (typename block_set_t::iterator it = blocks.begin();
            it != blocks.end(); ++it) {
            assert(blocks_.find(*it) != blocks_.end());
            (*it)->to_host();
        }
    }

    /**
     * Constructs a pool.  block_size is the number of T's which are allocated
     * at a time.  If at any point there would be at least max_free T's that
     * are free after deallocating a contiguous block, that block is
     * deallocated.
     */
    gpu_pool(size_t block_size, size_t max_free) : block_size_(block_size),
            max_free_(max_free) {
        assert(block_size < (size_t) std::numeric_limits<off_t>::max());
    }

    /**
     * Destructs the pool.   All pointers held by handles are invalidated.
     */
    ~gpu_pool() {
        assert(in_use_.size() == 0);

        // Free the blocks
        for (typename block_set_t::iterator it = blocks_.begin();
                it != blocks_.end(); ++it) {
            free_.erase(*it);
            VALGRIND_DESTROY_MEMPOOL((*it)->host());
            delete *it;
        }
    }
private:
    /**
     * Raw blocks.
     */
    typedef std::set<storage_type *> block_set_t;
    block_set_t blocks_;

    /**
     * List of free T's.  (block) -> (index)
     */
    typedef std::multimap<storage_type *, size_t> free_map_t;
    free_map_t free_;

    /**
     * Mapping for host pointers which have been given out to their parent
     * blocks.  Host pointers alone are necessary since everything is given
     * out in pairs.  Pointer arithmetic is sufficient to compute the index.
     */
    typedef std::map<T*, storage_type *> used_map_t;
    used_map_t in_use_;

    const size_t block_size_;
    const size_t max_free_;
}; // end class gpu_pool

} // end namespace panoptes

#endif // __PANOPTES__GPU_POOL_H_
