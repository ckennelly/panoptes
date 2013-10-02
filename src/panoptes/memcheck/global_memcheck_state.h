/**
 * Panoptes - A framework for detecting memory errors in GPU-based programs
 * Copyright (C) 2011 Chris Kennelly <chris@ckennelly.com>
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

#ifndef __PANOPTES__GLOBAL_MEMCHECK_STATE_H__
#define __PANOPTES__GLOBAL_MEMCHECK_STATE_H__

#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <map>
#include <panoptes/host_gpu_vector.h>
#include <panoptes/memcheck/metadata.h>
#include <set>

namespace panoptes {

/**
 * This class manages a small bit of state related to memory allocations of the
 * various devices.  Since it needs to destruct after global_context_memcheck
 * and the context_memchecks (e.g., after global_context), we keep it in a
 * shared_ptr.
 */
class global_memcheck_state {
public:
    global_memcheck_state();
    ~global_memcheck_state();

    typedef host_gpu_vector<metadata_ptrs> master_t;
    void register_master(int device, const metadata_ptrs & defaults,
        master_t * master);
    void unregister_master(int device);

    typedef std::pair<size_t, metadata_ptrs> chunk_update_t;
    typedef std::vector<chunk_update_t> chunk_updates_t;
    void update_master(int device, bool add, const chunk_updates_t & updates);

    void disable_peers(int device, int peer);
    void enable_peers(int device, int peer);

    void register_stream(cudaStream_t stream, int device);
    bool lookup_stream(cudaStream_t stream, int *device) const;
    void unregister_stream(cudaStream_t stream);
private:
    void disable_peers_impl(int device, int peer);
    void enable_peers_impl(int device, int peer);

    typedef std::set<int> peer_set_t;
    struct master_data_t {
        master_data_t(int device);

        metadata_ptrs   defaults;

        typedef std::vector<int> ownership_t;

        ownership_t      ownership;
        master_t       * master;
        peer_set_t       peers;
    };

    typedef std::map<int, master_data_t> masters_t;
    masters_t masters_;

    typedef boost::unordered_map<cudaStream_t, int> stream_map_t;
    stream_map_t streams_;

    mutable boost::mutex mx_;
};

typedef boost::shared_ptr<global_memcheck_state> state_ptr_t;
}

#endif // __PANOPTES__GLOBAL_MEMCHECK_STATE_H__
