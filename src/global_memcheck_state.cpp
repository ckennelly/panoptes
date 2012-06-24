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

#include "global_memcheck_state.h"

using namespace panoptes;

typedef boost::unique_lock<boost::mutex> scoped_lock;

global_memcheck_state::master_data_t::master_data_t(int device) :
        ownership(1 << (lg_max_memory - lg_chunk_bytes), device) { }

void global_memcheck_state::register_master(int device,
        metadata_chunk * default_chunk, master_t * master) {
    scoped_lock lock(mx_);

    /**
     * TODO:  Check that this is not an existing device.
     */
    master_data_t data(device);
    data.default_chunk = default_chunk;
    data.master        = master;
    data.peers.insert(device);

    masters_.insert(masters_t::value_type(device, data));
    master->to_gpu();
}

void global_memcheck_state::disable_peers(int device, int peer) {
    scoped_lock lock(mx_);
    disable_peers_impl(device, peer);
}

void global_memcheck_state::enable_peers(int device, int peer) {
    scoped_lock lock(mx_);
    enable_peers_impl(device, peer);
}

void global_memcheck_state::disable_peers_impl(int device, int peer) {
    {
        masters_t::iterator jit = masters_.find(peer);
        if (jit == masters_.end()) {
            return;
        }

        jit->second.peers.erase(device);
    }

    masters_t::iterator it = masters_.find(device);
    if (it == masters_.end()) {
        return;
    }

    master_data_t & data = it->second;
    const size_t N = data.ownership.size();
    master_t       * data_master  = data.master;
    metadata_chunk ** const host  = data_master->host();
    metadata_chunk * data_default = data.default_chunk;

    bool dirty = false;
    for (size_t i = 0; i < N; i++) {
        if (data.ownership[i] == peer) {
            data.ownership[i] = device;
            host[i] = data_default;
            dirty = true;
        }
    }

    if (dirty) {
        data_master->to_gpu();
    }
}

void global_memcheck_state::enable_peers_impl(int device, int peer) {
    masters_t::iterator peer_it = masters_.find(peer);
    if (peer_it == masters_.end()) {
        return;
    }

    masters_t::iterator device_it = masters_.find(device);
    if (device_it == masters_.end()) {
        return;
    }

    master_data_t & peer_data = peer_it->second;
    /* Add device as a peer. */
    peer_data.peers.insert(device);

    master_data_t & device_data = device_it->second;

    /* Iterate over peer_data, for any non-default chunk it owns, copy into
     * device. */
    const size_t N = peer_data.ownership.size();
    assert(N == device_data.ownership.size());
    metadata_chunk ** const phost = peer_data.master->host();
    metadata_chunk ** const dhost = device_data.master->host();

    metadata_chunk * const  pdefault = peer_data.default_chunk;
    metadata_chunk * const  ddefault = device_data.default_chunk;

    bool dirty = false;
    for (size_t i = 0; i < N; i++) {
        if (peer_data.ownership[i] == peer) {
            metadata_chunk * chunk = phost[i];
            if (chunk != pdefault) {
                assert(dhost[i] == ddefault && "Shared chunks not supported.");
                dhost[i] = chunk;
                dirty    = true;
            }
        }
    }

    if (dirty) {
        device_data.master->to_gpu();
    }
}

void global_memcheck_state::unregister_master(int device) {
    scoped_lock lock(mx_);

    masters_t::iterator it = masters_.find(device);
    if (it != masters_.end()) {
        const peer_set_t & peers = it->second.peers;
        for (peer_set_t::const_iterator jit = peers.begin();
                jit != peers.end(); ++jit) {
            const int peer = *jit;
            if (peer == device) {
                continue;
            }

            disable_peers_impl(device, peer);
        }

        masters_.erase(it);
    }
}

void global_memcheck_state::update_master(int device, bool add,
        const chunk_updates_t & updates) {
    const size_t n_updates = updates.size();
    if (n_updates == 0) {
        return;
    }

    scoped_lock lock(mx_);

    masters_t::iterator it = masters_.find(device);
    if (it == masters_.end()) {
        assert(0 && "The impossible happened.");
        return;
    }

    const peer_set_t & peers = it->second.peers;

    for (peer_set_t::iterator jit = peers.begin(); jit != peers.end(); ++jit) {
        const int peer = *jit;
        masters_t::iterator kit = masters_.find(peer);
        assert(kit != masters_.end());
        if (kit == masters_.end()) {
            continue;
        }

        master_t *        const master    = kit->second.master;
        metadata_chunk ** const host      = master->host();
        master_data_t::ownership_t & ownership = kit->second.ownership;

        const int owner = add ? device : peer;

        for (size_t i = 0; i < n_updates; i++) {
            const chunk_update_t & update = updates[i];
            const size_t index = update.first;

            metadata_chunk * fill = NULL;
            if (add) {
                fill = update.second;
            } else {
                fill = kit->second.default_chunk;
            }

            host[index] = fill;
            ownership[index] = owner;
        }

        const cudaError_t ret = cudaSetDevice(peer);
        assert(ret == cudaSuccess);
        master->to_gpu();
    }

    const cudaError_t ret = cudaSetDevice(device);
    assert(ret == cudaSuccess);
}

global_memcheck_state::global_memcheck_state() { }

global_memcheck_state::~global_memcheck_state() { }

void global_memcheck_state::register_stream(cudaStream_t stream,
        int device) {
    scoped_lock lock(mx_);
    streams_.insert(stream_map_t::value_type(stream, device));
}

bool global_memcheck_state::lookup_stream(cudaStream_t stream,
        int *device) const {
    scoped_lock lock(mx_);
    stream_map_t::const_iterator it = streams_.find(stream);
    if (it == streams_.end()) {
        return false;
    }

    *device = it->second;
    return true;
}

void global_memcheck_state::unregister_stream(cudaStream_t stream) {
    scoped_lock lock(mx_);
    streams_.erase(stream);
}
