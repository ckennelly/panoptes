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
#ifndef __PANOPTES__GLOBAL_CONTEXT_H__
#define __PANOPTES__GLOBAL_CONTEXT_H__

#include <boost/thread/mutex.hpp>
#include <boost/thread/tss.hpp>
#include <boost/unordered_map.hpp>
#include <boost/utility.hpp>
#include <cuda_runtime_api.h>
#include <vector>

namespace panoptes {

class context;
struct ptx_t;

/**
 * Panoptes adopts the CUDA 4.0 context model, that is context-per-device,
 * at least for interactions with the runtime API.
 *
 * TODO:  Support explicit context creation and deletion operations in order
 *        to support the driver API.
 */
class global_context : public boost::noncopyable {
public:
    /**
     * Accessors for this thread's current context.
     */
    panoptes::context * context();
    const panoptes::context * context() const;

    panoptes::context * context(unsigned device);
    const panoptes::context * context(unsigned device) const;
protected:
    /**
     * Accessors for particular device contexts.  No locks are held.
     */
    virtual panoptes::context * context_impl(unsigned device, bool instantiate);
    virtual const panoptes::context * context_impl(unsigned device,
        bool instantiate) const;
public:
    static global_context & instance();

    /**
     * Destructor.
     */
    virtual ~global_context();

    /**
     * CUDA methods.
     */
    virtual cudaError_t cudaChooseDevice(int *device,
        const struct cudaDeviceProp *prop) = 0;
    virtual cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
        int peerDevice) = 0;
    virtual cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) = 0;
    virtual cudaError_t cudaDeviceEnablePeerAccess(int peerDevice,
        unsigned int flags) = 0;
    virtual cudaError_t cudaDeviceGetByPCIBusId(int *device, char *pciBusId) = 0;
    virtual cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len,
        int device) = 0;
    virtual cudaError_t cudaDeviceReset();
    virtual cudaError_t cudaGetDevice(int *device) const;
    virtual cudaError_t cudaGetDeviceCount(int *count) const;
    virtual void** cudaRegisterFatBinary(void *fatCubin) = 0;
    virtual void** cudaUnregisterFatBinary(void **fatCubinHandle) = 0;
    virtual void cudaRegisterFunction(void **fatCubinHandle,
        const char *hostFun, char *deviceFun, const char *deviceName,
        int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
        int *wSize) = 0;
    virtual void cudaRegisterVar(void **fatCubinHandle,char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global) = 0;
    virtual void cudaRegisterTexture(void **fatCubinHandle,
        const struct textureReference *hostVar, const void **deviceAddress,
        const char *deviceName, int dim, int norm, int ext) = 0;
    virtual cudaError_t cudaSetDevice(int device);
    virtual cudaError_t cudaSetDeviceFlags(unsigned int flags);
    virtual cudaError_t cudaSetValidDevices(int *device_arr, int len) = 0;
    virtual cudaError_t cudaThreadExit(void);

    unsigned devices() const;
protected:
    /**
     * Constructor.
     */
    global_context();
protected:
    virtual panoptes::context * factory(int device,
        unsigned int flags) const = 0;

    mutable boost::mutex mx_;

    /**
     * A small bundle of data for each thread to reference particular device.
     */
    struct thread_info_t {
        thread_info_t();

        unsigned device;
        bool set_on_thread;
    };
    mutable boost::thread_specific_ptr<thread_info_t> threads_;
          thread_info_t * current();
    const thread_info_t * current() const;
    unsigned current_device() const;

    void set_devices(unsigned n);
    unsigned device_flags(unsigned device) const;
private:
    /**
     * This stores the flags to initialize a device with (when the time
     * comes).
     */
    std::vector<unsigned int> device_flags_;
    mutable std::vector<panoptes::context *> device_contexts_;
    unsigned devices_;
}; // end class global_context

} // end namespace panoptes

#endif // __PANOPTES__GLOBAL_CONTEXT_H__
