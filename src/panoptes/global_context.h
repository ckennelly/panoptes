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
#ifndef __PANOPTES__GLOBAL_CONTEXT_H__
#define __PANOPTES__GLOBAL_CONTEXT_H__

#include <boost/thread/mutex.hpp>
#include <boost/thread/tss.hpp>
#include <boost/unordered_map.hpp>
#include <boost/utility.hpp>
#include <cuda_runtime_api.h>
#include <vector>

namespace panoptes {
/**
 * Forward declaration.
 */
namespace internal {
    struct modules_t;
    struct module_t;
}

class cuda_context;
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
    cuda_context * context();
    const cuda_context * context() const;

    cuda_context * context(unsigned device);
    const cuda_context * context(unsigned device) const;
protected:
    /**
     * Accessors for particular device contexts.  No locks are held.
     */
    cuda_context * context_impl(unsigned device);
    const cuda_context * context_impl(unsigned device) const;
public:
    static global_context & instance();

    /**
     * Destructor.
     */
    virtual ~global_context();

    /**
     * CUDA methods.
     */
    virtual cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
        int peerDevice);
    virtual cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
    virtual cudaError_t cudaDeviceEnablePeerAccess(int peerDevice,
        unsigned int flags);
    cudaError_t cudaDeviceGetByPCIBusId(int *device, char *pciBusId);
    cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);
    cudaError_t cudaDeviceReset();
    cudaError_t cudaGetDevice(int *device) const;
    cudaError_t cudaGetDeviceCount(int *count) const;
    virtual void** cudaRegisterFatBinary(void *fatCubin);
    virtual void** cudaUnregisterFatBinary(void **fatCubinHandle);
    virtual void cudaRegisterFunction(void **fatCubinHandle,
        const char *hostFun, char *deviceFun, const char *deviceName,
        int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
        int *wSize);
    virtual void cudaRegisterVar(void **fatCubinHandle,char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global);
    virtual void cudaRegisterTexture(void **fatCubinHandle,
        const struct textureReference *hostVar, const void **deviceAddress,
        const char *deviceName, int dim, int norm, int ext);
    cudaError_t cudaSetDevice(int device);
    cudaError_t cudaSetDeviceFlags(unsigned int flags);
    cudaError_t cudaSetValidDevices(int *device_arr, int len);
    cudaError_t cudaThreadExit(void);
protected:
    /**
     * Constructor.
     */
    global_context();
public:
    /**
     * Actually loads the registered PTX code with CUDA.
     */
    void load_ptx(internal::modules_t * target);
protected:
    virtual void instrument(void **fatCubinHandle, ptx_t * target);
    virtual cuda_context * factory(int device, unsigned int flags) const;

    mutable boost::mutex mx_;
    mutable std::vector<cuda_context *> device_contexts_;

    /**
     * Module registry mapping from the handle returned to the module.
     */
    typedef boost::unordered_map<void **, internal::module_t *> module_map_t;
    module_map_t modules_;

    /**
     * fatCubin to registration handle map
     */
    typedef boost::unordered_map<void *, void**> fatbin_map_t;
    fatbin_map_t fatbins_;

    typedef boost::unordered_map<void **, void *> fatbin_imap_t;
    fatbin_imap_t ifatbins_;

    /**
     * Mapping of functions to their parent modules.
     */
    typedef boost::unordered_map<const void *,
        internal::module_t *> function_map_t;
    function_map_t functions_;
public:
    typedef boost::unordered_map<std::string, const char *> function_name_map_t;

    const function_name_map_t & function_names() const;
protected:
    function_name_map_t function_names_;

    /**
     * Mapping of variables to their parent modules.
     */
    typedef boost::unordered_map<const void *, internal::module_t *>
        variable_map_t;
    variable_map_t variables_;
public:
    typedef boost::unordered_map<std::string, const void *> variable_name_map_t;

    const variable_name_map_t & variable_names() const;
protected:
    variable_name_map_t variable_names_;

    /**
     * Mapping of textures to their parent modules.
     */
    typedef boost::unordered_map<const struct textureReference *,
        internal::module_t *> texture_map_t;
    texture_map_t textures_;
public:
    typedef boost::unordered_map<std::string, const struct textureReference *>
        texture_name_map_t;
    const texture_name_map_t & texture_names() const;

    bool is_texture_reference(const void * ptr) const;
protected:
    texture_name_map_t texture_names_;

    /**
     * This stores the flags to initialize a device with (when the time
     * comes).
     */
    std::vector<unsigned int> device_flags_;

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

    int driver_version_;
    unsigned devices_;
public:
    unsigned devices() const;
}; // end class global_context

} // end namespace panoptes

#endif // __PANOPTES__GLOBAL_CONTEXT_H__
