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

#ifndef __PANOPTES__GLOBAL_CUDA_CONTEXT_H__
#define __PANOPTES__GLOBAL_CUDA_CONTEXT_H__

#include <panoptes/global_context.h>

namespace panoptes {

/**
 * Forward declaration.
 */
namespace internal {
    struct modules_t;
    struct module_t;
}

/**
 * global_cuda_context provides a partial implementation of global_context,
 * fleshing out some methods that will use libcuda.
 */
class global_cuda_context : public global_context {
public:
    ~global_cuda_context();

    cudaError_t cudaChooseDevice(int *device,
        const struct cudaDeviceProp *prop);
    cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
        int peerDevice);
    cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
    cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
    cudaError_t cudaDeviceGetByPCIBusId(int *device, char *pciBusId);
    cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);
    cudaError_t cudaDeviceReset();
    void** cudaRegisterFatBinary(void *fatCubin);
    void** cudaUnregisterFatBinary(void **fatCubinHandle);
    void cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
    void cudaRegisterVar(void **fatCubinHandle,char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global);
    void cudaRegisterTexture(void **fatCubinHandle,
        const struct textureReference *hostVar, const void **deviceAddress,
        const char *deviceName, int dim, int norm, int ext);
    cudaError_t cudaSetDevice(int device);
    cudaError_t cudaSetValidDevices(int *device_arr, int len);

    /**
     * Actually loads the registered PTX code with CUDA.
     */
    void load_ptx(internal::modules_t * target);

    typedef boost::unordered_map<std::string, const char *> function_name_map_t;
    const function_name_map_t & function_names() const;

    typedef boost::unordered_map<std::string, const void *> variable_name_map_t;
    const variable_name_map_t & variable_names() const;

    typedef boost::unordered_map<std::string, const struct textureReference *>
        texture_name_map_t;
    const texture_name_map_t & texture_names() const;

    bool is_texture_reference(const void * ptr) const;
protected:
    global_cuda_context();

    /**
     * Accessors for particular device contexts.  No locks are held.
     */
    panoptes::context * context_impl(unsigned device, bool instantiate);
    const panoptes::context * context_impl(unsigned device,
        bool instantiate) const;

    /**
     * Instrument the target.
     */
    virtual void instrument(void **fatCubinHandle, ptx_t * target);

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

    function_name_map_t function_names_;

    /**
     * Mapping of variables to their parent modules.
     */
    typedef boost::unordered_map<const void *, internal::module_t *>
        variable_map_t;
    variable_map_t variables_;
    variable_name_map_t variable_names_;

    /**
     * Mapping of textures to their parent modules.
     */
    typedef boost::unordered_map<const struct textureReference *,
        internal::module_t *> texture_map_t;
    texture_map_t textures_;

    texture_name_map_t texture_names_;

    int driver_version_;
};

}

#endif // __PANOPTES__GLOBAL_CUDA_CONTEXT_H__
