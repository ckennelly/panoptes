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

#include <boost/utility.hpp>
#include "context.h"
#include <stack>

namespace panoptes {

/**
 * Panoptes adopts the CUDA 4.0 context model, that is context-per-device,
 * at least for interactions with the runtime API.
 *
 * TODO:  Support explicit context creation and deletion operations in order
 *        to support the driver API.
 */
class thread_context : public boost::noncopyable {
public:
    /**
     * Accessors for this thread's current context.  Returns NULL if no context
     * is on the stack.
     */
    cuda_context * context();
    const cuda_context * context() const;

    /**
     * Returns the thread's current cuda context.
     */
    static thread_context & instance();

    /**
     * Destructor.
     */
    ~thread_context();

    /**
     * Initializes the context according to the runtime; that is, it uses the
     * context for the current device (as done by CUDA 4.0).  For driver API
     * usage, not calling this lives the context stack empty for appropriate
     * context initialization.
     */
    void init_runtime();

    /**
     * Sets the last error value for cudaGetLastError.
     */
    cudaError_t setLastError(cudaError_t error);

    cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int
        peerDevice);
    cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
    cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

    cudaError_t cudaDeviceGetByPCIBusId(int *device, char *pciBusId);
    cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);

    /**
     * Implementation of cudaGetDevice
     */
    cudaError_t cudaGetDevice(int *device) const;

    /**
     * Implementation of cudaGetDeviceCount.
     */
    cudaError_t cudaGetDeviceCount(int *count) const;

    cudaError_t cudaGetLastError(void);
    cudaError_t cudaPeekAtLastError(void);

    /**
     * Implementation of cudaSetDevice.
     *
     * TODO:  This function only swaps contexts if the current context belongs to
     *        the corresponding process-wide context for its current device.  This
     *        may be a violation of how the CUDA library acts.
     */
    cudaError_t cudaSetDevice(int device);

    cudaError_t cudaSetDeviceFlags(unsigned int flags);

    cudaError_t cudaSetValidDevices(int *device_arr, int len);

    cudaError_t cudaThreadExit(void);
private:
    /**
     * Constructor.
     */
    thread_context();

    /**
     * Indicates whether this thread has been initialized by a runtime call.
     */
    bool runtime_;

    /**
     * Maintains the currently selected device
     */
    int device_;

    cudaError_t last_error_;

    /**
     * Stack for contexts.
     */
    std::stack<cuda_context *> context_stack_;
}; // end class thread_context

} // end namespace panoptes
