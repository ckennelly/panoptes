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

#include <panoptes/utilities.h>

namespace panoptes {

cudaError_t cuToCUDA(CUresult ret) {
    switch (ret) {
        case CUDA_SUCCESS:
            return cudaSuccess;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_DEINITIALIZED:
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
            return cudaErrorInitializationError;
        default:
            return cudaErrorUnknown;
    }
}

}
