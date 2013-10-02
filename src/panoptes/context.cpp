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

#include <panoptes/context.h>

using namespace panoptes;

context::context() : error_(cudaSuccess) { }

context::~context() { }

cudaError_t context::cudaGetLastError() {
    cudaError_t ret = error_;
    error_ = cudaSuccess;
    return ret;
}

cudaError_t context::cudaPeekAtLastError() {
    return error_;
}

cudaError_t context::setLastError(cudaError_t error) const {
    if (error != cudaSuccess) {
        error_ = error;
    }

    return error_;
}

cudaChannelFormatDesc context::cudaCreateChannelDesc(int x, int y, int z, int w,
        enum cudaChannelFormatKind f) {
    cudaChannelFormatDesc ret;
    ret.x = x;
    ret.y = y;
    ret.z = z;
    ret.w = w;
    ret.f = f;
    return ret;
}
