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

#ifndef __PANOPTES__PANOPTES__UTILITIES_H__
#define __PANOPTES__PANOPTES__UTILITIES_H__

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace panoptes {

/**
 * This converts from driver API return codes to runtime API return codes.
 */
cudaError_t cuToCUDA(CUresult ret);

}

#endif // __PANOPTES__PANOPTES__UTILITIES_H__
