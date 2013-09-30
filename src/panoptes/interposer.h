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

#ifndef __PANOPTES__INTERPOSER_H_
#define __PANOPTES__INTERPOSER_H_

#ifdef __CPLUSPLUS
 {
#endif

extern "C"
void** __cudaRegisterFatBinary(void *fatCubin);

extern "C"
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);

extern "C"
void __cudaRegisterTexture(void **fatCubinHandle,
        const struct textureReference *hostVar, const void **deviceAddress,
        const char *deviceName, int dim, int norm, int ext);

extern "C"
void __cudaRegisterVar(void **fatCubinHandle,char *hostVar, char *deviceAddress,
        const char *deviceName, int ext, int size, int constant, int global);

extern "C"
void __cudaUnregisterFatBinary(void **fatCubinHandle);

#ifdef __CPLUSPLUS
}
#endif

#endif // __PANOPTES__INTERPOSER_H_
