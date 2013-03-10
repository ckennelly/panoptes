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
 *
 *****************************************************************************
 *
 * This code is largely adopted from GPU Ocelot for its labeling of fat binary
 * data structures produced by nvcc version 4.0 and version 5.0.
 */

const static int __cudaFatMAGIC2 = 0x466243b1;

typedef struct __cudaFatCudaBinary2HeaderRec {
    unsigned int            magic;
    unsigned int            version;
    unsigned long long int  length;
} __cudaFatCudaBinary2Header;

enum FatBin2EntryType {
    FATBIN_2_PTX = 0x1
};

typedef struct __cudaFatCudaBinary2EntryRec {
    unsigned int            type;
    unsigned int            binary;
    unsigned int            binarySize;
    unsigned int            unknown2;
    unsigned int            kindOffset;
    unsigned int            unknown3;
    unsigned int            unknown4;
    unsigned int            unknown5;
    unsigned int            name;
    unsigned int            nameSize;
    unsigned long long int  flags;
    unsigned long long int  unknown7;
    unsigned long long int  uncompressedSize;
} __cudaFatCudaBinary2Entry;

typedef struct __cudaFatCudaBinaryRec2 {
    int                         magic;
    int                         version;
    const unsigned long long *  fatbinData;
    char *                      f;
} __cudaFatCudaBinary2;
