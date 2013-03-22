/**
 * Panoptes - A Binary Translation Framework for CUDA
 * (c) 2011-2013 Chris Kennelly <chris@ckennelly.com>
 *
 * This header (and this header alone) is licensed under the zlib license.
 * All other Panoptes-related materials are licensed under the GNU
 * General Public License version 3 or later.
 *
 ******************************************************************************
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising from the
 * use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software in a
 *    product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 *
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef __PANOPTES__PANOPTES_H__
#define __PANOPTES__PANOPTES_H__

/**
 * This file defines several hooks to access and modify the state of Panoptes
 * when a program is run under it.  These hooks require a compiler and linker
 * that support weak symbols.
 */
#ifdef __cplusplus
extern "C" {
#endif

/* Actual symbols. ************************************************************/

/**
 * This returns an integer indicating that the program is running under
 * Panoptes.  While this symbol being defined likely indicates that the program
 * is being run under Panoptes, this function may return values greater than 1
 * should Panoptes ever be adapted to run under itself.
 */
extern int __panoptes__running_on_panoptes(void) __attribute__((weak));

/* Wrappers. ******************************************************************/
inline int panoptes_running_on_panoptes(void) {
    if (__panoptes__running_on_panoptes) {
        return __panoptes__running_on_panoptes();
    } else {
        return 0;
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // __PANOPTES__PANOPTES_H__

