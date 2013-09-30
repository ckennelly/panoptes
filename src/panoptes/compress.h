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

#ifndef __PANOPTES__COMPRESS_H__
#define __PANOPTES__COMPRESS_H__

#include <boost/utility.hpp>

namespace panoptes {

class CompressZlib : boost::noncopyable {
public:
    CompressZlib();
    ~CompressZlib();

    bool decompress(const void *in, size_t in_size, void *out,
        size_t *out_size);
};

}

#endif // __PANOPTES__COMPRESS_H__
