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

#ifndef __PANOPTES__TEST__PTX_IO__TEST_UTILITIES_H__
#define __PANOPTES__TEST__PTX_IO__TEST_UTILITIES_H__

#include <string>
#include <vector>

namespace panoptes {

/**
 * This provides a basic interface for forking and running ptxas, as to sanity
 * check PTX programs being tested.
 */
class ptx_checker {
public:
    ptx_checker();
    ~ptx_checker();

    /**
     * Attempts to compile the specified PTX program.  Returns true if
     * successful.
     *
     * This program blocks on the completion of ptxas.
     */
    bool check(const std::string & program);
    bool check(const std::string & program,
        const std::vector<std::string> & args);
};

}

#endif // __PANOPTES__TEST__PTX_IO__TEST_UTILITIES_H__
