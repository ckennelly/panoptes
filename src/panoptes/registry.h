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

#ifndef __PANOPTES__REGISTRY_H__
#define __PANOPTES__REGISTRY_H__

#include <boost/functional/factory.hpp>
#include <boost/function.hpp>
#include <map>
#include <panoptes/global_context.h>

namespace panoptes {

/**
 * This provides a registry of tool (global context types).
 */
class registry {
public:
    static registry & instance();

    /**
     * Registers a tool.  This is not thread-safe.
     */
    template<typename T>
    void add_tool(const std::string & name) {
        entries_[name] = boost::factory<T *>();
    }

    global_context* create(const std::string & name);
private:
    registry();

    typedef std::map<std::string, boost::function<global_context*()> > EntryMap;
    EntryMap entries_;
};

}

#endif // __PANOPTES__REGISTRY_H__
