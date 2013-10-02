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

#include <panoptes/registry.h>

using namespace panoptes;

registry & registry::instance() {
    static registry i;
    return i;
}

global_context* registry::create(const std::string & name) {
    EntryMap::const_iterator it = entries_.find(name);
    if (it == entries_.end()) {
        return NULL;
    } else {
        return it->second();
    }
}

registry::registry() { }
