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

#ifndef __PANOPTES__BACKTRACE_H__
#define __PANOPTES__BACKTRACE_H__

namespace panoptes {

class backtrace_t {
    enum {
        max_backtrace = 32
    };
public:
    backtrace_t();

    backtrace_t(const backtrace_t & rhs);
    backtrace_t & operator=(const backtrace_t & rhs);

    int size() const { return bt_size_; }
    void * const * pointers() const { return bt_; }
    void print() const;
    void refresh();

    static backtrace_t & instance();
private:
    void * bt_[max_backtrace];
    int bt_size_;
}; // end class backtrace_t

} // end namespace panoptes

#endif // __PANOPTES__BACKTRACE_H__
