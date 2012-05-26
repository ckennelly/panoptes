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

#ifndef __PANOPTES__PTX_FORMATTER_H_
#define __PANOPTES__PTX_FORMATTER_H_

#include <ostream>
#include "ptx_ir.h"

std::ostream & operator<<(std::ostream &, const panoptes::approximation_t &);
std::ostream & operator<<(std::ostream &, const panoptes::atom_op_t &);
std::ostream & operator<<(std::ostream &, const panoptes::barrier_op_t &);
std::ostream & operator<<(std::ostream &, const panoptes::barrier_scope_t &);
std::ostream & operator<<(std::ostream &, const panoptes::type_t &);
std::ostream & operator<<(std::ostream &, const panoptes::vector_t &);
std::ostream & operator<<(std::ostream &, const panoptes::space_t &);
std::ostream & operator<<(std::ostream &, const panoptes::statement_t &);
std::ostream & operator<<(std::ostream &, const panoptes::variable_t &);
std::ostream & operator<<(std::ostream &, const panoptes::block_t &);
std::ostream & operator<<(std::ostream &, const panoptes::cache_t &);
std::ostream & operator<<(std::ostream &, const panoptes::geom_t &);
std::ostream & operator<<(std::ostream &, const panoptes::texture_t &);
std::ostream & operator<<(std::ostream &, const panoptes::label_t &);
std::ostream & operator<<(std::ostream &, const panoptes::linkage_t &);
std::ostream & operator<<(std::ostream &, const panoptes::scope_t &);
std::ostream & operator<<(std::ostream &, const panoptes::param_t &);
std::ostream & operator<<(std::ostream &, const panoptes::function_t &);
std::ostream & operator<<(std::ostream &, const panoptes::op_t &);
std::ostream & operator<<(std::ostream &, const panoptes::field_t &);
std::ostream & operator<<(std::ostream &, const panoptes::op_set_cmp_t &);
std::ostream & operator<<(std::ostream &, const panoptes::op_set_bool_t &);
std::ostream & operator<<(std::ostream &, const panoptes::operand_t &);
std::ostream & operator<<(std::ostream &, const panoptes::prmt_mode_t &);
std::ostream & operator<<(std::ostream &, const panoptes::ptx_t &);
std::ostream & operator<<(std::ostream &, const panoptes::rounding_t &);
std::ostream & operator<<(std::ostream &, const panoptes::sm_t &);
std::ostream & operator<<(std::ostream &, const panoptes::testp_op_t &);
std::ostream & operator<<(std::ostream &, const panoptes::op_mul_width_t &);
std::ostream & operator<<(std::ostream &, const panoptes::variant_t &);
std::ostream & operator<<(std::ostream &, const panoptes::vote_mode_t &);

#endif // __PANOPTES__PTX_FORMATTER_H_
