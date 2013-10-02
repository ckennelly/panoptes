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

#include <cassert>
#include <cstdio>
#include <panoptes/memcheck/context_memcheck_internal.h>
#include <ptx_io/ptx_ir.h>
#include <cstring>

using namespace panoptes;
using namespace panoptes::internal;

static std::string make_temp_identifier(type_t type, unsigned id) {
    const char * type_name = NULL;

    switch (type) {
        case b8_type:   type_name = "b8"; break;
        case s8_type:   type_name = "s8"; break;
        case u8_type:   type_name = "u8"; break;
        case b16_type:  type_name = "b16"; break;
        case f16_type:  type_name = "f16"; break;
        case s16_type:  type_name = "s16"; break;
        case u16_type:  type_name = "u16"; break;
        case b32_type:  type_name = "b32"; break;
        case f32_type:  type_name = "f32"; break;
        case s32_type:  type_name = "s32"; break;
        case u32_type:  type_name = "u32"; break;
        case b64_type:  type_name = "b64"; break;
        case f64_type:  type_name = "f64"; break;
        case s64_type:  type_name = "s64"; break;
        case u64_type:  type_name = "u64"; break;
        case pred_type: type_name = "pred"; break;
        case texref_type:
        case invalid_type:
            assert(0 && "Unsupported type.");
            break;
    }

    assert(type_name);

    char buf[24];
    int ret = snprintf(buf, sizeof(buf), "__panoptes_%s_%u",
        type_name, id);
    assert(ret < (int) sizeof(buf));

    return buf;
}

temp_operand::temp_operand(auxillary_t * parent, type_t type) :
        parent_(*parent), type_(type), id_(parent_.allocate(type)),
        identifier_(make_temp_identifier(type, id_)),
        operand_(operand_t::make_identifier(identifier_)) { }

temp_operand::~temp_operand() {
    parent_.deallocate(type_, id_);
}

temp_operand::operator const std::string &() const {
    return identifier_;
}

temp_operand::operator const operand_t &() const {
    return operand_;
}

bool temp_operand::operator!=(const operand_t & rhs) const {
    return operand_ != rhs;
}

static std::string make_temp_pointer(unsigned id) {
    char buf[24];
    int ret = snprintf(buf, sizeof(buf), "__panoptes_ptr%u", id);
    assert(ret < (int) sizeof(buf));

    return buf;
}

temp_ptr::temp_ptr(auxillary_t * parent) :
        parent_(*parent), id_(parent_.allocate_ptr()),
        identifier_(make_temp_pointer(id_)),
        operand_(operand_t::make_identifier(identifier_)) { }

temp_ptr::~temp_ptr() {
    parent_.deallocate_ptr(id_);
}

temp_ptr::operator const std::string &() const {
    return identifier_;
}

temp_ptr::operator const operand_t &() const {
    return operand_;
}

bool temp_ptr::operator!=(const operand_t & rhs) const {
    return operand_ != rhs;
}

auxillary_t::auxillary_t() : pred(0), ptr(0), pred_(0), ptr_(0) {
    memset(b, 0, sizeof(b));
    memset(s, 0, sizeof(s));
    memset(u, 0, sizeof(u));
    memset(b_, 0, sizeof(b_));
    memset(s_, 0, sizeof(s_));
    memset(u_, 0, sizeof(u_));
}

unsigned auxillary_t::allocate(type_t type) {
    unsigned ret;
    switch (type) {
        case b8_type:
            ret = b_[0]++;
            b[0] = std::max(b[0], b_[0]);
            break;
        case f16_type:
        case b16_type:
            ret = b_[1]++;
            b[1] = std::max(b[1], b_[1]);
            break;
        case f32_type:
        case b32_type:
            ret = b_[2]++;
            b[2] = std::max(b[2], b_[2]);
            break;
        case f64_type:
        case b64_type:
            ret = b_[3]++;
            b[3] = std::max(b[3], b_[3]);
            break;
        case s8_type:
            ret = s_[0]++;
            s[0] = std::max(s[0], s_[0]);
            break;
        case s16_type:
            ret = s_[1]++;
            s[1] = std::max(s[1], s_[1]);
            break;
        case s32_type:
            ret = s_[2]++;
            s[2] = std::max(s[2], s_[2]);
            break;
        case s64_type:
            ret = s_[3]++;
            s[3] = std::max(s[3], s_[3]);
            break;
        case u8_type:
            ret = u_[0]++;
            u[0] = std::max(u[0], u_[0]);
            break;
        case u16_type:
            ret = u_[1]++;
            u[1] = std::max(u[1], u_[1]);
            break;
        case u32_type:
            ret = u_[2]++;
            u[2] = std::max(u[2], u_[2]);
            break;
        case u64_type:
            ret = u_[3]++;
            u[3] = std::max(u[3], u_[3]);
            break;
        case pred_type:
            ret = pred_++;
            pred = std::max(pred, pred_);
            break;
        case texref_type:
        case invalid_type:
            assert(0 && "Invalid type.");
    }

    return ret;
}

unsigned auxillary_t::allocate_ptr() {
    unsigned ret = ptr_++;
    ptr = std::max(ptr, ptr_);
    return ret;
}

void auxillary_t::deallocate(type_t type, unsigned id) {
    unsigned * addr = NULL;

    switch (type) {
        case b8_type:
            addr = &b_[0];
            break;
        case f16_type:
        case b16_type:
            addr = &b_[1];
            break;
        case f32_type:
        case b32_type:
            addr = &b_[2];
            break;
        case f64_type:
        case b64_type:
            addr = &b_[3];
            break;
        case s8_type:
            addr = &s_[0];
            break;
        case s16_type:
            addr = &s_[1];
            break;
        case s32_type:
            addr = &s_[2];
            break;
        case s64_type:
            addr = &s_[3];
            break;
        case u8_type:
            addr = &u_[0];
            break;
        case u16_type:
            addr = &u_[1];
            break;
        case u32_type:
            addr = &u_[2];
            break;
        case u64_type:
            addr = &u_[3];
            break;
        case pred_type:
            addr = &pred_;
            break;
        case texref_type:
        case invalid_type:
            assert(0 && "Invalid type.");
    }

    assert(addr);
    assert(*addr > 0);
    (*addr)--;
    assert(*addr == id);
}

void auxillary_t::deallocate_ptr(unsigned id) {
    assert(ptr_ > 0);
    ptr_--;
    assert(ptr_ == id);
}

