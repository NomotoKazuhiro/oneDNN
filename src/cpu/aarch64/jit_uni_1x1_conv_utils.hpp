/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_AARCH64_JIT_UNI_1X1_CONV_UTILS_HPP
#define CPU_AARCH64_JIT_UNI_1X1_CONV_UTILS_HPP

#include "common/convolution_pd.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/primitive_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct reduce_to_unit_stride_t {
    convolution_desc_t conv_d_;
    bool reduce_src_;
    size_t space_per_thread_;
};

/* 1x1-kernel does not support non-unit strides so far, so the idea is:
 *  - for fwd or bwd_weights: to copy src to a scratch memory (with strides
 *    equal to 1) and then call the kernel
 *  - for bwd_data: reduce the problem to the one with unit stride by
 *    performing computations in a scratch memory (with strides equal to 1)
 *    and then copy the result to diff_src */
template <typename conv_pd_t>
inline void rtus_prepare(conv_pd_t *self, const convolution_desc_t *&conv_d,
        const memory_desc_t *&src_d, const memory_desc_t *dst_d) {
    const int ndims = src_d->ndims;

    bool rtus_applicable = utils::one_of(ndims, 3, 4);
    if (ndims == 3)
        rtus_applicable = rtus_applicable && conv_d->strides[0] != 1
                && conv_d->src_desc.data_type != data_type::s32;
    else
        rtus_applicable = rtus_applicable
                && (conv_d->strides[0] != 1 || conv_d->strides[1] != 1);
    for (int d = 2; d < ndims; ++d) {
        /* TODO: relax these conditions (by improving reducer) */
        rtus_applicable = rtus_applicable && conv_d->padding[0][d - 2] == 0
                && dst_d->dims[d] * conv_d->strides[d - 2] == src_d->dims[d];
    }
    if (!rtus_applicable) return;

    const auto dat_tag = ndims == 3
            ? memory_desc_wrapper(src_d).matches_one_of_tag(
                    format_tag::nCw8c, format_tag::nCw16c, format_tag::nwc)
            : memory_desc_wrapper(src_d).matches_one_of_tag(
                    format_tag::nChw8c, format_tag::nChw16c, format_tag::nhwc);
    if (dat_tag == format_tag::undef) return;

    const bool is_nspc
            = utils::one_of(dat_tag, format_tag::nwc, format_tag::nhwc);
    if (is_nspc && !mayiuse(sve_256)) return;

#if 1
    // rtus is applicable, configure it.
    self->rtus_.reduce_src_ = true;
    conv_d = &(self->rtus_.conv_d_ = *conv_d);
    self->rtus_.conv_d_.strides[0] = 1;
    if (ndims == 4) self->rtus_.conv_d_.strides[1] = 1;
    utils::array_set(self->rtus_.conv_d_.padding[0], 0, 2);
    if (ndims == 4) utils::array_set(self->rtus_.conv_d_.padding[1], 0, 2);
    const int ic = src_d->dims[1];
    if (self->desc()->prop_kind == prop_kind::backward_data) {
        data_type_t data_type = self->rtus_.conv_d_.diff_src_desc.data_type;
        src_d = &(self->rtus_.conv_d_.diff_src_desc = *dst_d);
        self->rtus_.conv_d_.diff_src_desc.dims[1] = ic;
        self->rtus_.conv_d_.diff_src_desc.data_type = data_type;
        memory_desc_wrapper::compute_blocking(
                self->rtus_.conv_d_.diff_src_desc, dat_tag);
    } else {
        data_type_t data_type = self->rtus_.conv_d_.src_desc.data_type;
        src_d = &(self->rtus_.conv_d_.src_desc = *dst_d);
        self->rtus_.conv_d_.src_desc.dims[1] = ic;
        self->rtus_.conv_d_.src_desc.data_type = data_type;
        memory_desc_wrapper::compute_blocking(
                self->rtus_.conv_d_.src_desc, dat_tag);
    }
#endif
}

template <typename conv_pd_t>
inline void rtus_prepare(conv_pd_t *self, const convolution_desc_t *&conv_d,
        const memory_desc_t *&src_d, const memory_desc_t *dst_d,
        const memory_desc_t *weights_d) {
    const int ndims = src_d->ndims;

    const bool with_groups
            = memory_desc_wrapper(weights_d).ndims() == ndims + 1;

    bool rtus_applicable = utils::one_of(ndims, 3, 4)
            && IMPLICATION(with_groups, weights_d->dims[0] == 1);
    if (ndims == 3)
        rtus_applicable = rtus_applicable && conv_d->strides[0] != 1
                && conv_d->src_desc.data_type != data_type::s32;
    else
        rtus_applicable = rtus_applicable
                && (conv_d->strides[0] != 1 || conv_d->strides[1] != 1);
    for (int d = 2; d < ndims; ++d) {
        /* TODO: relax these conditions (by improving reducer) */
        rtus_applicable = rtus_applicable && conv_d->padding[0][d - 2] == 0
                && dst_d->dims[d] * conv_d->strides[d - 2] == src_d->dims[d];
    }
    if (!rtus_applicable) return;

    const auto dat_tag = ndims == 3
            ? memory_desc_wrapper(src_d).matches_one_of_tag(
                    format_tag::nCw8c, format_tag::nCw16c, format_tag::nwc)
            : memory_desc_wrapper(src_d).matches_one_of_tag(
                    format_tag::nChw8c, format_tag::nChw16c, format_tag::nhwc);
    if (dat_tag == format_tag::undef) return;

    const bool is_nspc
            = utils::one_of(dat_tag, format_tag::nwc, format_tag::nhwc);
    if (is_nspc && !(mayiuse(sve_512))) return;

#if 1
    // rtus is applicable, configure it.
    self->rtus_.reduce_src_ = true;
    conv_d = &(self->rtus_.conv_d_ = *conv_d);
    self->rtus_.conv_d_.strides[0] = 1;
    if (ndims == 4) self->rtus_.conv_d_.strides[1] = 1;
    utils::array_set(self->rtus_.conv_d_.padding[0], 0, 2);
    if (ndims == 4) utils::array_set(self->rtus_.conv_d_.padding[1], 0, 2);
    const int ic = src_d->dims[1];
    if (self->desc()->prop_kind == prop_kind::backward_data) {
        data_type_t data_type = self->rtus_.conv_d_.diff_src_desc.data_type;
        src_d = &(self->rtus_.conv_d_.diff_src_desc = *dst_d);
        self->rtus_.conv_d_.diff_src_desc.dims[1] = ic;
        self->rtus_.conv_d_.diff_src_desc.data_type = data_type;
        memory_desc_wrapper::compute_blocking(
                self->rtus_.conv_d_.diff_src_desc, dat_tag);
    } else {
        data_type_t data_type = self->rtus_.conv_d_.src_desc.data_type;
        src_d = &(self->rtus_.conv_d_.src_desc = *dst_d);
        self->rtus_.conv_d_.src_desc.dims[1] = ic;
        self->rtus_.conv_d_.src_desc.data_type = data_type;
        memory_desc_wrapper::compute_blocking(
                self->rtus_.conv_d_.src_desc, dat_tag);
    }
#endif
}

template <typename conv_pd_t>
inline void rtus_prepare_space_info(conv_pd_t *self,
        memory_tracking::registrar_t &scratchpad, int max_threads) {
    if (!self->rtus_.reduce_src_) return;
    const auto &jcp = self->jcp_;
    const bool is_nspc
            = utils::one_of(jcp.src_tag, format_tag::nhwc, format_tag::nwc);

    const size_t factor = utils::pick_by_prop_kind(self->desc()->prop_kind,
            jcp.nb_reduce, jcp.nb_load_blocking_max, jcp.nb_bcast_blocking);
    size_t typesize
            = types::data_type_size(self->invariant_src_md()->data_type);

    self->rtus_.space_per_thread_
            = is_nspc ? jcp.is * jcp.ic : factor * jcp.is * jcp.ic_block;
    scratchpad.book(memory_tracking::names::key_conv_rtus_space,
            max_threads * self->rtus_.space_per_thread_, typesize);
}

template <cpu_isa_t isa>
struct rtus_driver_t : public jit_generator {

    struct call_params_t {
        const void *ws; /* reduced image (w/ strides = 1) */
        const void *src; /* source image (w/ non-unit strides) */
        size_t icb;
        size_t os;
        size_t iw_start;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(rtus_driver_t)

    Xbyak_aarch64::XReg reg_ws = x5;
    Xbyak_aarch64::XReg reg_src = x6;
    Xbyak_aarch64::XReg reg_icb = x7;
    Xbyak_aarch64::XReg reg_os = x8;
    Xbyak_aarch64::XReg reg_iw_start = x9;

    Xbyak_aarch64::XReg reg_cur_os = x10;
    Xbyak_aarch64::XReg reg_cur_iw = x11;
    Xbyak_aarch64::XReg reg_cur_src = x12;
    Xbyak_aarch64::XReg reg_cur_src_fin = reg_cur_iw; /* just reuse */

    Xbyak_aarch64::PReg tail_mask = p1;

    // nspc section
    Xbyak_aarch64::XReg reg_cur_icb = x13;
    Xbyak_aarch64::XReg reg_tail_mask = x14;
    Xbyak_aarch64::XReg reg_icb_remainder = x15;
    Xbyak_aarch64::XReg reg_ws_copy = x16;
    Xbyak_aarch64::XReg reg_tmp_imm = x17;
    Xbyak_aarch64::XReg reg_tmp = x18;

    Xbyak_aarch64::ZReg reg_zero = Xbyak_aarch64::ZReg(0);
    Xbyak_aarch64::ZReg reg_v = Xbyak_aarch64::ZReg(1);

    int iw_, stride_w_;
    int src_step_h_, src_step_icb_, ws_step_icb_, vlen_, vlen_shift_;
    bool src_to_ws_;
    size_t typesize_;
    int ic_, ic_tail_;
    bool is_nspc_;

    rtus_driver_t(int iw, int stride_w, int src_step_h, int src_step_icb,
            int ws_step_icb, bool src_to_ws, size_t typesize, int ic,
            bool is_nspc = false)
        : iw_(iw)
        , stride_w_(stride_w)
        , src_step_h_(src_step_h)
        , src_step_icb_(src_step_icb)
        , ws_step_icb_(ws_step_icb)
        , src_to_ws_(src_to_ws)
        , typesize_(typesize)
        , ic_(ic)
        , is_nspc_(is_nspc) {
        using namespace Xbyak_aarch64;

        assert(ic_ > 0);

        /*FIXME: derive Vmm type on compile time.
         * changing register type  on runtime
         * seems dangerous,and some xbyak functions might
         * fail to work on reg_v, reg_zero because of this
         * data_type change, e.g. uni_vpxor doen't
         * work on reg_zero now*/
        auto Vmm = [=](int idx, size_t typesize) {
            ZReg res = ZReg(idx);
            if (is_nspc_) {
                switch (isa) {
                    case sve_512: res = ZReg(idx); break;
                    default: assert(!"Not supported isa"); res = ZReg(idx);
                }
                return res;
            }
            switch (isa) {
                case sve_512:
                    switch (typesize) {
                        case 4: res = ZReg(idx); break;
                        //case 2: res = Ymm(idx); break;
                        //case 1: res = Xmm(idx); break;
                        default:
                            assert(!"Not supported typesize");
                            res = ZReg(idx);
                    }
            }
            return res;
        };

        reg_zero = Vmm(0, typesize);
        reg_v = Vmm(1, typesize);

        vlen_ = reg_v.getBit() / 8;
        vlen_shift_ = 0;

        int tvlen = is_nspc_ ? typesize_ : vlen_;
        while (tvlen > 1) {
            tvlen /= 2;
            vlen_shift_++;
        }

        const int simd_w = vlen_ / sizeof(float);
        ic_tail_ = ic_ % simd_w;
    }

    void loop_is() {
        using namespace Xbyak_aarch64;

        xa_->mov(reg_cur_src, reg_src);
        xa_->mov(reg_cur_iw, reg_iw_start);
        xa_->mov(reg_cur_os, reg_os);

        Label is_loop;
        L(is_loop);

        if (src_to_ws_) {
            xa_->ldr(reg_v, Xbyak_aarch64::ptr(reg_cur_src));
            xa_->str(reg_v, Xbyak_aarch64::ptr(reg_ws));
        } else {
            xa_->ldr(reg_v, Xbyak_aarch64::ptr(reg_ws));
            xa_->str(reg_v, Xbyak_aarch64::ptr(reg_cur_src));
            for (int w = 1; w < stride_w_; ++w) {
                xa_->add_imm(reg_tmp, reg_cur_src, w * vlen_, reg_tmp_imm);
                xa_->str(reg_zero, Xbyak_aarch64::ptr(reg_tmp));
            }
        }

        xa_->add_imm(reg_ws, reg_ws, vlen_, reg_tmp_imm);
        xa_->add_imm(reg_cur_src, reg_cur_src, stride_w_ * vlen_, reg_tmp_imm);

        // for 1d or stride_h=1 convolutions the loop over h should be skipped
        if (!(src_step_icb_ == iw_ || src_step_h_ == iw_)) {
            Label skip_h_step;
            xa_->add_imm(reg_cur_iw, reg_cur_iw, stride_w_, reg_tmp_imm);
            xa_->cmp(reg_cur_iw, iw_);
            xa_->b(LT, skip_h_step); //jl(skip_h_step);

            if (src_to_ws_) {
                xa_->add_imm(reg_cur_src, reg_cur_src,
                        (src_step_h_ - iw_) * vlen_, reg_tmp_imm);
            } else {
                xa_->mov(reg_cur_src_fin, reg_cur_src);
                xa_->add_imm(reg_cur_src_fin, reg_cur_src_fin,
                        (src_step_h_ - iw_) * vlen_, reg_tmp_imm);
                Label ih_loop;
                L(ih_loop);

                for (int w = 0; w < stride_w_; ++w) {
                    xa_->add_imm(reg_tmp, reg_cur_src, w * vlen_, reg_tmp_imm);
                    xa_->str(reg_zero, Xbyak_aarch64::ptr(reg_tmp));
                }

                xa_->add_imm(reg_cur_src, reg_cur_src, stride_w_ * vlen_,
                        reg_tmp_imm);
                xa_->cmp(reg_cur_src, reg_cur_src_fin);
                xa_->b(LT, ih_loop); //jl(ih_loop);
            }
            xa_->mov(reg_cur_iw, 0); //xor_(reg_cur_iw, reg_cur_iw);
            L(skip_h_step);
        }

        xa_->subs_imm(reg_cur_os, reg_cur_os, vlen_, reg_tmp_imm);
        xa_->b(NE, is_loop); //jnz(is_loop);

        /* restore dst */
        xa_->sub(reg_ws, reg_ws, reg_os);
    }

    void loop_is_nspc() {
#if 1
        using namespace Xbyak_aarch64;

        xa_->mov(reg_cur_src, reg_src);
        xa_->mov(reg_cur_iw, reg_iw_start);

        if (isa == sve_512) {
            //            push(rcx); // preserve rcx, used for shift
            xa_->mov(reg_icb_remainder, reg_icb);
            xa_->and_(reg_icb_remainder, reg_icb_remainder,
                    (vlen_ / typesize_) - 1); // # of elements in tail
            xa_->mov(reg_tail_mask, 1);
            xa_->lsl(reg_tail_mask, reg_tail_mask,
                    reg_icb_remainder); // shl(reg_tail_mask, reg_icb_remainder.cvt8());
            xa_->subs(reg_tail_mask, reg_tail_mask, 1); // dec(reg_tail_mask);
            //            pop(rcx);

            switch (typesize_) {
                    //                case 4: kmovw(tail_mask, reg_tail_mask.cvt32()); break;
                    //                case 2: kmovd(tail_mask, reg_tail_mask.cvt32()); break;
                case 1: // kmovq(tail_mask, reg_tail_mask); break;
                    xa_->sub(X_SP, X_SP, 8);
                    xa_->str(reg_tail_mask, Xbyak_aarch64::ptr(X_SP));
                    xa_->ldr(tail_mask, Xbyak_aarch64::ptr(X_SP));
                    xa_->add(X_SP, X_SP, 8);
                    break;
                default: assert(!"Unsupported typesize");
            }
        }

        auto load_reg = [=](const ZReg &vreg, const XReg &reg,
                                const int64_t offset, const int load_size,
                                const PReg &tail_mask) {
            if (isa == sve_512) {
                //                const Address &addr = ptr[reg + offset];
                switch (typesize_) {
                        //                    case 4: vmovups(vreg, addr); break;
                        //                    case 2: vmovdqu16(vreg, addr); break;
                    case 1: // vmovdqu8(vreg, addr); break;
                        xa_->add_imm(X_TMP_1, reg, offset, X_TMP_0);
                        xa_->ld1b(vreg.b, tail_mask / Xbyak_aarch64::T_z,
                                Xbyak_aarch64::ptr(X_TMP_1));
                        break;
                    default: assert(!"Unsupported typesize");
                }
            } else {
#if 1
                assert(!"Unsupported load_bytes");
#else
                load_bytes(vreg, reg, offset, load_size);
#endif
            }
        };

        auto store_reg = [=](const XReg &reg, const ZReg &vreg,
                                 const int64_t offset, const int store_size,
                                 const PReg &tail_mask) {
            if (isa == sve_512) {
                //                const Address &addr = ptr[reg + offset];
                switch (typesize_) {
                        //                    case 4: vmovups(addr, vreg); break;
                        //                    case 2: vmovdqu16(addr, vreg); break;
                    case 1: // vmovdqu8(addr, vreg); break;
                        xa_->add_imm(X_TMP_1, reg, offset, X_TMP_0);
                        xa_->st1b(
                                vreg.b, tail_mask, Xbyak_aarch64::ptr(X_TMP_1));
                        break;
                    default: assert(!"Unsupported typesize");
                }
            } else {
#if 1
                assert(!"Unsupported store_bytes");
#else
                store_bytes(vreg, reg, offset, store_size);
#endif
            }
        };

        xa_->mov(P_TMP.b, P_ALL_ONE.b);
        xa_->mov(reg_ws_copy, reg_ws);
        xa_->lsl(reg_icb, reg_icb, vlen_shift_); // shl(reg_icb, vlen_shift_);

        const size_t w_step_factor = ic_ * typesize_;
        const size_t max_load_store_bytes = typesize_ == 4 ? 32 : 16;
        const size_t load_store_size
                = isa == sve_512 ? vlen_ : max_load_store_bytes;
        size_t load_store_tail_size = (typesize_ == 1 ? max_load_store_bytes
                                                      : ic_tail_ * typesize_);

        Label is_loop, ic_loop, ic_loop_tail, ic_loop_finish;
        L(is_loop);
        {
            xa_->mov(reg_cur_src, reg_src);
            xa_->mov(reg_ws, reg_ws_copy);
            xa_->mov(reg_cur_icb, reg_icb);

            L(ic_loop);
            {
                xa_->cmp(reg_cur_icb, load_store_size);
                xa_->b(LT, ic_loop_tail); // jl(ic_loop_tail);

                if (src_to_ws_) {
                    load_reg(reg_v, reg_cur_src, 0, load_store_size, P_TMP);
                    store_reg(reg_ws, reg_v, 0, load_store_size, P_TMP);
                } else {
                    load_reg(reg_v, reg_ws, 0, load_store_size, P_TMP);
                    store_reg(reg_cur_src, reg_v, 0, load_store_size, P_TMP);
                    for (int w = 1; w < stride_w_; ++w)
                        store_reg(reg_cur_src, reg_zero, w * w_step_factor,
                                load_store_size, P_TMP);
                }
                xa_->add_imm(reg_ws, reg_ws, load_store_size, reg_tmp_imm);
                xa_->add_imm(
                        reg_cur_src, reg_cur_src, load_store_size, reg_tmp_imm);

                xa_->sub_imm(
                        reg_cur_icb, reg_cur_icb, load_store_size, reg_tmp_imm);
                xa_->b(ic_loop); // jmp(ic_loop);
            }

            L(ic_loop_tail);
            {
                xa_->cmp(reg_cur_icb, 0);
                xa_->b(EQ, ic_loop_finish); // je(ic_loop_finish);

                if (src_to_ws_) {
                    load_reg(reg_v, reg_cur_src, 0, load_store_tail_size,
                            tail_mask);
                    store_reg(
                            reg_ws, reg_v, 0, load_store_tail_size, tail_mask);
                } else {
                    load_reg(reg_v, reg_ws, 0, load_store_tail_size, tail_mask);
                    store_reg(reg_cur_src, reg_v, 0, load_store_tail_size,
                            tail_mask);
                    for (int w = 1; w < stride_w_; ++w)
                        store_reg(reg_cur_src, reg_zero, w * w_step_factor,
                                load_store_tail_size, tail_mask);
                }
            }
            L(ic_loop_finish);

            xa_->add_imm(reg_ws_copy, reg_ws_copy, w_step_factor, reg_tmp_imm);
            xa_->add_imm(
                    reg_src, reg_src, stride_w_ * w_step_factor, reg_tmp_imm);

            // for 1d or stride_h=1 convolutions the loop over h should be skipped
            const bool skip_oh_step = src_step_h_ == iw_;
            if (!skip_oh_step) {
                xa_->mov(reg_cur_src, reg_src);
                Label skip_h_step;
                xa_->add_imm(reg_cur_iw, reg_cur_iw, stride_w_, reg_tmp_imm);
                xa_->cmp(reg_cur_iw, iw_);
                xa_->b(LT, skip_h_step); // jl(skip_h_step, T_NEAR);

                if (src_to_ws_) {
                    xa_->add_imm(reg_src, reg_src,
                            (src_step_h_ - iw_) * w_step_factor, reg_tmp_imm);
                } else {
                    xa_->mov(reg_cur_src_fin, reg_cur_src);
                    xa_->add_imm(reg_cur_src_fin, reg_cur_src_fin,
                            (src_step_h_ - iw_) * w_step_factor, reg_tmp_imm);
                    Label ih_loop_nhwc, ic_ih_loop_nhwc, ic_tail_ih_loop_nhwc,
                            ic_finish_ih_loop_nhwc;
                    L(ih_loop_nhwc);
                    xa_->mov(reg_cur_src, reg_src);
                    xa_->mov(reg_cur_icb, reg_icb);
                    L(ic_ih_loop_nhwc);
                    xa_->cmp(reg_cur_icb, load_store_size);
                    xa_->b(LT,
                            ic_tail_ih_loop_nhwc); // jl(ic_tail_ih_loop_nhwc);

                    for (int w = 0; w < stride_w_; ++w)
                        store_reg(reg_cur_src, reg_zero, w * w_step_factor,
                                load_store_size, P_TMP);

                    xa_->add_imm(reg_cur_src, reg_cur_src, load_store_size,
                            reg_tmp_imm);
                    xa_->sub_imm(reg_cur_icb, reg_cur_icb, load_store_size,
                            reg_tmp_imm);
                    xa_->b(NE, ic_ih_loop_nhwc); // jnz(ic_ih_loop_nhwc);

                    L(ic_tail_ih_loop_nhwc);
                    xa_->cmp(reg_cur_icb, 0);
                    xa_->b(LE,
                            ic_finish_ih_loop_nhwc); // jle(ic_finish_ih_loop_nhwc);

                    for (int w = 0; w < stride_w_; ++w)
                        store_reg(reg_cur_src, reg_zero, w * w_step_factor,
                                load_store_tail_size, tail_mask);

                    L(ic_finish_ih_loop_nhwc);

                    xa_->add_imm(reg_src, reg_src, stride_w_ * w_step_factor,
                            reg_tmp_imm);
                    xa_->cmp(reg_src, reg_cur_src_fin);
                    xa_->b(LT, ih_loop_nhwc); // jl(ih_loop_nhwc);
                }
                xa_->mov(reg_cur_iw, 0); // xor_(reg_cur_iw, reg_cur_iw);
                L(skip_h_step);
            }

            xa_->subs(reg_os, reg_os, 1);
            xa_->b(NE, is_loop); // jnz(is_loop);
        }
#endif
    }

    void generate() {
#if 1
        using namespace Xbyak_aarch64;
        assert(isa == sve_512);

        preamble();
#define READ_PARAM(what) \
    xa_->ldr(reg_##what, \
            Xbyak_aarch64::ptr(abi_param1, \
                    static_cast<int32_t>(offsetof(call_params_t, what))))
        READ_PARAM(src);
        READ_PARAM(icb);
        READ_PARAM(os);
        READ_PARAM(iw_start);
        READ_PARAM(ws);
#undef READ_PARAM

        if (!src_to_ws_) {
            switch (reg_zero.getBit() / 8) {
                //case 16 /*xmm*/: uni_vpxor(reg_zero, reg_zero, reg_zero); break;
                //case 32 /*ymm*/: {
                //    Xbyak::Ymm ymm_z(reg_zero.getIdx());
                //    uni_vpxor(ymm_z, ymm_z, ymm_z);
                //    break;
                //}
                case 64 /*zmm*/: {
                    Xbyak_aarch64::ZRegS zreg_s(reg_zero.getIdx());
                    xa_->fmov(zreg_s); // zero clear
                    break;
                }
                default: assert(!"rtus kernel failure");
            }
        }
        if (is_nspc_) {
            loop_is_nspc();
        } else {
            lsl(reg_os, reg_os, vlen_shift_);

            Label icb_loop;
            L(icb_loop);

            loop_is();

            xa_->add_imm(reg_ws, reg_ws, ws_step_icb_ * vlen_, reg_tmp_imm);
            xa_->add_imm(reg_src, reg_src, src_step_icb_ * vlen_, reg_tmp_imm);

            xa_->subs_imm(reg_icb, reg_icb, vlen_ / typesize_, reg_tmp_imm);
            xa_->b(NE, icb_loop); //jnz(icb_loop, T_NEAR);
        }

        //uni_vzeroupper();
        postamble();
#endif
    }
};

template <cpu_isa_t isa, typename conv_t>
inline status_t init_rtus_driver(conv_t *self) {
    const auto &conf = *self->pd();
    if (!conf.rtus_.reduce_src_) return status::success;

    const auto &cd = *conf.desc();
    const int ndims = conf.ndims();
    const int stride_h = (conf.ndims() == 3) ? 1 : cd.strides[0];
    const int stride_w = cd.strides[ndims - 3];

    const bool is_bwd_data = cd.prop_kind == prop_kind::backward_data;
    const auto &src_d = is_bwd_data ? *conf.diff_src_md() : *conf.src_md();

    const int ih = ndims == 3 ? 1 : src_d.dims[2];
    const int iw = src_d.dims[ndims - 1];
    const int ic = src_d.dims[1];

    const auto src_tag = memory_desc_wrapper(src_d).matches_one_of_tag(
            format_tag::nhwc, format_tag::nwc);
    const bool is_nspc = src_tag != format_tag::undef;
    const int src_step_h = stride_h * iw;
    const int src_step_icb = !is_nspc ? ih * iw : 1;
    const int ws_step_icb = !is_nspc ? conf.jcp_.is : 1;
    const bool src_to_ws = !is_bwd_data;
    const size_t typesize
            = types::data_type_size(self->pd()->invariant_src_md()->data_type);

    CHECK(safe_ptr_assign(self->rtus_driver_,
            new rtus_driver_t<isa>(iw, stride_w, src_step_h, src_step_icb,
                    ws_step_icb, src_to_ws, typesize, ic, is_nspc)));
    return self->rtus_driver_->create_kernel();
}

inline int best_divider(int value, int min_divider, int max_divider,
        bool find_max, int step = 1) {
    using namespace dnnl::impl::utils;
    max_divider = nstl::max(1, nstl::min(max_divider, value));
    min_divider = nstl::max(1, nstl::min(min_divider, max_divider));

    auto loss_ratio = [](int total, int chunk) {
        return float(rnd_up(total, chunk) - total) / rnd_up(total, chunk);
    };

    float min_loss = FLT_MAX;
    int x_divider = max_divider;
    for (int divider = max_divider; divider >= min_divider; divider -= step) {
        const float loss = loss_ratio(value, divider);
        if ((find_max && loss < min_loss) || (!find_max && loss <= min_loss)) {
            min_loss = loss;
            x_divider = divider;
        }
    }
    return x_divider;
}

typedef jit_1x1_conv_conf_t jcp_t;

inline bool is_bcast_layout_nxc(const jcp_t &jcp) {
    switch (jcp.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
        case prop_kind::backward_weights:
            return utils::one_of(jcp.src_tag, format_tag::ndhwc,
                    format_tag::nhwc, format_tag::nwc);
        case prop_kind::backward_data:
            return utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                    format_tag::nhwc, format_tag::nwc);
        default: assert(!"invalid prop_kind"); return false;
    }
}

inline bool is_load_layout_nxc(const jcp_t &jcp) {
    return jcp.prop_kind == prop_kind::backward_weights
            && utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                    format_tag::nwc);
}

inline bool is_out_layout_nxc(const jcp_t &jcp) {
    switch (jcp.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            return utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                    format_tag::nhwc, format_tag::nwc);
        case prop_kind::backward_data:
            return utils::one_of(jcp.src_tag, format_tag::ndhwc,
                    format_tag::nhwc, format_tag::nwc);
        case prop_kind::backward_weights: return false;
        default: assert(!"invalid prop_kind"); return false;
    }
}

inline size_t get_bcast_u_offset(const jcp_t &jcp) {
    return is_bcast_layout_nxc(jcp) ? jcp.ic : jcp.ic_block;
}

inline size_t get_bcast_j_offset(const jcp_t &jcp) {
    return is_bcast_layout_nxc(jcp) ? jcp.reduce_dim : jcp.reduce_loop_unroll;
}

inline size_t get_bcast_offset(const jcp_t &jcp, int u, int j) {
    size_t offset;
    if (utils::one_of(jcp.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference, prop_kind::backward_data)) {
        assert(jcp.reduce_loop_unroll == jcp.reduce_block);
        if (is_bcast_layout_nxc(jcp) || u != jcp.reduce_loop_unroll) {
            offset = j * get_bcast_j_offset(jcp) + u;
        } else {
            offset = (jcp.bcast_dim + j) * get_bcast_j_offset(jcp);
        }
    } else {
        offset = u * get_bcast_u_offset(jcp) + j;
    }
    return sizeof(float) * offset;
}

inline size_t get_load_u_offset(const jcp_t &jcp) {
    return is_load_layout_nxc(jcp) ? jcp.oc : jcp.oc_block;
}

inline size_t get_load_i_offset(const jcp_t &jcp) {
    return is_load_layout_nxc(jcp) ? jcp.oc_block : jcp.os;
}

inline size_t get_load_bwd_w_offset(const jcp_t &jcp, int i, int u0) {
    if (is_load_layout_nxc(jcp)) {
        return i * get_load_i_offset(jcp) + u0 * get_load_u_offset(jcp);
    } else {
        return (i * get_load_i_offset(jcp) + u0) * get_load_u_offset(jcp);
    }
}

inline size_t get_output_i_offset(const jcp_t &jcp) {
    if (is_out_layout_nxc(jcp)) {
        return jcp.load_block;
    } else {
        return (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) * jcp.load_block;
    }
}

inline size_t get_output_j_offset(const jcp_t &jcp) {
    return is_out_layout_nxc(jcp) ? jcp.load_dim : jcp.load_block;
}

inline size_t get_load_loop_output_fwd_offset(
        const jcp_t &jcp, int load_loop_blk) {
    size_t offset = load_loop_blk * jcp.oc_block * sizeof(float);
    if (!is_out_layout_nxc(jcp)) {
        offset *= jcp.with_dw_conv ? jcp.ow : jcp.os;
    }
    return offset;
}

inline size_t get_load_loop_output_bwd_d_offset(
        const jcp_t &jcp, int load_loop_blk) {
    size_t offset = load_loop_blk * jcp.ic_block * sizeof(float);
    if (!is_out_layout_nxc(jcp)) { offset *= jcp.os; }
    return offset;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
