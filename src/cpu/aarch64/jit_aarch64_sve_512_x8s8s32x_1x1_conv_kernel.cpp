/*******************************************************************************
* Copyright 2019-2020 FUJITSU LIMITED
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

/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/aarch64/jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel.hpp"
#include "cpu/aarch64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) \
    static_cast<int32_t>(offsetof(jit_1x1_conv_call_s, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::data_type;
using namespace Xbyak;

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

#define SVE_compress_addr(base, offt) xa::ptr(get_comp_addr_reg(base, offt))

template <typename Vmm>
bool _jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel<Vmm>::maybe_eltwise(
        int position) {
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* eltwise before sum */
        return p.contain(eltwise, 0);
    } else if (position == 1) {
        /* eltwise after sum */
        return p.contain(sum, 0) && p.contain(eltwise, 1);
    }

    return false;
}

template <typename Vmm>
void _jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel<Vmm>::bcast_loop(
        int load_loop_blk) {
    CGA64::mov(aux1_reg_bcast_data, reg_bcast_data);
    CGA64::mov(aux_reg_bcast_data, reg_bcast_data);

    CGA64::mov(aux_reg_output_data, reg_output_data);
    CGA64::ldr(
            bcast_loop_iter, SVE_compress_addr(reg_rsp, bcast_loop_work_off));

    xa::LabelAArch64 bcast_loop;
    xa::LabelAArch64 bcast_loop_tail;

    CGA64::cmp(bcast_loop_iter, jcp.ur);
    CGA64::b(xa::LT, bcast_loop_tail);

    CGA64::L_aarch64(bcast_loop);
    {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                CGA64::add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_substep, reg_tmp0_imm);
                CGA64::add_imm(aux_reg_output_data, aux_reg_output_data,
                        jcp.bcast_loop_output_substep, reg_tmp0_imm);
            } else {
                CGA64::add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_bcast_substep,
                        reg_tmp0_imm);
                int output_offset = jcp.bcast_loop_output_step
                        - (num_substeps - 1) * jcp.bcast_loop_output_substep;

                CGA64::add_imm(aux_reg_output_data, aux_reg_output_data,
                        output_offset, reg_tmp0_imm);
            }
        }
        CGA64::subs(bcast_loop_iter, bcast_loop_iter, jcp.bcast_block);
        CGA64::cmp(bcast_loop_iter, jcp.bcast_block);
        CGA64::b(xa::GE, bcast_loop);
    }

    CGA64::L_aarch64(bcast_loop_tail);
    if (jcp.ur_tail) {
        xa::LabelAArch64 bcast_loop_tail_out;
        CGA64::cmp(bcast_loop_iter, 0);
        CGA64::b(xa::EQ, bcast_loop_tail_out);
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        CGA64::L_aarch64(bcast_loop_tail_out);
    }
}

//template <typename Vmm>
//void _jit_avx512_x8s8s32x_1x1_conv_kernel<Vmm>::cvt2ps(data_type_t type_in,
//        const Vmm vmm_in, const Xbyak::Operand &op, bool mask_flag) {
//    const Vmm vmm = mask_flag ? vmm_in | ktail_mask | T_z : vmm_in;
//    switch (type_in) {
//        case data_type::f32:
//        case data_type::s32: vmovups(vmm, op); break;
//        case data_type::s8: vpmovsxbd(vmm, op); break;
//        case data_type::u8: vpmovzxbd(vmm, op); break;
//        default: assert(!"unsupported data type");
//    }
//    if (type_in != data_type::f32) vcvtdq2ps(vmm_in, vmm_in);
//}

template <typename Vmm>
void _jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel<Vmm>::reduce_loop(
        int load_loop_blk, int ur, int substep, bool wraparound) {
    auto vreg_load
            = [=](int i_load) { return xa::ZReg(ur * load_loop_blk + i_load); };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return xa::ZReg(i_ur * load_loop_blk + i_load);
    };

    auto bias_ptr = [=](xa::ZReg bias_reg, int i_load, bool mask_flag) {
        int offt = get_offset(jcp.typesize_bia * jcp.oc_block * i_load);

        CGA64::add_imm(reg_tmp0_adr, reg_bias_data, offt, reg_tmp0_imm);
        if (mask_flag)
            CGA64::ld1w(
                    bias_reg.s, ktail_mask / xa::T_z, xa::ptr(reg_tmp0_adr));
        else
            CGA64::ldr(bias_reg, xa::ptr(reg_tmp0_adr));
    };

    auto bias_ptr8 = [=](xa::ZReg bias_reg, int i_load, bool mask_flag) {
        int offt = get_offset(jcp.typesize_bia * jcp.oc_block * i_load);

        CGA64::add_imm(reg_tmp0_adr, reg_bias_data, offt, reg_tmp0_imm);
        if (mask_flag) {
            CGA64::uzp1(ktail_load_mask.h, ktail_mask.h, mask_all_zero.h);
            CGA64::uzp1(ktail_load_mask.b, ktail_load_mask.b, mask_all_zero.b);
            CGA64::ld1b(bias_reg.b, ktail_load_mask / xa::T_z,
                    xa::ptr(reg_tmp0_adr));
        } else {
            CGA64::ldr(xa::QReg(bias_reg.getIdx()), xa::ptr(reg_tmp0_adr));
        }
    };

    auto comp_ptr = [=](xa::ZReg comp_reg, int i_load, bool mask_flag) {
        int offt = get_offset(sizeof(int32_t) * jcp.oc_block * i_load);

        CGA64::add_imm(reg_tmp0_adr, reg_comp_data, offt, reg_tmp0_imm);
        if (mask_flag)
            CGA64::ld1w(
                    comp_reg.s, ktail_mask / xa::T_z, xa::ptr(reg_tmp0_adr));
        else
            CGA64::ldr(comp_reg, xa::ptr(reg_tmp0_adr));
    };

    auto scale_ptr = [=](xa::ZReg scale_reg, int i_load) {
        int ofs = get_offset(
                jcp.is_oc_scale * (sizeof(float) * jcp.oc_block * i_load));

        if (ofs == 0) {
            CGA64::ldr(scale_reg, xa::ptr(reg_ptr_scales));
        } else {
            auto reg_tmp_adr = ((i_load % 4) == 0) ? reg_tmp0_adr
                                                   : ((i_load % 4) == 1)
                            ? reg_tmp1_adr
                            : ((i_load % 4) == 2) ? reg_tmp2_adr : reg_tmp3_adr;
            auto reg_tmp_imm = ((i_load % 4) == 0) ? reg_tmp0_imm
                                                   : ((i_load % 4) == 1)
                            ? reg_tmp1_imm
                            : ((i_load % 4) == 2) ? reg_tmp2_imm : reg_tmp3_imm;
            CGA64::add_imm(reg_tmp_adr, reg_ptr_scales, ofs, reg_tmp_imm);
            CGA64::ldr(scale_reg, xa::ptr(reg_tmp_adr));
        }
    };

    auto bcast_ptr = [=](xa::ZReg bcast_reg, int i_reduce, int i_ur,
                             bool bcast) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        assert(jcp.reduce_loop_unroll == jcp.reduce_block);

        int _offt = (jcp.ic_without_padding * i_ur * jcp.ngroups + i_reduce);

        //        return EVEX_compress_addr(
        //                aux_reg_bcast_data, jcp.typesize_in * offt, bcast);

        auto base = aux_reg_bcast_data;
        auto ofs = get_offset(jcp.typesize_in * _offt);

        if (bcast)
            assert(!"unimplemented");
        else {
            if ((-0x40 <= ofs) && (ofs < 0x40) && ((ofs % 4) == 0))
                CGA64::ld1rw(xa::ZRegS(bcast_reg.getIdx()),
                        xa::PReg(vmask.getIdx()),
                        xa::ptr(xa::XReg(base.getIdx()),
                                static_cast<int32_t>(ofs)));
            else {
                auto reg_tmp_adr = ((i_ur % 4) == 0)
                        ? reg_tmp0_adr
                        : ((i_ur % 4) == 1) ? reg_tmp1_adr
                                            : ((i_ur % 4) == 2) ? reg_tmp2_adr
                                                                : reg_tmp3_adr;
                auto reg_tmp_imm = ((i_ur % 4) == 0)
                        ? reg_tmp0_imm
                        : ((i_ur % 4) == 1) ? reg_tmp1_imm
                                            : ((i_ur % 4) == 2) ? reg_tmp2_imm
                                                                : reg_tmp3_imm;
                CGA64::add_imm(
                        reg_tmp_adr, xa::XReg(base.getIdx()), ofs, reg_tmp_imm);
                CGA64::ld1rw(xa::ZRegS(bcast_reg.getIdx()),
                        xa::PReg(vmask.getIdx()), xa::ptr(reg_tmp_adr));
            }
        }
    };

    auto load_ptr = [=](xa::ZReg load_reg, int i_reduce, int i_load) {
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;

        int offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;
        int ofs = get_offset(
                u1 * jcp.reduce_loop_load_step + jcp.typesize_in * offt);

        if (ofs == 0) {
            CGA64::ldr(load_reg, xa::ptr(aux_reg_load_data));
        } else {
            auto reg_tmp_adr = ((i_load % 4) == 0) ? reg_tmp0_adr
                                                   : ((i_load % 4) == 1)
                            ? reg_tmp1_adr
                            : ((i_load % 4) == 2) ? reg_tmp2_adr : reg_tmp3_adr;
            auto reg_tmp_imm = ((i_load % 4) == 0) ? reg_tmp0_imm
                                                   : ((i_load % 4) == 1)
                            ? reg_tmp1_imm
                            : ((i_load % 4) == 2) ? reg_tmp2_imm : reg_tmp3_imm;
            CGA64::add_imm(reg_tmp_adr, aux_reg_load_data, ofs, reg_tmp_imm);
            CGA64::ldr(load_reg, xa::ptr(reg_tmp_adr));
        }
    };

    auto output_ptr = [=](xa::ZReg output_reg, int i_load, int i_ur,
                              bool mask_flag) {
        const size_t ur_stride = jcp.with_dw_conv
                ? jcp.nb_load_blocking * jcp.oc_block * i_ur
                : jcp.oc_without_padding * jcp.ngroups * i_ur;

        int offt = get_offset(
                jcp.typesize_out * (ur_stride + i_load * jcp.load_block));

        CGA64::add_imm(reg_tmp0_adr, aux_reg_output_data, offt, reg_tmp0_imm);
        if (mask_flag)
            CGA64::ld1w(
                    output_reg.s, ktail_mask / xa::T_z, xa::ptr(reg_tmp0_adr));
        else
            CGA64::ldr(output_reg, xa::ptr(reg_tmp0_adr));
    };

    auto output_ptr8 = [=](xa::ZReg output_reg, int i_load, int i_ur,
                               bool mask_flag) {
        const size_t ur_stride = jcp.with_dw_conv
                ? jcp.nb_load_blocking * jcp.oc_block * i_ur
                : jcp.oc_without_padding * jcp.ngroups * i_ur;

        int offt = get_offset(
                jcp.typesize_out * (ur_stride + i_load * jcp.load_block));

        CGA64::add_imm(reg_tmp0_adr, aux_reg_output_data, offt, reg_tmp0_imm);
        if (mask_flag) {
            CGA64::uzp1(ktail_load_mask.h, ktail_mask.h, mask_all_zero.h);
            CGA64::uzp1(ktail_load_mask.b, ktail_load_mask.b, mask_all_zero.b);
            CGA64::ld1b(output_reg.b, ktail_load_mask / xa::T_z,
                    xa::ptr(reg_tmp0_adr));
        } else {
            CGA64::ldr(xa::QReg(output_reg.getIdx()), xa::ptr(reg_tmp0_adr));
        }
    };

    auto init = [=]() {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                CGA64::eor(r.d, r.d, r.d);
            }
        if (!jcp.signed_input) {
            //mov(reg_scratch, -128);
            //vpbroadcastb(vmm_shift, reg_scratch.cvt8());
            CGA64::dup(vmm_shift.b, -128);
        }
    };

    auto store = [=](const bool mask_flag_in) {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale = nullptr;
        if (sum_idx != -1) p_sum_scale = &p.entry_[sum_idx].sum.scale;
        CGA64::str(
                reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));
        CGA64::ldr(reg_ptr_scales,
                SVE_compress_addr(reg_rsp, reg_ptr_sum_scale_off));
        if (p_sum_scale && *p_sum_scale != 1.f) {
            CGA64::str(reg_load_data,
                    SVE_compress_addr(reg_rsp, reg_load_data_off));
            CGA64::mov_imm(reg_ptr_sum_scale, (size_t)p_sum_scale);
        }
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            auto vmm_bias = vmm_tmp;
            auto vmm_comp = vmm_bcast;
            if (jcp.with_bias) {
                if (!jcp.signed_input)
                    CGA64::ldr(reg_bias_data,
                            SVE_compress_addr(reg_rsp, reg_bias_data_off));
                //cvt2ps(jcp.bia_dt, vmm_bias, bias_ptr(i_load), mask_flag);
                switch (jcp.bia_dt) {
                    case data_type::f32:
                    case data_type::s32:
                        bias_ptr(vmm_bias, i_load, mask_flag);
                        break;
                    case data_type::s8:
                        CGA64::sub(x22, x22, 64);
                        CGA64::str(xa::ZReg(29), xa::ptr(x22));
                        bias_ptr8(xa::ZReg(29), i_load, mask_flag);
                        CGA64::zip1(
                                xa::ZRegB(29), xa::ZRegB(29), xa::ZRegB(29));
                        CGA64::zip1(
                                xa::ZRegH(29), xa::ZRegH(29), xa::ZRegH(29));
                        CGA64::sxtb(xa::ZRegS(vmm_bias.getIdx()),
                                vmask / xa::T_m, xa::ZRegS(29));
                        if (mask_flag) {
                            CGA64::not_(mask_tmp.b, vmask.b, ktail_mask.b);
                            CGA64::mov(vmm_bias.s, mask_tmp / xa::T_m, 0);
                        }
                        CGA64::ldr(xa::ZReg(29), xa::ptr(x22));
                        CGA64::add(x22, x22, 64);
                        break;
                    case data_type::u8:
                        CGA64::sub(x22, x22, 64);
                        CGA64::str(xa::ZReg(29), xa::ptr(x22));
                        bias_ptr8(xa::ZReg(29), i_load, mask_flag);
                        CGA64::zip1(
                                xa::ZRegB(29), xa::ZRegB(29), xa::ZRegB(29));
                        CGA64::zip1(
                                xa::ZRegH(29), xa::ZRegH(29), xa::ZRegH(29));
                        CGA64::uxtb(xa::ZRegS(vmm_bias.getIdx()),
                                vmask / xa::T_m, xa::ZRegS(29));
                        if (mask_flag) {
                            CGA64::not_(mask_tmp.b, vmask.b, ktail_mask.b);
                            CGA64::mov(vmm_bias.s, mask_tmp / xa::T_m, 0);
                        }
                        CGA64::ldr(xa::ZReg(29), xa::ptr(x22));
                        CGA64::add(x22, x22, 64);
                        break;
                    default: assert(!"unsupported data type");
                }
                if (jcp.bia_dt != data_type::f32)
                    CGA64::scvtf(xa::ZRegS(vmm_bias.getIdx()), vmask,
                            xa::ZRegS(vmm_bias.getIdx()));
            }
            if (!jcp.signed_input) {
                CGA64::ldr(reg_comp_data,
                        SVE_compress_addr(reg_rsp, reg_comp_data_off));
                // cvt2ps(data_type::s32, vmm_comp, comp_ptr(i_load), mask_flag);
                comp_ptr(vmm_comp, i_load, mask_flag);
                CGA64::scvtf(vmm_comp.s, vmask, vmm_comp.s);
            }

            auto vmm_scale = vmm_one;
            scale_ptr(vmm_scale, i_load);
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                CGA64::scvtf(xa::ZRegS(r.getIdx()), xa::PReg(vmask.getIdx()),
                        xa::ZRegS(r.getIdx())); //< vcvtdq2ps(r, r);
                if (!jcp.signed_input) CGA64::fsub(r.s, r.s, vmm_comp.s);
                if (jcp.with_bias) CGA64::fadd(r.s, r.s, vmm_bias.s);

                // const Vmm mask_vmm = mask_flag ? r | ktail_mask | T_z : r;
                zmm_t mask_vmm = r;
                // vmulps(mask_vmm, r, scale_ptr(i_load));
                CGA64::fmul(xa::ZRegS(mask_vmm.getIdx()), xa::ZRegS(r.getIdx()),
                        xa::ZRegS(vmm_scale.getIdx()));
                if (mask_flag) {
                    CGA64::not_(mask_tmp.b, vmask.b, ktail_mask.b);
                    CGA64::mov(xa::ZRegS(mask_vmm.getIdx()), mask_tmp / xa::T_m,
                            0);
                }
            }
        }

        if (maybe_eltwise(0))
            eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);

        if (p_sum_scale) { // post_op: sum
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                const bool mask_flag
                        = mask_flag_in && i_load == load_loop_blk - 1;
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    CGA64::eor(vmm_zero.d, vmm_zero.d, vmm_zero.d);
                    auto vmm_prev_dst = vmm_zero;

                    auto r = vreg_accum(i_load, i_ur);
                    // cvt2ps(jcp.dst_dt, vmm_prev_dst, output_ptr(i_load, i_ur),
                    //         mask_flag);
                    switch (jcp.dst_dt) {
                        case data_type::f32:
                        case data_type::s32:
                            output_ptr(vmm_prev_dst, i_load, i_ur, mask_flag);
                            break;
                        case data_type::s8:
                            CGA64::sub(x22, x22, 64);
                            CGA64::str(xa::ZReg(29), xa::ptr(x22));
                            output_ptr8(xa::ZReg(29), i_load, i_ur, mask_flag);
                            CGA64::zip1(xa::ZRegB(29), xa::ZRegB(29),
                                    xa::ZRegB(29));
                            CGA64::zip1(xa::ZRegH(29), xa::ZRegH(29),
                                    xa::ZRegH(29));
                            CGA64::sxtb(xa::ZRegS(vmm_prev_dst.getIdx()),
                                    vmask / xa::T_m, xa::ZRegS(29));
                            if (mask_flag) {
                                CGA64::not_(mask_tmp.b, vmask.b, ktail_mask.b);
                                CGA64::mov(
                                        vmm_prev_dst.s, mask_tmp / xa::T_m, 0);
                            }
                            CGA64::ldr(xa::ZReg(29), xa::ptr(x22));
                            CGA64::add(x22, x22, 64);
                            break;
                        case data_type::u8:
                            CGA64::sub(x22, x22, 64);
                            CGA64::str(xa::ZReg(29), xa::ptr(x22));
                            output_ptr8(xa::ZReg(29), i_load, i_ur, mask_flag);
                            CGA64::zip1(xa::ZRegB(29), xa::ZRegB(29),
                                    xa::ZRegB(29));
                            CGA64::zip1(xa::ZRegH(29), xa::ZRegH(29),
                                    xa::ZRegH(29));
                            CGA64::uxtb(xa::ZRegS(vmm_prev_dst.getIdx()),
                                    vmask / xa::T_m, xa::ZRegS(29));
                            if (mask_flag) {
                                CGA64::not_(mask_tmp.b, vmask.b, ktail_mask.b);
                                CGA64::mov(
                                        vmm_prev_dst.s, mask_tmp / xa::T_m, 0);
                            }
                            CGA64::ldr(xa::ZReg(29), xa::ptr(x22));
                            CGA64::add(x22, x22, 64);
                            break;
                        default: assert(!"unsupported data type");
                    }
                    if (jcp.dst_dt != data_type::f32)
                        CGA64::scvtf(xa::ZRegS(vmm_prev_dst.getIdx()), vmask,
                                xa::ZRegS(vmm_prev_dst.getIdx()));

                    if (*p_sum_scale == 1.f) {
                        // vaddps(r, vmm_prev_dst);
                        CGA64::fadd(r.s, r.s, vmm_prev_dst.s);
                    } else {
                        // vfmadd231ps(
                        //         r, vmm_prev_dst, zword_b[reg_ptr_sum_scale]);
                        CGA64::sub(x22, x22, 64);
                        CGA64::str(xa::ZReg(29), xa::ptr(x22));
                        CGA64::ld1rw(xa::ZRegS(29), vmask / xa::T_z,
                                xa::ptr(reg_ptr_sum_scale));
                        CGA64::fmla(r.s, vmask / xa::T_m, vmm_prev_dst.s,
                                xa::ZRegS(29));
                        CGA64::ldr(xa::ZReg(29), xa::ptr(x22));
                        CGA64::add(x22, x22, 64);
                    }
                }
            }
        }

        if (maybe_eltwise(1))
            eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);

        // Properly saturate the accumulators for integer datatypes
        if (one_of(jcp.dst_dt, u8, s8, s32)) {
            //            init_saturate_f32(vmm_zero, vmm_saturation,
            //                    reg_ptr_saturation_ubound, f32, jcp.dst_dt);

            if (jcp.dst_dt == data_type::u8) {
                CGA64::eor(vmm_zero.d, vmm_zero.d, vmm_zero.d);
            }
            float saturation_ubound = types::max_value<float>(jcp.dst_dt);
            CGA64::mov_imm(reg_tmp0_imm, float2int(saturation_ubound));
            CGA64::dup(vmm_saturation.s, xa::WReg(reg_tmp0_imm.getIdx()));

            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    auto r = vreg_accum(i_load, i_ur);
                    // saturate_f32(r, vmm_zero, vmm_saturation, jcp.dst_dt);

                    if (jcp.dst_dt == data_type::u8) {
                        // vmaxps(vmm, vmm, vmm_lbound);
                        CGA64::fmaxnm(r.s, vmask, vmm_zero.s);
                        CGA64::fmax(r.s, vmask, vmm_zero.s);
                    }
                    // vminps(vmm, vmm, vmm_ubound);
                    CGA64::fminnm(r.s, vmask, vmm_saturation.s);
                    CGA64::fmin(r.s, vmask, vmm_saturation.s);

                    // vcvtps2dq(r, r);
#if 1
                    CGA64::frintn(r.s, vmask, r.s); // T_rn_sae
                    CGA64::fcvtzs(r.s, vmask, r.s);
#else
                    CGA64::frintm(r.s, vmask, r.s); // T_rd_sae
                    CGA64::fcvtzs(r.s, vmask, r.s);
#endif
                }
            }
        }

        // store to the destination
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                // const Vmm r_vmm = mask_flag ? r | ktail_mask : r;
                zmm_t r_vmm = r;

                auto base = aux_reg_output_data;
                auto raw_offt = jcp.typesize_out
                        * (jcp.oc_without_padding * i_ur
                                + i_load * jcp.load_block);

                assert(raw_offt <= INT_MAX);
                auto offt = static_cast<int>(raw_offt);

                int scale = 0;

                if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
                    offt = offt - 2 * EVEX_max_8b_offt;
                    scale = 1;
                } else if (3 * EVEX_max_8b_offt <= offt
                        && offt < 5 * EVEX_max_8b_offt) {
                    offt = offt - 4 * EVEX_max_8b_offt;
                    scale = 2;
                }

                auto re = offt;
                if (scale) re = re + (2 * EVEX_max_8b_offt) * scale;

                auto reg_tmp_adr = ((i_ur % 4) == 0)
                        ? reg_tmp0_adr
                        : ((i_ur % 4) == 1) ? reg_tmp1_adr
                                            : ((i_ur % 4) == 2) ? reg_tmp2_adr
                                                                : reg_tmp3_adr;
                auto reg_tmp_imm = ((i_ur % 4) == 0)
                        ? reg_tmp0_imm
                        : ((i_ur % 4) == 1) ? reg_tmp1_imm
                                            : ((i_ur % 4) == 2) ? reg_tmp2_imm
                                                                : reg_tmp3_imm;
                CGA64::add_imm(
                        reg_tmp_adr, xa::XReg(base.getIdx()), re, reg_tmp_imm);

                auto _mask = mask_flag ? ktail_mask : vmask;
                switch (jcp.dst_dt) {
                    case data_type::f32:
                    case data_type::s32:
                        // vmovups(output_ptr(i_load, i_ur), r_vmm);
                        CGA64::st1w(r_vmm.s, _mask, xa::ptr(reg_tmp_adr));
                        break;
                    case data_type::s8:
                        // vpmovsdb(output_ptr(i_load, i_ur), r_vmm);
                        CGA64::smin(r_vmm.s, 127);
                        CGA64::smax(r_vmm.s, -128);
                        CGA64::st1b(r_vmm.s, _mask, xa::ptr(reg_tmp_adr));
                        break;
                    case data_type::u8:
                        // vpmovusdb(output_ptr(i_load, i_ur), r_vmm);
                        CGA64::umin(r_vmm.s, 255);
                        CGA64::st1b(r_vmm.s, _mask, xa::ptr(reg_tmp_adr));
                        break;
                    default: assert(!"unknown dst_dt");
                }
            }
        }
        CGA64::ldr(
                reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));
        if (p_sum_scale && *p_sum_scale != 1.f)
            CGA64::ldr(reg_load_data,
                    SVE_compress_addr(reg_rsp, reg_load_data_off));
    };

    auto compute = [=](xa::ZReg vreg_acc, xa::ZReg vreg_wei,
                           xa::ZReg vreg_src) {
        // vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        CGA64::sdot(xa::ZRegS(vreg_acc.getIdx()), xa::ZRegB(vreg_src.getIdx()),
                xa::ZRegB(vreg_wei.getIdx()));
    };

    auto fma_block = [=](bool last_block) {
        int reduce_step = 4;
        int ic_tail_size = jcp.ic_without_padding % reduce_step;
        int loop_unroll = last_block && jcp.ic != jcp.ic_without_padding
                ? rnd_up(jcp.ic_without_padding % jcp.ic_block, reduce_step)
                : jcp.reduce_loop_unroll;
        for (int i_reduce = 0; i_reduce < loop_unroll;
                i_reduce += reduce_step) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                load_ptr(vreg_load(i_load), i_reduce, i_load);
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                if (jcp.signed_input) {
                    if (last_block && ic_tail_size != 0
                            && i_reduce == loop_unroll - reduce_step) {
                        auto xmm_bcast = xa::VReg16B(vmm_bcast.getIdx());
                        // load_bytes(xmm_bcast, aux_reg_bcast_data,
                        //         jcp.ic_without_padding * i_ur + i_reduce,
                        //         ic_tail_size);
                        for (int r = 0; r < ic_tail_size; ++r) {
                            CGA64::add_imm(reg_tmp0_adr, aux_reg_bcast_data,
                                    (jcp.ic_without_padding * i_ur + i_reduce
                                            + r),
                                    reg_tmp0_imm);
                            CGA64::ldrb(xa::WReg(reg_tmp1_imm.getIdx()),
                                    xa::ptr(reg_tmp0_adr));
                            CGA64::ins_(xa::VReg16B(xmm_bcast.getIdx())[r],
                                    xa::WReg(reg_tmp1_imm.getIdx()));
                        }
                        // vpbroadcastd(vmm_bcast, xmm_bcast);
                        auto _bcast
                                = ((i_ur % 2) == 0) ? vmm_bcast : vmm_bcast2;
                        CGA64::dup(xa::ZRegS(_bcast.getIdx()),
                                xa::ZRegS(xmm_bcast.getIdx())[0]);
                    } else {
                        if (i_ur == 0) {
                            // vpbroadcastd(vmm_bcast, bcast_ptr(i_reduce, i_ur, false));
                            bcast_ptr(vmm_bcast, i_reduce, i_ur, false);
                        }
                        if ((i_ur + 1) < ur) {
                            xa::ZReg _bcast = ((i_ur % 2) == 0) ? vmm_bcast2
                                                                : vmm_bcast;
                            // vpbroadcastd(vmm_bcast, bcast_ptr(i_reduce, (i_ur+1), false));
                            bcast_ptr(_bcast, i_reduce, (i_ur + 1), false);
                        }
                    }
                    for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                        xa::ZReg _bcast
                                = ((i_ur % 2) == 0) ? vmm_bcast : vmm_bcast2;
                        compute(vreg_accum(i_load, i_ur), vreg_load(i_load),
                                _bcast);
                    }
                } else {
                    if (last_block && ic_tail_size != 0
                            && i_reduce == loop_unroll - reduce_step) {
                        auto xmm_bcast = xa::VReg16B(vmm_bcast.getIdx());
                        for (int r = 0; r < ic_tail_size; ++r) {
                            CGA64::add_imm(reg_tmp0_adr, aux_reg_bcast_data,
                                    (jcp.ic_without_padding * i_ur + i_reduce
                                            + r),
                                    reg_tmp0_imm);
                            CGA64::ldrb(xa::WReg(reg_tmp1_imm.getIdx()),
                                    xa::ptr(reg_tmp0_adr));
                            CGA64::ins_(xa::VReg16B(xmm_bcast.getIdx())[r],
                                    xa::WReg(reg_tmp1_imm.getIdx()));
                        }
                        CGA64::dup(xa::ZRegS(vmm_bcast.getIdx()),
                                xa::ZRegS(xmm_bcast.getIdx())[0]);
                    } else {
                        bcast_ptr(vmm_bcast, i_reduce, i_ur, false);
                    }
                    CGA64::add(vmm_bcast.b, vmm_bcast.b, vmm_shift.b);
                    for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                        compute(vreg_accum(i_load, i_ur), vreg_load(i_load),
                                vmm_bcast);
                    }
                }
            }
        }
    };

    xa::LabelAArch64 reduce_loop;
    xa::LabelAArch64 reduce_loop_tail;

    CGA64::mov(aux_reg_load_data, reg_load_data);

    CGA64::mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    CGA64::mov(reduce_loop_iter, reg_reduce_loop_work);
    CGA64::subs(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll);
    CGA64::b(xa::LE, reduce_loop_tail);

    CGA64::L_aarch64(reduce_loop);
    {
        fma_block(false);
        CGA64::adds(aux_reg_bcast_data, aux_reg_bcast_data,
                jcp.reduce_loop_bcast_step);
        CGA64::adds(aux_reg_load_data, aux_reg_load_data,
                jcp.reduce_loop_load_step);
        CGA64::subs(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll);
        CGA64::b(xa::GT, reduce_loop);
    }

    CGA64::L_aarch64(reduce_loop_tail);
    if (jcp.ic != jcp.ic_without_padding) {
        fma_block(true);
    } else {
        fma_block(false);
    }

    if (jcp.oc_without_padding != jcp.oc) {
        xa::LabelAArch64 end_store, common_store;
        CGA64::str(
                reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));

        /*Check if it is the last load_loop_blk*/
        CGA64::subs(reg_load_loop_work, reg_load_loop_work,
                load_loop_blk * jcp.load_loop_iter_step);
        CGA64::cmp(reg_load_loop_work, 0);
        CGA64::b(xa::GT, common_store);

        /*Check if it is the last ocb*/
        CGA64::tst(reg_reduce_pos_flag, FLAG_OC_LAST);
        CGA64::b(xa::EQ, common_store);

        store(true);
        CGA64::b(end_store);

        CGA64::L_aarch64(common_store);
        store(false);

        CGA64::L_aarch64(end_store);

        CGA64::add_imm(reg_load_loop_work, reg_load_loop_work,
                load_loop_blk * jcp.load_loop_iter_step, reg_tmp0_imm);
    } else {
        store(false);
    }
}

template <typename Vmm>
void _jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel<Vmm>::generate() {

    preamble();
    const int simd_w = jcp.ic_block;

    CGA64::ptrue(xa::PRegB(vmask.getIdx()));
    CGA64::pfalse(xa::PRegB(mask_all_zero.getIdx()));

    // xor_(reg_scratch, reg_scratch);
    // Reg16 _t = reg_scratch.cvt16();
    // mov(_t, 0x1);
    // vpbroadcastw(vmm_one, _t);
    CGA64::dup(vmm_one.h, 0x1);

    CGA64::subs(reg_rsp, reg_rsp, stack_space_needed);

    if (jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        // Reg32 regw_tmp = reg_last_load.cvt32();
        auto regw_tmp = reg_last_load;
        CGA64::mov(regw_tmp, mask);
        // kmovw(ktail_mask, regw_tmp);
        CGA64::index(xa::ZRegS(0), 0, 1);
        CGA64::mov(xa::ZRegS(1), 1);
        CGA64::lsl(xa::ZRegS(1), vmask / xa::T_m, xa::ZRegS(0));
        CGA64::dup(xa::ZRegS(0), xa::WReg(regw_tmp.getIdx()));
        CGA64::and_(xa::ZRegD(0), xa::ZRegD(0), xa::ZRegD(1));
        CGA64::cmpne(ktail_mask.s, vmask, xa::ZRegS(0), 0);
    }

    if (jcp.with_bias)
        CGA64::ldr(reg_bias_data, xa::ptr(reg_abi_param1, GET_OFF(bias_data)));
    if (!jcp.signed_input) {
        CGA64::str(
                reg_bias_data, SVE_compress_addr(reg_rsp, reg_bias_data_off));
        CGA64::ldr(
                reg_comp_data, xa::ptr(reg_abi_param1, GET_OFF(compensation)));
        CGA64::str(
                reg_comp_data, SVE_compress_addr(reg_rsp, reg_comp_data_off));
    }

    CGA64::ldr(reg_ptr_scales, xa::ptr(reg_abi_param1, GET_OFF(scales)));
    CGA64::str(
            reg_ptr_scales, SVE_compress_addr(reg_rsp, reg_ptr_sum_scale_off));
    CGA64::ldr(reg_bcast_data, xa::ptr(reg_abi_param1, GET_OFF(bcast_data)));
    CGA64::ldr(reg_load_data, xa::ptr(reg_abi_param1, GET_OFF(load_data)));
    CGA64::ldr(reg_output_data, xa::ptr(reg_abi_param1, GET_OFF(output_data)));

    CGA64::ldr(reg_load_loop_work, xa::ptr(reg_abi_param1, GET_OFF(load_dim)));
    CGA64::ldr(
            reg_bcast_loop_work, xa::ptr(reg_abi_param1, GET_OFF(bcast_dim)));
    CGA64::str(reg_bcast_loop_work,
            SVE_compress_addr(reg_rsp, bcast_loop_work_off));
    CGA64::ldr(
            reg_reduce_loop_work, xa::ptr(reg_abi_param1, GET_OFF(reduce_dim)));
    CGA64::ldr(reg_reduce_pos_flag,
            xa::ptr(reg_abi_param1, GET_OFF(first_last_flag)));

    auto load_loop_body = [=](int load_loop_blk) {
        bcast_loop(load_loop_blk);
        CGA64::add_imm(reg_load_data, reg_load_data,
                load_loop_blk * jcp.load_loop_load_step, reg_tmp0_imm);
        if (jcp.with_bias) {
            if (!jcp.signed_input)
                CGA64::ldr(reg_bias_data,
                        SVE_compress_addr(reg_rsp, reg_bias_data_off));
            CGA64::add_imm(reg_bias_data, reg_bias_data,
                    load_loop_blk * jcp.load_block * jcp.typesize_bia,
                    reg_tmp0_imm);
            if (!jcp.signed_input)
                CGA64::str(reg_bias_data,
                        SVE_compress_addr(reg_rsp, reg_bias_data_off));
        }
        if (!jcp.signed_input) {
            CGA64::ldr(reg_comp_data,
                    SVE_compress_addr(reg_rsp, reg_comp_data_off));
            CGA64::add_imm(reg_comp_data, reg_comp_data,
                    load_loop_blk * jcp.load_block * sizeof(int32_t),
                    reg_tmp0_imm);
            CGA64::str(reg_comp_data,
                    SVE_compress_addr(reg_rsp, reg_comp_data_off));
        }
        CGA64::str(
                reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));
        CGA64::ldr(reg_ptr_scales,
                SVE_compress_addr(reg_rsp, reg_ptr_sum_scale_off));
        CGA64::add_imm(reg_ptr_scales, reg_ptr_scales,
                jcp.is_oc_scale * load_loop_blk * jcp.load_block
                        * sizeof(float),
                reg_tmp0_imm);
        CGA64::str(reg_ptr_scales,
                SVE_compress_addr(reg_rsp, reg_ptr_sum_scale_off));
        CGA64::ldr(
                reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));
        CGA64::adds(reg_output_data, reg_output_data,
                load_loop_blk * jcp.load_block * jcp.typesize_out);
        CGA64::subs(reg_load_loop_work, reg_load_loop_work,
                load_loop_blk * jcp.load_loop_iter_step);
    };

    xa::LabelAArch64 load_loop_blk[7];

    static const int ur_cases_fma_expl_bcast[] = {2, 5, 6, 9, 14, 32};
    const int size_ur_cases_fma = sizeof(ur_cases_fma_expl_bcast);
    const int *ur_cases_fma = ur_cases_fma_expl_bcast;
    const int *ur_cases = ur_cases_fma;
    const int num_ur_cases = (size_ur_cases_fma) / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            CGA64::cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            CGA64::b(xa::LE, load_loop_blk[label_idx]);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = num_ur_cases - ur_idx - 1;
            CGA64::L_aarch64(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    CGA64::cmp(reg_load_loop_work, 0);
                    CGA64::b(xa::EQ, load_loop_blk[num_ur_cases]);
                }

                for (int _i = 1; _i <= label_idx + 1; _i++) {
                    // prefetcht0(ptr[reg_load_data + _i * jcp.ic * jcp.oc_block]);
                    // prefetcht1(ptr[reg_output_data + _i * jcp.oc_block]);
                    CGA64::add_imm(reg_tmp0_adr, reg_load_data,
                            (_i * jcp.ic * jcp.oc_block), reg_tmp0_imm);
                    CGA64::add_imm(reg_tmp1_adr, reg_output_data,
                            (_i * jcp.oc_block), reg_tmp1_imm);
                    CGA64::prfm(xa::PLDL1KEEP, xa::ptr(reg_tmp0_adr));
                    CGA64::prfm(xa::PLDL2KEEP, xa::ptr(reg_tmp1_adr));
                }

                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    CGA64::cmp(reg_load_loop_work, 2 * label_idx * simd_w);
                    CGA64::b(xa::EQ, load_loop_blk[label_idx - 1]);
                }
                CGA64::cmp(reg_load_loop_work, (label_idx + 1) * simd_w);
                CGA64::b(xa::GE, load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                CGA64::cmp(reg_load_loop_work, simd_w * (idx + 1));
                CGA64::b(xa::EQ, load_loop_blk[idx]);
            }
            if (ur_idx < num_ur_cases - 2) {
                CGA64::cmp(reg_load_loop_work, simd_w);
                CGA64::b(xa::LE, load_loop_blk[0]);
            }
        }
    }
    CGA64::L_aarch64(load_loop_blk[num_ur_cases]);

    CGA64::add_imm(reg_rsp, reg_rsp, stack_space_needed, reg_tmp0_imm);

    postamble();

    if (jcp.with_eltwise) {
        eltwise_injector_->prepare_table();
#ifdef DNNL_INDIRECT_JIT_AARCH64
        binCommit();
#endif
    }
}

bool jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_convolution
            = [&](int idx) { return p.entry_[idx].is_convolution(); };

    int dw_idx = p.find(primitive_kind::convolution);
    int len = dw_idx != -1 ? dw_idx + 1 : p.len();

    switch (len) {
        case 0: return true;
        case 1: return is_eltwise(0) || p.contain(sum, 0) || is_convolution(0);
        case 2:
            return (p.contain(sum, 0) && is_eltwise(1))
                    || (p.contain(sum, 1) && is_eltwise(0))
                    || (is_eltwise(0) && is_convolution(1));
        default: return false;
    }

    return false;
}

status_t jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_t *&src_md, memory_desc_t &weights_md,
        memory_desc_t &dst_md, memory_desc_t &bias_md,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {

    if (!mayiuse(sve)) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!one_of(src_d.data_type(), data_type::u8, data_type::s8)
            || weights_d.data_type() != data_type::s8
            || !one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8))
        return status::unimplemented;

    jcp.nthr = nthreads;

    jcp.ver = ver_sve;

    int ndims = src_d.ndims();

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;

    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    jcp.signed_input = (src_d.data_type() == data_type::s8);

    dim_t output_spatial = jcp.od * jcp.oh * jcp.ow;
    dim_t input_spatial = jcp.id * jcp.ih * jcp.iw;

    // FIXME: jcp.os and jcp.is fields have data type of int
    if (output_spatial > INT_MAX || input_spatial > INT_MAX)
        return status::unimplemented;

    jcp.os = output_spatial;
    jcp.is = input_spatial;

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;
    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.
    const int eltwise_ind = p.find(primitive_kind::eltwise, 0, dw_conv_ind);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (jcp.eltwise.alg == alg_kind::eltwise_pow)
            return status::unimplemented;
    }

    format_tag_t dat_tag = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);

    bool args_ok = jcp.src_tag == dat_tag && jcp.dst_tag == dat_tag;
    if (!args_ok) return status::unimplemented;

    if (jcp.ngroups == 1) {
        jcp.oc = rnd_up(jcp.oc, 16);
        jcp.ic = rnd_up(jcp.ic, 16);
    }

    const int simd_w = (jcp.ic % 16 == 0 && jcp.oc % 16 == 0)
            ? 16
            : (jcp.ic % 8 == 0 && jcp.oc % 8 == 0) ? 8 : 4;

    auto set_or_check_wei_format = [&]() -> bool {
        using namespace format_tag;
        const format_tag_t wei_tags[3][2][3]
                = {{{OIw4i16o4i, OIhw4i16o4i, OIdhw4i16o4i},
                           {gOIw4i16o4i, gOIhw4i16o4i, gOIdhw4i16o4i}},
                        {{OIw2i8o4i, OIhw2i8o4i, OIdhw2i8o4i},
                                {gOIw2i8o4i, gOIhw2i8o4i, gOIdhw2i8o4i}},
                        {{OIw4o4i, OIhw4o4i, OIdhw4o4i},
                                {gOIw4o4i, gOIhw4o4i, gOIdhw4o4i}}};

        const int simd_idx = simd_w == 16 ? 0 : simd_w == 8 ? 1 : 2;
        const auto wei_tag = wei_tags[simd_idx][with_groups][ndims - 3];
        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);
        if (!jcp.signed_input) {
            want_wei_md.extra.flags = 0
                    | memory_extra_flags::compensation_conv_s8s8
                    | memory_extra_flags::scale_adjust;
            want_wei_md.extra.compensation_mask
                    = (1 << 0) + (with_groups ? (1 << 1) : 0);
            want_wei_md.extra.scale_adjust
                    // = mayiuse(avx512_core_vnni) ? 1.f : 0.5f;
                    = 1.f;
        }

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) return status::unimplemented;

    args_ok = true && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
            && jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0
            && jcp.stride_d == 1 && jcp.stride_h == 1
            && jcp.stride_w == 1 // TODO: support some strides
            && jcp.od == jcp.id && jcp.oh == jcp.ih
            && jcp.ow == jcp.iw // enforce rpad = 0
            && jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    const int SMALL_SPATIAL = 7 * 7;
    const int BIG_REDUCE_DIM = 1024;

    int load_blocking = 0;
    int load_blocking_max = 0;
    int bcast_blocking = 0;
    int bcast_blocking_max = 0;
    int reduce_blocking = 0;
    int reduce_blocking_max = 0;
    jcp.load_grp_count = 1;
    jcp.use_vmovntps = false;

    const int L2_size
            = platform::get_per_core_cache_size(2) / sizeof(jcp.typesize_in);
    const int L2_capacity = (L2_size * 3) / 4;

    int size_treshold = 28;
    int max_regs = 0;
    int min_regs = 6;
    if (jcp.ver == ver_sve)
        max_regs = ((jcp.oh > size_treshold && jcp.ow > size_treshold)
                           && (jcp.oc < 128 || jcp.ic < 128))
                ? min_regs
                : 9;
    else
        max_regs = 8;
    jcp.expl_bcast = true;

    if (jcp.mb == 1 && jcp.ic > 128
            && (jcp.oh <= size_treshold && jcp.ow <= size_treshold)) {
        if (jcp.os <= SMALL_SPATIAL && jcp.oc * jcp.ic < L2_size)
            max_regs = min_regs; // mobilenet_v2 performance improvement
        jcp.ur = nstl::min(max_regs, jcp.os);
    } else {
        const int spatial = jcp.od * jcp.oh;
        jcp.ur = 1;
        for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
            if ((spatial >= size_treshold && spatial % ur_w == 0)
                    || (spatial < size_treshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = nstl::min(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i--) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
    }
    if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);

    jcp.reduce_dim = jcp.ic;
    jcp.reduce_block = jcp.ic_block;

    jcp.load_dim = jcp.oc;
    jcp.load_block = jcp.oc_block;

    jcp.bcast_dim = jcp.is;

    jcp.bcast_block = jcp.ur;

    jcp.reduce_loop_unroll = jcp.reduce_block;
    jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll * jcp.typesize_in;

    jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;

    jcp.bcast_loop_output_step
            = jcp.ur * jcp.ngroups * jcp.oc_without_padding * jcp.typesize_out;
    jcp.bcast_loop_output_substep = -1; // unused
    jcp.bcast_loop_bcast_step
            = jcp.ur * jcp.ngroups * jcp.ic_without_padding * jcp.typesize_in;
    jcp.bcast_loop_bcast_substep = -1; // unused

    jcp.load_loop_load_step = jcp.reduce_dim * jcp.load_block * jcp.typesize_in;

    jcp.load_loop_iter_step = jcp.load_block;

    jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

    int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    reduce_blocking = nb_reduce;
    if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 64;
    else if (jcp.bcast_dim > SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 16;
    reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
    reduce_blocking *= jcp.reduce_block;

    bool cmp_reduce = reduce_blocking <= jcp.reduce_dim;
    if (cmp_reduce) jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
    load_blocking = jcp.load_dim;

    jcp.load_grp_count = div_up(jcp.nthr, jcp.mb * jcp.ngroups * nb_bcast);
    jcp.load_grp_count = best_divider(
            jcp.nthr, jcp.load_grp_count, 2 * jcp.load_grp_count, false);

    if (jcp.bcast_dim <= SMALL_SPATIAL
            && jcp.load_dim * jcp.reduce_dim >= L2_size) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 4);
    } else if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.mb <= jcp.nthr
            && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2); //
        load_blocking = jcp.load_block;
    }

    bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                             div_up(jcp.nthr, jcp.load_grp_count))
            * jcp.bcast_block;
    bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
    bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

    int space_for_bcast = (L2_capacity - /* kernel_size - */
            2 * jcp.load_block * reduce_blocking - jcp.ur * reduce_blocking
            - 3 * 1024);
    if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity) space_for_bcast /= 2;

    int bcast_in_cache
            = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
    bcast_blocking = nstl::min(
            bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

    load_blocking_max = load_blocking;
    bcast_blocking_max = bcast_blocking * 3 / 2;
    reduce_blocking_max = reduce_blocking;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);
    assert(load_blocking % jcp.load_block == 0);
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);

    assert(jcp.reduce_loop_unroll % 4 == 0);
    assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);

    assert(jcp.bcast_block % jcp.ur == 0);
    assert(jcp.reduce_dim % jcp.reduce_block == 0);

    jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;
    jcp.nb_reduce_blocking_max = reduce_blocking_max / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    // miniumum size of load dim chunk for work distribution within threads
    jcp.nb_load_chunk = 1;
    // peformance improvements for googlenet_v3, mb=1;
    // TODO: generalize this condition and rewrite it in appropriate manner
    int ncores_per_socket = (int)cpu.getNumCores(
            Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    if (jcp.mb == 1 && jcp.nb_load % 4 == 0 && jcp.ic / jcp.oc >= 4
            && jcp.ic * jcp.oc <= L2_size && jcp.nthr <= ncores_per_socket) {
        jcp.nb_load_chunk = 4;
        jcp.load_grp_count = nstl::max(jcp.nb_load / 4, jcp.load_grp_count);
    }

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    return status::success;
}

void jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace dnnl::impl::memory_tracking::names;
}

template struct _jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel<Xbyak::Zmm>;
template struct _jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel<Xbyak::Ymm>;
template struct _jit_aarch64_sve_512_x8s8s32x_1x1_conv_kernel<Xbyak::Xmm>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
