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

#ifndef CPU_AARCH64_JIT_SVE512_CORE_X8S8S32X_1X1_CONV_KERNEL_HPP
#define CPU_AARCH64_JIT_SVE512_CORE_X8S8S32X_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
#include "cpu/aarch64/jit_uni_eltwise_injector.hpp"

#define ADDMAX 4095
#define MOVMAX 65535
#define LDRMAX 255

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

template <typename Vmm>
struct _jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel
    : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            _jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_fwd_ker_t)
    _jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel(
            const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jit_generator(nullptr, 1024 * 1024)
        , jcp(ajcp)
        , attr_(attr)
        , eltwise_injector_(nullptr) {
        if (jcp.with_eltwise) {
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_core>(
                    this, jcp.eltwise);
        }

        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *))this->getCode32();
    }

    ~_jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel() {
        delete eltwise_injector_;
    }

    bool maybe_eltwise(int position);
    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

private:
    jit_uni_eltwise_injector_f32<avx512_core> *eltwise_injector_;

    using reg64_t = const xa::XReg;
    using zmm_t = const xa::ZReg;
    using mask_t = const xa::PReg;

    /* register mapping */
    const xa::XReg reg_last_load = x8;
    const xa::XReg reg_bcast_data = x8;
    const xa::XReg reg_ptr_scales = x8;
    const xa::XReg reg_ptr_saturation_ubound = x8;
    const xa::XReg reg_output_data = x9;
    const xa::XReg reg_load_data = x10;
    const xa::XReg reg_ptr_sum_scale = x10;
    const xa::XReg reg_reduce_loop_work = x11;
    const xa::XReg reg_bias_data = x12;
    const xa::XReg reg_comp_data = x12;
    const xa::XReg reg_scratch = x13;
    const xa::XReg aux_reg_bcast_data = x14;
    const xa::XReg aux_reg_load_data = x15;
    const xa::XReg imm_addr64 = x15;
    const xa::XReg reg_reduce_pos_flag = x0; //rax;
    const xa::XReg aux1_reg_bcast_data = x3; //rbx;
    const xa::XReg reg_bcast_loop_work = x3; //rbx;
    const xa::XReg bcast_loop_iter = x2; //rdx; // Note: Fix me
    const xa::XReg reg_load_loop_work = x6; //rsi;
    const xa::XReg reg_rsp = x4; //rsp;
    const xa::XReg aux_reg_output_data = x1; //abi_not_param1;
    const xa::XReg reduce_loop_iter = x7; //abi_param1;
    const xa::XReg reg_abi_param1 = x7; // abi_param1

    const xa::PReg ktail_mask = p6;
    const xa::PReg vmask = p7;
    const xa::PReg mask_tmp = p8;

    /* Temporay registers */
    const xa::XReg reg_tmp0_imm = x18; // tmp for add_imm
    const xa::XReg reg_tmp1_imm = x19; // tmp for add_imm
    const xa::XReg reg_tmp2_imm = x20; // tmp for add_imm
    const xa::XReg reg_tmp3_imm = x21; // tmp for add_imm
    const xa::XReg reg_tmp0_adr = x23; // tmp for address value
    const xa::XReg reg_tmp1_adr = x24; // tmp for address value
    const xa::XReg reg_tmp2_adr = x25; // tmp for address value
    const xa::XReg reg_tmp3_adr = x26; // tmp for address value

    const xa::ZReg vmm_tmp = xa::ZReg(28);
    const xa::ZReg vmm_saturation = xa::ZReg(28);
    const xa::ZReg vmm_one = xa::ZReg(29);
    const xa::ZReg vmm_zero = xa::ZReg(30);
    const xa::ZReg vmm_prev_dst = xa::ZReg(30);
    const xa::ZReg vmm_shift = xa::ZReg(30);
    const xa::ZReg vmm_bcast = xa::ZReg(31);
    const xa::ZReg vmm_bcast2 = xa::ZReg(30);

    int bcast_loop_work_off = 0;
    int reg_bias_data_off = 8;
    int reg_bcast_data_off = 16;
    int reg_load_data_off = 24;
    int reg_ptr_sum_scale_off = 32;
    int reg_comp_data_off = 40;
    int stack_space_needed = 48;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate();
    // void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
    //         bool mask_flag);

    int get_offset(int raw_offt) {

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

        return re;
    }

    xa::XReg get_comp_addr_reg(xa::XReg base, int offset = 0) {
        auto offt = get_offset(offset);

        if (offt == 0) return base;

        auto reg_tmp_adr = reg_tmp0_adr;
        auto reg_tmp_imm = reg_tmp0_imm;
        CGA64::add_imm(reg_tmp_adr, base, offt, reg_tmp_imm);

        return reg_tmp_adr;
    }
};

struct jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel {
    jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel(
            const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jit_ker(nullptr)
        , zmm_kernel_(nullptr)
        , ymm_kernel_(nullptr)
        , xmm_kernel_(nullptr) {
        int ch_block = ajcp.ic_block;
        switch (ch_block) {
            case 16:
                zmm_kernel_
                        = new _jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel<
                                Xbyak::Zmm>(ajcp, attr);
                jit_ker = zmm_kernel_->jit_ker;
                return;
            case 8:
                ymm_kernel_
                        = new _jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel<
                                Xbyak::Ymm>(ajcp, attr);
                jit_ker = ymm_kernel_->jit_ker;
                return;
            case 4:
                xmm_kernel_
                        = new _jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel<
                                Xbyak::Xmm>(ajcp, attr);
                jit_ker = xmm_kernel_->jit_ker;
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    ~jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel() {
        delete xmm_kernel_;
        delete ymm_kernel_;
        delete zmm_kernel_;
    }

    static bool post_ops_ok(
            jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t *&src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads,
            bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    void (*jit_ker)(jit_1x1_conv_call_s *);
    _jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel<Xbyak::Zmm> *zmm_kernel_;
    _jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel<Xbyak::Ymm> *ymm_kernel_;
    _jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel<Xbyak::Xmm> *xmm_kernel_;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(
            jit_aarch64_sve_512_core_x8s8s32x_1x1_conv_kernel);
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
