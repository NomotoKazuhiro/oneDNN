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
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_AARCH64_JIT_SVE_512_X8S8S32X_CONV_KERNEL_HPP
#define CPU_AARCH64_JIT_SVE_512_X8S8S32X_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
#include "cpu/aarch64/jit_uni_eltwise_injector.hpp"

#define ADDMAX 4095
#define MOVMAX 65535

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

template <typename Vmm>
struct _jit_aarch64_sve_512_x8s8s32x_fwd_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_aarch64_sve_512_x8s8s32x_conv_fwd_ker_t)

    enum { STATE_FIRST_DST_LOAD = 0x1U };

    _jit_aarch64_sve_512_x8s8s32x_fwd_kernel(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_core>(
                    this, jcp.eltwise);

        generate();
        jit_ker_ = (void (*)(jit_conv_call_s *))getCode();
    }

    ~_jit_aarch64_sve_512_x8s8s32x_fwd_kernel() { delete eltwise_injector_; }

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker_)(jit_conv_call_s *);

private:
    jit_uni_eltwise_injector_f32<avx512_core> *eltwise_injector_;

    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
        ker_dw_reg_base_idx = 30,
    };
    typedef enum {
        no_last_block,
        last_ic_block,
        last_sp_block,
    } ic_block_t;

    /* data regs */
    const xa::XReg reg_ptr_scales = x0; //rax;
    const xa::XReg aux_reg_saturation = x0; //rax;
    const xa::XReg reg_inp = x8; //r8;
    const xa::XReg reg_ker = x9; //r9;
    const xa::XReg reg_out = x10; //r10;
    const xa::XReg aux_reg_inp = x11; //r11;
    const xa::XReg reg_ptr_sum_scale = x11; //r11;
    const xa::XReg aux_reg_ker = x12; //r12;
    const xa::XReg reg_compensation = x14; //r14;
    const xa::XReg aux_reg_inp_d = x13; //r13;
    const xa::XReg aux_reg_ker_d = x15; //r15;
    // Using 3d regs as depthwise3d is not yet supported
    const xa::XReg reg_inp_buffer_ptr = aux_reg_inp_d;
    const xa::XReg aux_reg_inp_buffer_ptr = aux_reg_ker_d;

    /* counter regs */
    const xa::XReg reg_bias_alpha = x1; //abi_not_param1;
    const xa::XReg reg_param1 = x7; //abi_param1;
    const xa::XReg reg_oi = x3; //rbx;
    const xa::XReg reg_bias = x2; //rdx;
    const xa::XReg reg_oc_blocks = x6; //rsi;
    const xa::XReg reg_owb = aux_reg_ker;
    const xa::XReg reg_scratch = reg_compensation;
    const xa::XReg reg_kj = reg_ptr_scales;
    const xa::XReg reg_ki = reg_compensation;
    const xa::XReg reg_overflow = reg_ptr_scales;
    const xa::XReg reg_icb = reg_bias;

    xa::XReg reg_stack = x22; // translator stack register

    /* Temporay registers */
    xa::XReg reg_tmp0_imm = x18; // tmp for add_imm
    xa::XReg reg_tmp1_imm = x19; // tmp for add_imm
    xa::XReg reg_tmp2_imm = x20; // tmp for add_imm
    xa::XReg reg_tmp3_imm = x21; // tmp for add_imm
    xa::XReg reg_tmp0_adr = x23; // tmp for address value
    xa::XReg reg_tmp1_adr = x24; // tmp for address value
    xa::XReg reg_tmp2_adr = x25; // tmp for address value
    xa::XReg reg_tmp3_adr = x26; // tmp for address value

    const xa::PReg ktail_mask = p2;
    const xa::PReg kblend_mask = p3;

    const xa::PReg mask_tmp = p7;
    const xa::PReg mask_tmp2 = p6;
    const xa::PReg mask_all_one = p4;

    const xa::ZReg vmm_wei = xa::ZReg(31);
    /* used during bias section of store_output */
    const xa::ZReg vmm_comp = xa::ZReg(30); // only for signed input
    const xa::ZReg vmm_bias = xa::ZReg(31);
    /* used during post_op sum section of store_output */
    const xa::ZReg vmm_prev_dst = xa::ZReg(31);
    /* used during write-out section of store_output */
    const xa::ZReg vmm_saturation = xa::ZReg(30);
    const xa::ZReg vmm_zero = xa::ZReg(31);

    /* used in compute_ker (but set during prepare_output) */
    const xa::ZReg vmm_shift = vmm_comp; // only for signed input
    /* used in compute_ker (but only for pre-VNNI machines) */
    const xa::ZReg vmm_tmp = xa::ZReg(28); // not used for depthwise
    const xa::ZReg vmm_one
            = xa::ZReg(29); // set at start of kernel, not used for depthwise.

    /* registers use only for depthwise
       groups are always blocked by 16(padded if needed),
       hence use only Zmm registers */
    const xa::ZReg zmm_wei = xa::ZReg(31);
    xa::ZReg zmm_tmp = xa::ZReg(0);
    xa::ZReg zmm_src = xa::ZReg(0);
    xa::ZReg zmm_shifted_zero = xa::ZReg(0);
    xa::ZReg zmm_permute = xa::ZReg(0);

    bool mask_gflag;

    xa::ZReg vmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx
                < (jcp.is_depthwise ? ker_dw_reg_base_idx : ker_reg_base_idx));
        return xa::ZReg(idx);
    }
    xa::ZReg zmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx
                < (jcp.is_depthwise ? ker_dw_reg_base_idx : ker_reg_base_idx));
        return xa::ZReg(idx);
    }
    xa::ZReg vmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return xa::ZReg(idx);
    }
    xa::ZReg zmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return xa::ZReg(idx);
    }
    xa::ZReg vmm_bias_alpha() {
        int nb_c_block
                = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
        return xa::ZReg(nb_c_block * jcp.ur_w);
    }
    xa::ZReg xmm_bias_alpha() {
        int nb_c_block
                = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
        return xa::ZReg(nb_c_block * jcp.ur_w);
    }
    int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }
    int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w
                - nstl::max(0,
                        utils::div_up(
                                pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1),
                                jcp.stride_w));
    }

    bool maybe_eltwise(int position);
    void prepare_output(int ur_w);
    void store_output(int ur_w, bool last_oc_block_flag);
    void compute_ker_dw(int ur_w, int pad_l, int pad_r,
            ic_block_t last_ic_block_flag, bool h_padded);
    void compute_ker(int ur_w, int pad_l, int pad_r,
            ic_block_t last_ic_block_flag, bool h_padded = false);
    void compute_eltwise(int ur_w);
    void kh_loop(int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag);
    void icb_loop(int ur_w, int pad_l, int pad_r, bool is_last_spatial_block);
    void generate();
    void cvt2ps(data_type_t type_in, xa::ZReg ymm_in, const xa::XReg reg_base,
            const int offset, bool mask_flag);
    // const Vmm vmm_mask(const Vmm vmm_in, bool mask_flag, bool store = false);
    void vmm_mask_all_one();
    void vmm_load_src(xa::ZReg src, xa::XReg reg_addr, bool mask_flag);

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

struct jit_aarch64_sve_512_x8s8s32x_fwd_kernel {

    jit_aarch64_sve_512_x8s8s32x_fwd_kernel(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jit_ker(nullptr)
        , zmm_kernel_(nullptr)
        , ymm_kernel_(nullptr)
        , xmm_kernel_(nullptr) {
        int ch_block = ajcp.is_depthwise ? ajcp.ch_block : ajcp.ic_block;
        switch (ch_block) {
            case 16:
                zmm_kernel_ = new _jit_aarch64_sve_512_x8s8s32x_fwd_kernel<
                        Xbyak::Zmm>(ajcp, attr);
                jit_ker = zmm_kernel_->jit_ker_;
                return;
            case 8:
                ymm_kernel_ = new _jit_aarch64_sve_512_x8s8s32x_fwd_kernel<
                        Xbyak::Ymm>(ajcp, attr);
                jit_ker = ymm_kernel_->jit_ker_;
                return;
            case 4:
                xmm_kernel_ = new _jit_aarch64_sve_512_x8s8s32x_fwd_kernel<
                        Xbyak::Xmm>(ajcp, attr);
                jit_ker = xmm_kernel_->jit_ker_;
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    ~jit_aarch64_sve_512_x8s8s32x_fwd_kernel() {
        delete xmm_kernel_;
        delete ymm_kernel_;
        delete zmm_kernel_;
    }

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, const primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    void (*jit_ker)(jit_conv_call_s *);
    _jit_aarch64_sve_512_x8s8s32x_fwd_kernel<Xbyak::Zmm> *zmm_kernel_;
    _jit_aarch64_sve_512_x8s8s32x_fwd_kernel<Xbyak::Ymm> *ymm_kernel_;
    _jit_aarch64_sve_512_x8s8s32x_fwd_kernel<Xbyak::Xmm> *xmm_kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
