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

#ifndef CPU_AARCH64_CPU_ISA_TRAITS_HPP
#define CPU_AARCH64_CPU_ISA_TRAITS_HPP

#include <type_traits>

#include "dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#define XBYAK64
#define XBYAK_NO_OP_NAMES
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* turn off `size_t to other-type implicit casting` warning
 * currently we have a lot of jit-generated instructions that
 * take uint32_t, but we pass size_t (e.g. due to using sizeof).
 * FIXME: replace size_t parameters with the appropriate ones */
#pragma warning(disable : 4267)
#endif

#include "cpu/aarch64/xbyak_translator_aarch64/xbyak/xbyak.h"
#include "cpu/aarch64/xbyak_translator_aarch64/xbyak/xbyak_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

enum cpu_isa_bit_t : unsigned {
    sse41_bit = 1u << 0,
    avx_bit = 1u << 1,
    avx2_bit = 1u << 2,
    avx512_common_bit = 1u << 3,
    avx512_mic_bit = 1u << 4,
    avx512_mic_4ops_bit = 1u << 5,
    avx512_core_bit = 1u << 6,
    avx512_core_vnni_bit = 1u << 7,
    avx512_core_bf16_bit = 1u << 8,
    simd_bit = 1u << 9,
    sve256_bit = 1u << 10,
    sve_bit = 1u << 11,
};

enum cpu_isa_t : unsigned {
    isa_any = 0u,
    sse41 = sse41_bit,
    avx = avx_bit | sse41,
    avx2 = avx2_bit | avx,
    avx512_common = avx512_common_bit | avx2,
    avx512_mic = avx512_mic_bit | avx512_common,
    avx512_mic_4ops = avx512_mic_4ops_bit | avx512_mic,
    avx512_core = avx512_core_bit | avx512_common,
    avx512_core_vnni = avx512_core_vnni_bit | avx512_core,
    avx512_core_bf16 = avx512_core_bf16_bit | avx512_core_vnni,
    simd = simd_bit | avx512_core_bf16,
    sve256 = sve256_bit | simd,
    sve = sve_bit | sve256,
    isa_all = ~0u,
};

const char *get_isa_info();

cpu_isa_t DNNL_API get_max_cpu_isa_mask(bool soft = false);
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
dnnl_cpu_isa_t get_effective_cpu_isa();

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_all;
    static constexpr const char *user_option_env = "ALL";
};

template <>
struct cpu_isa_traits<sse41> {
    typedef Xbyak::Xmm Vmm;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_sse41;
    static constexpr const char *user_option_env = "SSE41";
};

template <>
struct cpu_isa_traits<avx> {
    typedef Xbyak::Ymm Vmm;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx;
    static constexpr const char *user_option_env = "AVX";
};

template <>
struct cpu_isa_traits<avx2> : public cpu_isa_traits<avx> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2;
    static constexpr const char *user_option_env = "AVX2";
};

template <>
struct cpu_isa_traits<avx512_common> {
    typedef Xbyak::Zmm Vmm;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
};

template <>
struct cpu_isa_traits<avx512_core> : public cpu_isa_traits<avx512_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_core;
    static constexpr const char *user_option_env = "AVX512_CORE";
};

template <>
struct cpu_isa_traits<avx512_mic> : public cpu_isa_traits<avx512_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_mic;
    static constexpr const char *user_option_env = "AVX512_MIC";
};

template <>
struct cpu_isa_traits<avx512_mic_4ops> : public cpu_isa_traits<avx512_mic> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_mic_4ops;
    static constexpr const char *user_option_env = "AVX512_MIC_4OPS";
};

template <>
struct cpu_isa_traits<avx512_core_vnni> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_vnni;
    static constexpr const char *user_option_env = "AVX512_CORE_VNNI";
};

template <>
struct cpu_isa_traits<avx512_core_bf16> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_bf16;
    static constexpr const char *user_option_env = "AVX512_CORE_BF16";
};

template <>
struct cpu_isa_traits<simd> {
    typedef Xbyak::Xbyak_aarch64::VReg4S Vmm;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_simd;
    static constexpr const char *user_option_env = "SIMD";
};

template <>
struct cpu_isa_traits<sve256> {
    typedef Xbyak::Xbyak_aarch64::ZRegS Vmm;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_sve256;
    static constexpr const char *user_option_env = "SVE256";
};

template <>
struct cpu_isa_traits<sve> {
    typedef Xbyak::Xbyak_aarch64::ZRegS Vmm;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_sve;
    static constexpr const char *user_option_env = "SVE";
};

namespace {

static Xbyak::util::Cpu cpu;
static inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak::util;

    unsigned cpu_isa_mask = aarch64::get_max_cpu_isa_mask(soft);
    if ((cpu_isa_mask & cpu_isa) != cpu_isa) return false;

    switch (cpu_isa) {
        case sse41: return cpu.has(Cpu::tSSE41);
        case avx: return cpu.has(Cpu::tAVX);
        case avx2: return cpu.has(Cpu::tAVX2);
        case avx512_common: return cpu.has(Cpu::tAVX512F);
        case avx512_core:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW)
                    && cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
        case avx512_core_vnni:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW)
                    && cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ)
                    && cpu.has(Cpu::tAVX512_VNNI);
        case avx512_mic:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD)
                    && cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);
        case avx512_mic_4ops:
            return mayiuse(avx512_mic, soft) && cpu.has(Cpu::tAVX512_4FMAPS)
                    && cpu.has(Cpu::tAVX512_4VNNIW);
        case avx512_core_bf16:
            return mayiuse(avx512_core_vnni, soft)
                    && cpu.has(Cpu::tAVX512_BF16);
        case simd: return true && cpu.has(Cpu::tSIMD);
        case sve256: return true && cpu.has(Cpu::tSVE);
        case sve: return true && cpu.has(Cpu::tSVE);
        case isa_any: return true;
        case isa_all: return false;
    }
    return false;
}

inline bool isa_has_bf16(cpu_isa_t isa) {
    return false;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_any ? prefix STRINGIFY(any) : \
    ((isa) == simd ? prefix STRINGIFY(simd) : \
    ((isa) == sve256 ? prefix STRINGIFY(sve256) : \
    ((isa) == sve ? prefix STRINGIFY(sve) : \
    prefix suffix_if_any))))
/* clang-format on */

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
