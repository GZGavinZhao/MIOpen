/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
; generated by igemm_codegen.py (c8c86649c68a788f18be9f7a599a555f08903048)
;
.macro .mdiv_u32_ss s_quot s_numer s_magic s_shift s_tmp
    s_mul_hi_u32 s[\s_tmp], s[\s_magic], s[\s_numer]
    s_add_u32 s[\s_tmp], s[\s_tmp], s[\s_numer]
    s_lshr_b32 s[\s_quot], s[\s_tmp], s[\s_shift]
.endm

.macro .mdiv_u32_rem_ss s_rem s_quot s_numer s_magic s_shift s_denom s_tmp
    .mdiv_u32_ss \s_quot,\s_numer,\s_magic,\s_shift,\s_tmp
    s_mul_i32 s[\s_tmp], s[\s_denom], s[\s_quot]
    s_sub_u32 s[\s_rem], s[\s_numer], s[\s_tmp]
.endm

.macro .mdiv_u32_vs v_quot v_numer s_magic s_shift v_tmp
    v_mul_hi_u32 v[\v_tmp], s[\s_magic], v[\v_numer]
    v_add_u32 v[\v_tmp], v[\v_tmp], v[\v_numer]
    v_lshrrev_b32 v[\v_quot], s[\s_shift], v[\v_tmp]
.endm

.macro .mdiv_u32_rem_vs v_rem v_quot v_numer s_magic s_shift s_denom v_tmp
    .mdiv_u32_vs \v_quot,\v_numer,\s_magic,\s_shift,\v_tmp
    v_mul_lo_u32 v[\v_tmp], s[\s_denom], v[\v_quot]
    v_sub_u32 v[\v_rem], v[\v_numer], v[\v_tmp]
.endm

.macro .v_clear_acc_c a, num
    _a = \a
    .rept \num
        v_accvgpr_write_b32 a[_a], 0
        _a = _a + 1
    .endr
.endm

.macro .v_clear_nc vid, num
    _v = \vid
    .rept \num
        v_mov_b32 v[_v], 0
        _v = _v + 1
    .endr
.endm

;----------------------------------------------------------
; starting of kernel igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16
; tensor_layout              : 'nchw'
; gemm_m_per_block           : 64
; gemm_n_per_block           : 16
; gemm_k_per_block           : 8
; wave_tile_m                : 64
; wave_step_m                : 1
; wave_repeat_m              : 1
; wave_tile_n                : 4
; wave_step_n                : 2
; wave_repeat_n              : 1
; wave_tile_k                : 4
; tensor_a_thread_lengths    : [1, 2, 2, 1]
; tensor_a_cluster_lengths   : [1, 4, 1, 32]
; tensor_b_thread_lengths    : [1, 1, 1, 1]
; tensor_b_cluster_lengths   : [1, 8, 1, 16]
; direction                  : 'fwd'
; precision                  : 'fp16'
; nxb                        : 1
; nxe                        : 0
; 
; block_size                 : 128
; lds_total                  : 4096
; 
.set k_p_in, 0
.set k_p_wei, 8
.set k_p_out, 16
.set k_hi, 24
.set k_wi, 28
.set k_n, 32
.set k_k, 36
.set k_c, 40
.set k_ho, 44
.set k_wo, 48
.set k_stride_h, 52
.set k_stride_w, 56
.set k_dilation_h, 60
.set k_dilation_w, 64
.set k_pad_h, 68
.set k_pad_w, 72
.set k_y, 76
.set k_x, 80
.set k_group, 84
.set k_magic_0, 88
.set k_magic_1, 92
.set k_magic_2, 96
.set k_magic_3, 100
.set k_magic_4, 104
.set k_magic_5, 108
.set k_magic_6, 112
.set k_shift_pack_0, 116
.set k_shift_pack_1, 120
.set k__pack_0, 124
.set k_end, 128

.set s_ka, 0
.set s_bx, 2
.set s_by, 3
.set s_p_in, 4
.set s_p_wei, 8
.set s_p_out, 12
.set s_hi, 16
.set s_wi, 17
.set s_n, 18
.set s_k, 19
.set s_c, 20
.set s_group, 21
.set s_wei_stride_k0, 22
.set s_stride_hw, 23
.set s_in_stride_c, 24
.set s_in_stride_n, 25
.set s_out_stride_k, 26
.set s_out_stride_n, 27
.set s_in_stride_c_c1, 28
.set s_in_stride_c_c0_c1_diff, 29
.set s_block_gtc_ig, 30
.set s_block_gtc_ik, 31
.set s_block_gtc_in0, 32
.set s_block_gtc_in1b, 33
.set s_move_slice_k_c1e, 34
.set s_knum, 3
.set s_gemm_k_num_c1, 35
.set s_kitr, 1
.set s_in_offset, 36
.set s_wei_offset, 36
.set s_k_padded, 36
.set s_shift_pack_0, 14
.set s_shift_pack_1, 15
.set s_magic_2, 28
.set s_magic_3, 29
.set s_magic_4, 34
.set s_magic_5, 35
.set s_magic_6, 32
.set s_tmp, 38
.set s_magic_0, 10
.set s_magic_1, 11
.set s_end, 44

.set v_c, 0  ; coalescing:8, needed:0, resuable:31
.set v_a, 0
.set v_b, 4
.set v_gld_a, 12
.set v_gld_b, 14
.set v_sst_a_os, 15
.set v_sst_b_os, 16
.set v_sld_a_os, 17
.set v_sld_b_os, 18
.set v_in_os, 19
.set v_in_os_base, 20
.set v_wei_os, 21
.set v_gtc_ta_ik1, 22
.set v_gtc_ta_ik0, 23
.set v_gtc_ta_ic1e, 24
.set v_gtc_ta_ic0, 25
.set v_gtc_tb_in1b, 26
.set v_gtc_tb_in0, 27
.set v_gtc_tb_ic1e, 28
.set v_gtc_tb_in1, 29
.set v_gtc_tb_ib, 30
.set v_co_sst, 31
.set v_co_sld, 32
.set v_out_os, 33
.set v_out_in0, 34
.set v_out_in1b, 35
.set v_out_in1, 36
.set v_in_iho, 37
.set v_in_iwo, 38
.set v_in_ihi, 39
.set v_in_iwi, 40
.set v_move_slice_k_ic1, 28
.set v_gemm_in, 41
.set v_gemm_im, 42
.set v_out_iho, 43
.set v_out_iwo, 44
.set v_co_sub_m_index, 45
.set v_co_sub_n_index, 46
.set v_cur_k, 47
.set v_tmp, 48
.set v_end, 54

.set a_c, 0
.set a_end, 8

.text
.globl igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16
.p2align 8
.type igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16,@function
igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16:
    ; gemm_m_unmerge_cluster:0, gemm_n_unmerge_cluster:0, gemm_k_unmerge_cluster:0
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],    s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx2  s[s_p_wei+0:s_p_wei+1],   s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx2  s[s_p_out+0:s_p_out+1],   s[s_ka+0:s_ka+1],    0+k_p_out
    s_load_dwordx4 s[s_hi+0:s_hi+3],    s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dword s[s_c],    s[s_ka+0:s_ka+1],    0+k_c
    s_load_dword s[s_group],    s[s_ka+0:s_ka+1],    0+k_group
    s_load_dwordx2 s[s_magic_0+0:s_magic_0+1],  s[s_ka+0:s_ka+1],  0+k_magic_0
    s_load_dwordx2 s[s_tmp+2:s_tmp+3],  s[s_ka+0:s_ka+1],  0+k_magic_2
    s_load_dwordx2 s[s_tmp+4:s_tmp+5],  s[s_ka+0:s_ka+1],  0+k_magic_4
    s_load_dword s[s_magic_6],  s[s_ka+0:s_ka+1],  0+k_magic_6
    s_load_dwordx2 s[s_shift_pack_0+0:s_shift_pack_0+1], s[s_ka+0:s_ka+1],  0+k_shift_pack_0
    ; wei(c0, c1e, k0, k1) thread_lengths: 1x2x2x1, cluster_lengths:1x4x1x32
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_ta_ic1e], 3, v[v_tmp]
    v_lshlrev_b32 v[v_gtc_ta_ic1e], 1, v[v_gtc_ta_ic1e]
    v_lshrrev_b32 v[v_tmp], 2, v[v_tmp]
    v_mov_b32 v[v_gtc_ta_ic0], 0
    v_and_b32 v[v_gtc_ta_ik1], 31, v[v_tmp]
    v_lshrrev_b32 v[v_tmp], 5, v[v_tmp]
    v_mov_b32 v[v_gtc_ta_ik0], 0

    ; in(c0, c1e, n0, n1b), thread_lengths: 1x1x1x1, cluster_lengths:1x8x1x16
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_tb_in1b], 15, v[v_tmp]
    v_lshrrev_b32 v[v_tmp], 4, v[v_tmp]
    v_mov_b32 v[v_gtc_tb_in0], 0
    v_and_b32 v[v_gtc_tb_ic1e], 7, v[v_tmp]

    s_mov_b32 s[s_p_in+3], 0x27000
    s_waitcnt lgkmcnt(0)

    s_mov_b32 s[s_magic_2], s[s_tmp+2]
    s_mov_b32 s[s_magic_3], s[s_tmp+3]
    s_mov_b32 s[s_magic_4], s[s_tmp+4]
    s_mov_b32 s[s_magic_5], s[s_tmp+5]
    ; calculate index
    s_lshl_b32 s[s_wei_stride_k0], s[s_c], 5
    s_mul_i32 s[s_stride_hw], s[s_hi], s[s_wi]
    s_mov_b32 s[s_out_stride_k],       s[s_stride_hw]
    s_mov_b32 s[s_in_stride_c],       s[s_stride_hw]
    s_mul_i32 s[s_tmp], s[s_c], s[s_group]
    s_mul_i32 s[s_in_stride_n], s[s_tmp], s[s_stride_hw]
    s_mul_i32 s[s_tmp], s[s_group], s[s_k]
    s_mul_i32 s[s_out_stride_n], s[s_tmp], s[s_stride_hw]
    s_mul_i32  s[s_tmp], s[s_n], s[s_in_stride_n]
    s_mul_i32  s[s_tmp+1], s[s_n], s[s_out_stride_n]
    s_lshl_b32 s[s_tmp+4], s[s_tmp], 1
    s_lshl_b32 s[s_tmp+5], s[s_tmp+1], 1
    s_mul_i32 s[s_tmp], s[s_by], s[s_tmp+4]
    s_mul_hi_u32 s[s_tmp+1], s[s_by], s[s_tmp+4]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]
    s_mul_i32 s[s_tmp], s[s_by], s[s_tmp+5]
    s_mul_hi_u32 s[s_tmp+1], s[s_by], s[s_tmp+5]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]
    s_mov_b32 s[s_knum], s[s_c]

    ; pad k if need
    s_add_u32 s[s_tmp], 63, s[s_k]
    s_lshr_b32 s[s_tmp], s[s_tmp], 6
    s_lshl_b32 s[s_k_padded], s[s_tmp], 6

    ; gemm_m_per_block:64, gemm_n_per_block:16, source_access_order:1
    s_mul_i32 s[s_tmp], s[s_stride_hw], s[s_n]
    s_mul_i32 s[s_tmp+1], s[s_tmp], s[s_k_padded]
    s_lshr_b32 s[0], s[s_tmp+1], 10
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080010 ; offset:16, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_block_gtc_ig,s_bx,s_magic_6,s_tmp+3,0,s_tmp
    s_mov_b32 s[s_bx], s[s_tmp+4]
    s_lshr_b32 s[0], s[s_k_padded], 6
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080000 ; offset:0, width:8
    .mdiv_u32_rem_ss s_tmp+5,s_tmp+4,s_bx,s_magic_0,s_tmp+3,0,s_tmp
    ; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im
    s_lshl_b32 s[s_block_gtc_ik], s[s_tmp+5], 6
    s_mov_b32 s[0], s[s_stride_hw] ; total number of n1b
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080008 ; offset:8, width:8
    .mdiv_u32_rem_ss s_block_gtc_in1b,s_block_gtc_in0,s_tmp+4,s_magic_1,s_tmp+3,0,s_tmp
    s_lshl_b32 s[s_block_gtc_in1b], s[s_block_gtc_in1b], 4

    ; in c1e transform
    ; in n1b transform
    v_add_u32 v[v_tmp+5], s[s_block_gtc_in1b], v[v_gtc_tb_in1b]
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080000 ; offset:0, width:8
    .mdiv_u32_rem_vs v_tmp+4,v_gtc_tb_in1,v_tmp+5,s_magic_4,s_tmp+3,s_stride_hw,v_tmp
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080008 ; offset:8, width:8
    .mdiv_u32_rem_vs v_in_iwi,v_in_ihi,v_tmp+4,s_magic_5,s_tmp+3,s_wi,v_tmp
    ; calculate in offset
    s_mul_i32 s[s_p_in+2], s[s_in_stride_n], s[s_n]
    s_lshl_b32 s[s_p_in+2], s[s_p_in+2], 1
    s_mul_i32 s[s_tmp+5], s[s_c], s[s_in_stride_c]
    s_lshl_b32 s[s_block_gtc_ig], s[s_block_gtc_ig], 1
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_tmp+5]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_tmp+5]
    s_sub_u32 s[s_p_in+2], s[s_p_in+2], s[s_tmp]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]
    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_in0], 5
    s_mul_i32 s[s_tmp], s[s_in_stride_n], s[s_tmp+3]
    s_mul_hi_u32 s[s_tmp+1], s[s_in_stride_n], s[s_tmp+3]
    s_sub_u32 s[s_p_in+2], s[s_p_in+2], s[s_tmp]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]

    v_mov_b32 v[v_tmp], v[v_gtc_tb_ic1e]
    v_mul_lo_u32 v[v_tmp], s[s_in_stride_c], v[v_tmp]
    v_mov_b32 v[v_tmp+1], v[v_gtc_tb_in1]
    v_mul_lo_u32 v[v_tmp+1], s[s_in_stride_n], v[v_tmp+1]
    v_add_lshl_u32 v[v_tmp+4], v[v_tmp], v[v_tmp+1], 1
    v_mad_u32_u24 v[v_tmp], v[v_in_ihi], s[s_wi], v[v_in_iwi]
    v_lshl_add_u32 v[v_in_os], v[v_tmp], 1, v[v_tmp+4]


    
    ; load input
    buffer_load_short_d16 v[v_gld_b+0], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:0

    ; config for weight range
    s_mul_i32 s[s_p_wei+2], s[s_c], s[s_k]
    s_lshl_b32 s[s_p_wei+2], s[s_p_wei+2], 1
    s_mov_b32 s[s_p_wei+3], 0x27000
    ; calculate wei offset
    s_mul_i32 s[s_tmp+2], s[s_k], s[s_c]
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_tmp+2]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_tmp+2]
    s_add_u32 s[s_p_wei], s[s_p_wei], s[s_tmp]
    s_addc_u32 s[s_p_wei+1], s[s_p_wei+1], s[s_tmp+1]
    v_mov_b32 v[v_tmp], v[v_gtc_ta_ik1]
    v_add_u32 v[v_cur_k], s[s_block_gtc_ik], v[v_tmp]
    v_mul_lo_u32 v[v_tmp], s[s_c], v[v_cur_k]
    v_add_lshl_u32 v[v_wei_os], v[v_tmp], v[v_gtc_ta_ic1e], 1

    s_lshl_b32 s[s_wei_stride_k0], s[s_wei_stride_k0], 1

    
    ; load weight
    buffer_load_dword v[v_gld_a+0], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    buffer_load_dword v[v_gld_a+1], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_wei_stride_k0] offen offset:0

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get source matrix gemm index
    v_and_b32 v[v_gemm_in], 3, v[v_tmp+5]           ; block_n index 
    v_and_b32 v[v_gemm_im], 3, v[v_tmp+5]           ; block_m index 
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]
    v_and_b32 v[v_tmp + 1], 15, v[v_tmp+5]          ; block_m_per_wave index
    v_lshl_or_b32 v[v_gemm_im], v[v_tmp + 1], 2, v[v_gemm_im]
    v_lshrrev_b32 v[v_tmp+5], 4, v[v_tmp+5]
    v_and_b32 v[v_tmp + 2], 1, v[v_tmp+5]  ; waves_per_n index
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp + 2], 3, v[v_gemm_in]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get dst matrix gemm index
    v_and_b32 v[v_tmp+0], 3, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]
    v_and_b32 v[v_tmp+3], 15, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 4, v[v_tmp+5]
    v_mov_b32 v[v_co_sst], v[v_tmp+0]
    v_lshlrev_b32 v[v_co_sld], 2, v[v_tmp+3]
    v_and_b32 v[v_tmp+0], 1, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_lshl_or_b32 v[v_co_sst], v[v_tmp+0], 3, v[v_co_sst]

    ; LDS store, in: c0,c1e,n0,n1b: 1x1x1x1, 1x8x1x16, order:4
    v_lshlrev_b32 v[v_tmp], 2, v[v_gtc_tb_in1b]
    v_lshrrev_b32 v[v_tmp+1], 2, v[v_gtc_tb_ic1e]
    v_lshl_add_u32 v[v_tmp], v[v_tmp+1], 6, v[v_tmp]
    v_and_b32 v[v_tmp+1], 3, v[v_gtc_tb_ic1e]
    v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp+1]
    v_lshlrev_b32 v[v_sst_b_os], 1, v[v_tmp]
    v_add_u32 v[v_sst_b_os], 1024, v[v_sst_b_os]

    ; LDS store, wei: c0,c1e,c0,c1: 1x2x2x1, 1x4x1x32, order:0
    v_lshlrev_b32 v[v_tmp], 2, v[v_gtc_ta_ik1]
    v_lshrrev_b32 v[v_tmp+1], 2, v[v_gtc_ta_ic1e]
    v_lshl_add_u32 v[v_tmp], v[v_tmp+1], 8, v[v_tmp]
    v_and_b32 v[v_tmp+1], 3, v[v_gtc_ta_ic1e]
    v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp+1]
    v_lshlrev_b32 v[v_sst_a_os], 1, v[v_tmp]

    ; LDS load
    v_lshlrev_b32 v[v_sld_b_os], 3, v[v_gemm_in]
    v_lshlrev_b32 v[v_sld_a_os], 3, v[v_gemm_im]
    v_add_u32 v[v_sld_b_os], 1024, v[v_sld_b_os]

    v_mov_b32 v[v_gemm_in], v[v_co_sst]
    v_mov_b32 v[v_gemm_im], v[v_co_sld]
    ; init_co_lds_offset for xdlops
    v_lshrrev_b32 v[v_tmp], 2, v[v_gemm_im]
    v_and_b32 v[v_tmp], 0, v[v_tmp]   ; thread id of lanegroup_m_per_cluster
    v_lshlrev_b32 v[v_co_sst], 2, v[v_tmp]
    v_lshrrev_b32 v[v_tmp+1], 2, v[v_gemm_im]
    v_and_b32 v[v_tmp+1], 15  , v[v_tmp+1]   ; thread id of block_m_per_lanegroup
    v_lshl_or_b32 v[v_co_sst], v[v_tmp+1], 2, v[v_co_sst]
    v_lshrrev_b32 v[v_tmp], 2, v[v_co_sst]
    v_lshlrev_b32 v[v_tmp+1], 2, v[v_gemm_in]   ; implicit transpose with m granularity:4 while store
    v_lshl_or_b32 v[v_co_sst], v[v_tmp], 6, v[v_tmp+1]
    v_lshlrev_b32 v[v_co_sst], 1, v[v_co_sst]
    v_lshlrev_b32 v[v_co_sld], 3, v[0]
    ; init_co_sub_m_index xdlops, block_size:128, macro-tile:64x16 sub_m_index:[0, 4, 8, 12, 16, 20, 24, 28]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:1, l_ms:1, l_mw:1, l_mb:1, l_mt:4 | n_mc:1, n_ml:16, n_mv:1
    ; nd_stride:[1, 16, 1, 1, 1, 1, 1]
    v_lshrrev_b32 v[v_co_sub_m_index], 4, v[0]   ; get tid along m
    v_and_b32 v[v_tmp+0], 15, v[v_co_sub_m_index]                   ; => x_ml
    v_mov_b32 v[v_co_sub_m_index], v[v_tmp+0]      ; => accumulate x_ml
    v_lshlrev_b32 v[v_co_sub_m_index], 2, v[v_co_sub_m_index]
    ; init_co_sub_n_index xdlops
    v_and_b32 v[v_co_sub_n_index], 15, v[0]

    ; output offset
    s_mul_i32 s[s_tmp+2], s[s_k], s[s_out_stride_k]
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_tmp+2]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_tmp+2]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]
    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_in0], 5
    s_mul_i32 s[s_tmp], s[s_out_stride_n], s[s_tmp+3]
    s_mul_hi_u32 s[s_tmp+1], s[s_out_stride_n], s[s_tmp+3]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]

    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_ik], 1
    s_mul_i32 s[s_tmp], s[s_out_stride_k], s[s_tmp+3]
    s_mul_hi_u32 s[s_tmp+1], s[s_out_stride_k], s[s_tmp+3]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]

    ; compute v_co_sub_n_index along n0 x n1b : 1x16
    v_and_b32 v[v_out_in1b], 15, v[v_co_sub_n_index]     ; => N1B
    ;   compute from n1b
    v_add_u32 v[v_tmp+5], s[s_block_gtc_in1b], v[v_out_in1b]
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080000 ; offset:0, width:8
    .mdiv_u32_rem_vs v_tmp+4,v_out_in1,v_tmp+5,s_magic_4,s_tmp+3,s_stride_hw,v_tmp
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_1], 0x00080008 ; offset:8, width:8
    .mdiv_u32_rem_vs v_out_iwo,v_out_iho,v_tmp+4,s_magic_5,s_tmp+3,s_wi,v_tmp


    ; add in_in0, in_in1
    v_mul_lo_u32 v[v_out_os], s[s_out_stride_n], v[v_out_in1]
    ; add i_k
    v_mul_lo_u32 v[v_tmp], s[s_out_stride_k], v[v_co_sub_m_index]
    v_add_u32 v[v_out_os], v[v_out_os], v[v_tmp]
    ; add ho, wo
    v_mul_lo_u32 v[v_tmp+1], s[s_wi], v[v_out_iho]
    v_add3_u32 v[v_out_os], v[v_out_os], v[v_tmp+1], v[v_out_iwo]
    v_lshlrev_b32 v[v_out_os], 1, v[v_out_os]
    ; move slice stride
    s_mov_b32 s[s_move_slice_k_c1e], 8

    s_mul_i32 s[s_in_stride_c_c1], s[s_move_slice_k_c1e], s[s_stride_hw]  ; might be 0 or larger

    s_lshl_b32 s[s_in_stride_c_c1], s[s_in_stride_c_c1], 1
    s_lshl_b32 s[s_in_stride_c_c0_c1_diff], s[s_in_stride_c_c0_c1_diff], 1
    s_lshl_b32 s[s_in_stride_c], s[s_in_stride_c], 1
    s_lshl_b32 s[s_c], s[s_c], 1
    s_lshl_b32 s[s_out_stride_k], s[s_out_stride_k], 1
    s_lshl_b32 s[s_move_slice_k_c1e], s[s_move_slice_k_c1e], 1
    s_mov_b32 s[s_p_out+2], 0xffffffff
    s_mov_b32 s[s_p_out+3], 0x27000
    ; start MFMA loop, 64x4 wave tile with 1x1 repeat, 1x2 step
    s_waitcnt vmcnt(2)
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+0] 

    s_waitcnt vmcnt(0)
    ds_write2_b32 v[v_sst_a_os], v[v_gld_a], v[v_gld_a+1], offset0:0, offset1:64

    .v_clear_acc_c a_c, 8
    ; make sure acc WAR harzard, at least 1 nop for src_c
    s_sub_i32 s[s_kitr], s[s_knum], 8
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16_mfma_end

    v_add_u32 v[v_in_os], s[s_in_stride_c_c1], v[v_in_os]
    ; move slice window for weight
    v_add_u32 v[v_wei_os],  s[s_move_slice_k_c1e], v[v_wei_os]
    s_waitcnt lgkmcnt(0)
    s_barrier
L_igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16_mfma_body:
    ; do fma accumulate with unroll 8
    ds_read_b64 v[v_a:v_a+1], v[v_sld_a_os] 
    ds_read2_b64 v[v_b+0:v_b+3], v[v_sld_b_os], offset0:0, offset1:4
    ds_read_b64 v[v_a+2:v_a+2+1], v[v_sld_a_os] offset:512
    ds_read2_b64 v[v_b+4+0:v_b+4+3], v[v_sld_b_os], offset0:16, offset1:20
    ; load input
    buffer_load_short_d16 v[v_gld_b+0], v[v_in_os], s[s_p_in:s_p_in+3], 0 offen offset:0
    ; load weight
    buffer_load_dword v[v_gld_a+0], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    buffer_load_dword v[v_gld_a+1], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_wei_stride_k0] offen offset:0
    v_add_u32 v[v_in_os], s[s_in_stride_c_c1], v[v_in_os]
    ; move slice window for weight
    v_add_u32 v[v_wei_os],  s[s_move_slice_k_c1e], v[v_wei_os]
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_waitcnt vmcnt(2)
    ds_write_b16 v[v_sst_b_os], v[v_gld_b+0]
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+0:v_a+1], v[v_b+0:v_b+1], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    s_waitcnt vmcnt(0)
    ds_write2_b32 v[v_sst_a_os], v[v_gld_a], v[v_gld_a+1], offset0:0, offset1:64
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+0:v_a+1], v[v_b+2:v_b+3], a[a_c+4:a_c+7]     ; repeat:0x0, step:0x1, num_a_c:4
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+2:v_a+3], v[v_b+4:v_b+5], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+2:v_a+3], v[v_b+6:v_b+7], a[a_c+4:a_c+7]     ; repeat:0x0, step:0x1, num_a_c:4
    s_sub_i32 s[s_kitr], s[s_kitr], 8
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16_mfma_finishing
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_branch L_igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16_mfma_body
L_igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16_mfma_finishing:
L_igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16_mfma_end:
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b64 v[v_a:v_a+1], v[v_sld_a_os] 
    ds_read2_b64 v[v_b+0:v_b+3], v[v_sld_b_os], offset0:0, offset1:4
    ds_read_b64 v[v_a+2:v_a+2+1], v[v_sld_a_os] offset:512
    ds_read2_b64 v[v_b+4+0:v_b+4+3], v[v_sld_b_os], offset0:16, offset1:20
    
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+0:v_a+1], v[v_b+0:v_b+1], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+0:v_a+1], v[v_b+2:v_b+3], a[a_c+4:a_c+7]     ; repeat:0x0, step:0x1, num_a_c:4
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_4x4x4f16 a[a_c+0:a_c+3], v[v_a+2:v_a+3], v[v_b+4:v_b+5], a[a_c+0:a_c+3]     ; repeat:0x0, step:0x0, num_a_c:4
    v_mfma_f32_4x4x4f16 a[a_c+4:a_c+7], v[v_a+2:v_a+3], v[v_b+6:v_b+7], a[a_c+4:a_c+7]     ; repeat:0x0, step:0x1, num_a_c:4
    s_nop 3
    ; coalescing store, mapping:mt_m:64, mt_n:16, wt_m:64, wt_n:4, ws:2, r_m:1, r_n:1, s_m:1, s_n:2 | 4x4x4, lanegroup_m_tcbw:4x1x1x1, lanegroup_n_tcbw:1x4x1x1
    ; coalescing_groups:1, num_dword_per_group:8
    ; init_co_sub_m_index xdlops, block_size:128, macro-tile:64x16 sub_m_index:[0, 4, 8, 12, 16, 20, 24, 28]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:1, l_ms:1, l_mw:1, l_mb:1, l_mt:4 | n_mc:1, n_ml:16, n_mv:1
    ; nd_stride:[1, 16, 1, 1, 1, 1, 1]
    ; start group 0, i_g_mr:0, i_g_ms:0, i_g_mw:0, i_g_mb:0, i_g_mt:0, m index start from 0
    s_barrier
    v_accvgpr_read_b32 v[v_c], a[a_c]
    v_accvgpr_read_b32 v[v_c+1], a[a_c+1]
    v_accvgpr_read_b32 v[v_c+2], a[a_c+2]
    v_accvgpr_read_b32 v[v_c+3], a[a_c+3]
    v_cvt_f16_f32_e32 v[v_c], v[v_c]
    v_cvt_f16_f32_e32 v[v_c+1], v[v_c+1]
    v_cvt_f16_f32_e32 v[v_c+2], v[v_c+2]
    v_cvt_f16_f32_e32 v[v_c+3], v[v_c+3]
    v_pack_b32_f16 v[v_c], v[v_c], v[v_c+1]
    v_pack_b32_f16 v[v_c+1], v[v_c+2], v[v_c+3]
    ds_write_b64 v[v_co_sst], v[v_c:v_c+1]    ; idword:0(0,0),  0x0 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    v_accvgpr_read_b32 v[v_c+4], a[a_c+4]
    v_accvgpr_read_b32 v[v_c+5], a[a_c+5]
    v_accvgpr_read_b32 v[v_c+6], a[a_c+6]
    v_accvgpr_read_b32 v[v_c+7], a[a_c+7]
    v_cvt_f16_f32_e32 v[v_c+4], v[v_c+4]
    v_cvt_f16_f32_e32 v[v_c+5], v[v_c+5]
    v_cvt_f16_f32_e32 v[v_c+6], v[v_c+6]
    v_cvt_f16_f32_e32 v[v_c+7], v[v_c+7]
    v_pack_b32_f16 v[v_c+4], v[v_c+4], v[v_c+5]
    v_pack_b32_f16 v[v_c+5], v[v_c+6], v[v_c+7]
    ds_write_b64 v[v_co_sst], v[v_c+4:v_c+4+1] offset:32   ; idword:4(0,4),  0x4 | /4, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:1, i_nw:0
    s_waitcnt lgkmcnt(0)
    s_barrier
    ;   load from lds
    ds_read_b64 v[v_c:v_c+1], v[v_co_sld] 
    ds_read_b64 v[v_c+2:v_c+2+1], v[v_co_sld] offset:1024
    ;   store to global, m index start from 0, m0:0, m1:0
    s_mov_b32 s[s_tmp], 0   ; i_m:0(i_m0:0,i_m1:0)
    v_add_u32 v[v_cur_k], s[s_block_gtc_ik], v[v_co_sub_m_index]
    v_mov_b32 v[v_tmp], v[v_cur_k]
    s_waitcnt lgkmcnt(1)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_short v[v_c], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mov_b32 s[s_tmp], s[s_out_stride_k]   ; i_m:1(i_m0:0,i_m1:1)
    v_add_u32 v[v_tmp], 1, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_short_d16_hi v[v_c], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 2, s[s_out_stride_k]   ; i_m:2(i_m0:0,i_m1:2)
    v_add_u32 v[v_tmp], 2, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_short v[v_c+1], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 3, s[s_out_stride_k]   ; i_m:3(i_m0:0,i_m1:3)
    v_add_u32 v[v_tmp], 3, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_short_d16_hi v[v_c+1], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 32, s[s_out_stride_k]   ; i_m:32(i_m0:1,i_m1:0)
    v_add_u32 v[v_tmp], 32, v[v_cur_k]
    s_waitcnt lgkmcnt(0)
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_short v[v_c+2], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 33, s[s_out_stride_k]   ; i_m:33(i_m0:1,i_m1:1)
    v_add_u32 v[v_tmp], 33, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_short_d16_hi v[v_c+2], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 34, s[s_out_stride_k]   ; i_m:34(i_m0:1,i_m1:2)
    v_add_u32 v[v_tmp], 34, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_short v[v_c+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 35, s[s_out_stride_k]   ; i_m:35(i_m0:1,i_m1:3)
    v_add_u32 v[v_tmp], 35, v[v_cur_k]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_short_d16_hi v[v_c+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
L_igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16_out:
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 54
    .amdhsa_next_free_sgpr 44
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16
    .symbol: igemm_fwd_gtcx_nchw_fp16_bx1_ex0_bt64x16x8_wt64x4x4_ws1x2_wr1x1_ta1x2x2x1_1x4x1x32_tb1x1x1x1_1x8x1x16.kd
    .sgpr_count: 50
    .vgpr_count: 54
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [128, 1, 1]
    .max_flat_workgroup_size: 128
    .args:
    - { .name: p_in      , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_wei     , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out     , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: hi        , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi        , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n         , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k         , .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: c         , .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ho        , .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: wo        , .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: stride_h  , .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: stride_w  , .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_h, .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_w, .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32}
    - { .name: pad_h     , .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32}
    - { .name: pad_w     , .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32}
    - { .name: y         , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x         , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: group     , .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: magic_0   , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1   , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2   , .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32}
    - { .name: magic_3   , .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    - { .name: magic_4   , .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    - { .name: magic_5   , .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
    - { .name: magic_6   , .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0, .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_1, .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0  , .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
