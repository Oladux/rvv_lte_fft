#include "../include/ofdm_fft.h"

/* ---------------------------------------------------------------------------
 * r3_cpx_bfly - DIF radix-3 butterfly
 *
 *   a0r..a2i   input thirds (real, imag) - input
 *   w1r..w2i   twiddle factors (real, imag) W1, W2 - input
 *   y0r..y2i   output thirds (real, imag) - output
 *   vl         vector length - input
 * --------------------------------------------------------------------------*/
void r3_cpx_bfly(
    vfloat32m1_t a0r, vfloat32m1_t a0i,
    vfloat32m1_t a1r, vfloat32m1_t a1i,
    vfloat32m1_t a2r, vfloat32m1_t a2i,
    vfloat32m1_t w1r, vfloat32m1_t w1i,   
    vfloat32m1_t w2r, vfloat32m1_t w2i,   
    vfloat32m1_t* y0r, vfloat32m1_t* y0i, 
    vfloat32m1_t* y1r, vfloat32m1_t* y1i, 
    vfloat32m1_t* y2r, vfloat32m1_t* y2i, 
    size_t vl)
{
    vfloat32m1_t pr = __riscv_vfadd_vv_f32m1(a1r, a2r, vl); // p = a1+a2
    vfloat32m1_t pi = __riscv_vfadd_vv_f32m1(a1i, a2i, vl);
    vfloat32m1_t qr = __riscv_vfsub_vv_f32m1(a1r, a2r, vl); // q = a1-a2
    vfloat32m1_t qi = __riscv_vfsub_vv_f32m1(a1i, a2i, vl);
 
    *y0r = __riscv_vfadd_vv_f32m1(a0r, pr, vl); // y0 = a0 + p
    *y0i = __riscv_vfadd_vv_f32m1(a0i, pi, vl);
 
    vfloat32m1_t half_pr = __riscv_vfmul_vf_f32m1(pr, 0.5f, vl); 
    vfloat32m1_t half_pi = __riscv_vfmul_vf_f32m1(pi, 0.5f, vl); 

    vfloat32m1_t base_r = __riscv_vfsub_vv_f32m1(a0r, half_pr, vl); // base_r = a0r - pr*0.5
    vfloat32m1_t base_i = __riscv_vfsub_vv_f32m1(a0i, half_pi, vl); // base_i = a0i - pi*0.5
 
    vfloat32m1_t sq_qi = __riscv_vfmul_vf_f32m1(qi, SQRT3_OVER_2, vl); // sq_qi = qi*sqrt(3/2)
    vfloat32m1_t sq_qr = __riscv_vfmul_vf_f32m1(qr, SQRT3_OVER_2, vl); // sq_qr = qr*sqrt(3/2)

    vfloat32m1_t t1r = __riscv_vfadd_vv_f32m1(base_r, sq_qi, vl); // t1 = (base_r + sq_qi) + i*(base_i - sq_qr) 
    vfloat32m1_t t1i = __riscv_vfsub_vv_f32m1(base_i, sq_qr, vl); 
 
    vfloat32m1_t t2r = __riscv_vfnmsac_vf_f32m1(t1r, 2.0f, sq_qi, vl); // t2 = (base_r - sq_qi) + i*(base_i + sq_qr)
    vfloat32m1_t t2i = __riscv_vfmacc_vf_f32m1 (t1i, 2.0f, sq_qr, vl);
 
    cpx_cmul(t1r, t1i, w1r, w1i, y1r, y1i, vl); // y1_stored = t1·W1
    cpx_cmul(t2r, t2i, w2r, w2i, y2r, y2i, vl); // y2_stored = t2·W2
}

