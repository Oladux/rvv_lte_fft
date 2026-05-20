#include "../include/ofdm_fft.h"

/* ---------------------------------------------------------------------------
 * r4_cpx_bfly - DIF radix-4 butterfly 
 *   a0r..a3i   input quarters (real, imag) - input
 *   w1r..w3i   twiddle factors (real, imag) W1, W2, W3 - input
 *   y0r..y3i   output quarters (real, imag) - output
 *   vl         vector length - input
 * --------------------------------------------------------------------------*/
void r4_cpx_bfly(
    vfloat32m1_t a0r, vfloat32m1_t a0i,
    vfloat32m1_t a1r, vfloat32m1_t a1i,
    vfloat32m1_t a2r, vfloat32m1_t a2i,
    vfloat32m1_t a3r, vfloat32m1_t a3i,
    vfloat32m1_t w1r, vfloat32m1_t w1i,
    vfloat32m1_t w2r, vfloat32m1_t w2i,
    vfloat32m1_t w3r, vfloat32m1_t w3i,
    vfloat32m1_t* y0r, vfloat32m1_t* y0i, 
    vfloat32m1_t* y1r, vfloat32m1_t* y1i, 
    vfloat32m1_t* y2r, vfloat32m1_t* y2i,  
    vfloat32m1_t* y3r, vfloat32m1_t* y3i,  
    size_t vl)
{
    vfloat32m1_t t0r = __riscv_vfadd_vv_f32m1(a0r, a2r, vl); // t0 = a0+a2,
    vfloat32m1_t t0i = __riscv_vfadd_vv_f32m1(a0i, a2i, vl);  

    vfloat32m1_t t1r = __riscv_vfsub_vv_f32m1(a0r, a2r, vl); // t1 = a0−a2
    vfloat32m1_t t1i = __riscv_vfsub_vv_f32m1(a0i, a2i, vl);

    vfloat32m1_t t2r = __riscv_vfadd_vv_f32m1(a1r, a3r, vl); // t2 = a1+a3 
    vfloat32m1_t t2i = __riscv_vfadd_vv_f32m1(a1i, a3i, vl);

    vfloat32m1_t dr  = __riscv_vfsub_vv_f32m1(a1r, a3r, vl); // di − i·dr = (dr+i·di)·(−i) 
    vfloat32m1_t di  = __riscv_vfsub_vv_f32m1(a1i, a3i, vl);
    vfloat32m1_t t3r = di;                                   // t3 = (a1−a3)·(−i)
    vfloat32m1_t t3i = __riscv_vfneg_v_f32m1(dr, vl);

    *y0r = __riscv_vfadd_vv_f32m1(t0r, t2r, vl); // y0 = t0+t2 
    *y0i = __riscv_vfadd_vv_f32m1(t0i, t2i, vl);

    // swap vars
    vfloat32m1_t s0r = __riscv_vfsub_vv_f32m1(t0r, t2r, vl); // s0=t0−t2
    vfloat32m1_t s0i = __riscv_vfsub_vv_f32m1(t0i, t2i, vl);
    vfloat32m1_t s1r = __riscv_vfadd_vv_f32m1(t1r, t3r, vl); // s1=t1+t3
    vfloat32m1_t s1i = __riscv_vfadd_vv_f32m1(t1i, t3i, vl);
    vfloat32m1_t s2r = __riscv_vfsub_vv_f32m1(t1r, t3r, vl); // s2=t1−t3
    vfloat32m1_t s2i = __riscv_vfsub_vv_f32m1(t1i, t3i, vl);

    cpx_cmul(s0r, s0i, w2r, w2i, y1r, y1i, vl); // y1_stored = s0·W2 

    cpx_cmul(s1r, s1i, w1r, w1i, y2r, y2i, vl); // y2_stored = s1·W1 

    cpx_cmul(s2r, s2i, w3r, w3i, y3r, y3i, vl); // y3_stored = s2·W3
}

