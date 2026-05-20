#include "../include/ofdm_fft.h"

/* ---------------------------------------------------------------------------
 * r2_cpx_bfly - DIF radix-2 butterfly
 *
 *   ra, ia   first operand (real, imag) - input
 *   rb, ib   second operand (real, imag) - input
 *   rw, iw   twiddle factor W (real, imag) - input
 *   x1r, x1i sum result pointers (real, imag) - output
 *   x2r, x2i difference·W result pointers (real, imag) - output
 *   vl       vector length - input
 * --------------------------------------------------------------------------*/
void r2_cpx_bfly(
    vfloat32m1_t ra, vfloat32m1_t ia,
    vfloat32m1_t rb, vfloat32m1_t ib,
    vfloat32m1_t rw, vfloat32m1_t iw,
    vfloat32m1_t* x1r, vfloat32m1_t* x1i,
    vfloat32m1_t* x2r, vfloat32m1_t* x2i,
    size_t vl)
{
    *x1r = __riscv_vfadd_vv_f32m1(ra, rb, vl); // x1r = a + b
    *x1i = __riscv_vfadd_vv_f32m1(ia, ib, vl); // x1i = i*a + i*b

    vfloat32m1_t dr = __riscv_vfsub_vv_f32m1(ra, rb, vl); // dr = a - b
    vfloat32m1_t di = __riscv_vfsub_vv_f32m1(ia, ib, vl); // di = i*a - i*b

    cpx_cmul(dr, di, rw, iw, x2r, x2i, vl);
}

