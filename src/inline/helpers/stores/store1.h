#ifndef STORE1_H
#define STORE1_H

/* ---------------------------------------------------------------------------
 * cpx1v_store - store of one interleaved complex vector
 *
 *   pc     destination pointer - input
 *   j      element offset within the group - input
 *   xr, xi vector values (real, imag) - input
 *   vl     vector length - input
 * --------------------------------------------------------------------------*/
static inline void cpx1v_store(
    float* restrict pc,
    size_t    j,
    vfloat32m1_t xr, vfloat32m1_t xi,
    size_t vl)
{
    __riscv_vsseg2e32_v_f32m1x2( // segmented store of complex vector 
        pc + 2*j,
        __riscv_vcreate_v_f32m1x2(xr, xi), vl);
}

#endif