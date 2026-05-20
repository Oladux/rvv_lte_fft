#include "../include/ofdm_fft.h"

/* ---------------------------------------------------------------------------
 * cpx1v_load - load of one interleaved complex vector
 *
 *   pc    source pointer - input
 *   j     element offset within the group - input
 *   vl    vector length - input
 *   rc, ic vector pointers (real, imag) - output
 * --------------------------------------------------------------------------*/
void cpx1v_load(
    const float* restrict pc,
    size_t          j,
    size_t       vl,
    vfloat32m1_t* rc, vfloat32m1_t* ic)
{
    vfloat32m1x2_t vc = __riscv_vlseg2e32_v_f32m1x2(pc + 2*j, vl); // segmented complex load from source
    *rc = __riscv_vget_v_f32m1x2_f32m1(vc, 0);
    *ic = __riscv_vget_v_f32m1x2_f32m1(vc, 1);
}
