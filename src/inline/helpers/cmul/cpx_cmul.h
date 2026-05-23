#ifndef CMUL_H
#define CMUL_H


/* ---------------------------------------------------------------------------
 * cpx_cmul - complex multiplication: out = (ar + i·ai) × (br + i·bi)
 *
 *   ar, ai   first operand (real, imag) - input
 *   br, bi   second operand (real, imag) - input
 *   outr, outi result pointers (real, imag) - output
 *   vl       vector length - input
 * --------------------------------------------------------------------------*/
static inline void cpx_cmul(
    vfloat32m1_t ar, vfloat32m1_t ai,
    vfloat32m1_t br, vfloat32m1_t bi,
    vfloat32m1_t* outr, vfloat32m1_t* outi,
    size_t vl)
{
    *outr = __riscv_vfmul_vv_f32m1(ar, br, vl); // outr = ar*br - ai*bi
    *outr = __riscv_vfnmsac_vv_f32m1(*outr, ai, bi, vl);

    *outi = __riscv_vfmul_vv_f32m1(ai, br, vl);
    *outi = __riscv_vfmacc_vv_f32m1(*outi, ar, bi, vl); // outi = ai*br + ar*bi

}

#endif