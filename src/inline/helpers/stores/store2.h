#ifndef STORE2_H
#define STORE2_H

/* ---------------------------------------------------------------------------
 * cpx2v_store - store of two interleaved complex vectors
 *
 *   pa, pb   destination pointers - input
 *   j        element offset within the group - input
 *   x1r, x1i first vector values (real, imag) - input
 *   x2r, x2i second vector values (real, imag) - input
 *   vl       vector length - input
 * --------------------------------------------------------------------------*/
static inline void cpx2v_store(
    float* restrict pa,
    float* restrict pb,
    size_t    j,
    vfloat32m1_t x1r, vfloat32m1_t x1i,
    vfloat32m1_t x2r, vfloat32m1_t x2i,
    size_t vl)
{
    __riscv_vsseg2e32_v_f32m1x2(  // segmented store of input vector 
        pa + 2*j,
        __riscv_vcreate_v_f32m1x2(x1r, x1i), vl);

    __riscv_vsseg2e32_v_f32m1x2(
        pb + 2*j,
        __riscv_vcreate_v_f32m1x2(x2r, x2i), vl);
}

#endif