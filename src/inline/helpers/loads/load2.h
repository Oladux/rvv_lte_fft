#ifndef LOAD2_H
#define LOAD2_H

/* ---------------------------------------------------------------------------
 * cpx2v_load - load of two interleaved complex vectors
 *
 *   pa, pb   source pointers - input
 *   j        element offset within the group - input
 *   vl       vector length - input
 *   ra, ia   first vector pointers (real, imag) - output
 *   rb, ib   second vector pointers (real, imag) - output
 * --------------------------------------------------------------------------*/
static inline void cpx2v_load(
    const float* restrict pa,
    const float* restrict pb,
    size_t          j,
    size_t          vl,
    vfloat32m1_t* ra, vfloat32m1_t* ia,
    vfloat32m1_t* rb, vfloat32m1_t* ib)
{
    vfloat32m1x2_t va = __riscv_vlseg2e32_v_f32m1x2(pa + 2*j, vl); // segmented load from source
    vfloat32m1x2_t vb = __riscv_vlseg2e32_v_f32m1x2(pb + 2*j, vl);

    *ra = __riscv_vget_v_f32m1x2_f32m1(va, 0); 
    *ia = __riscv_vget_v_f32m1x2_f32m1(va, 1);
    *rb = __riscv_vget_v_f32m1x2_f32m1(vb, 0);
    *ib = __riscv_vget_v_f32m1x2_f32m1(vb, 1);
}

#endif