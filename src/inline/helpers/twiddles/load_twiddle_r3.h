#ifndef R3_TWDL_H
#define R3_TWDL_H

/* ---------------------------------------------------------------------------
 * r3_cpxt_load_stream - sequential load of radix-3 twiddle factors from stream
 *
 *   w1_cursor         pointer to current strided position in twiddle table - input/output
 *   w2_cursor         pointer to current strided position in twiddle table - input/output
 *   vl                vector length - input
 *   w1r, w1i          first twiddle factor pointers (real, imag) - output
 *   w2r, w2i          second twiddle factor pointers (real, imag) - output
 * --------------------------------------------------------------------------*/

static inline void r3_cpxt_load_stream(
    const float* restrict *w1_cursor,
    const float* restrict *w2_cursor,
    size_t vl,
    vfloat32m1_t* w1r, vfloat32m1_t* w1i,
    vfloat32m1_t* w2r, vfloat32m1_t* w2i)
{

    vfloat32m1x2_t t1 = __riscv_vlseg2e32_v_f32m1x2(*w1_cursor, vl); // segmented load of twiddles
    vfloat32m1x2_t t2 = __riscv_vlseg2e32_v_f32m1x2(*w2_cursor, vl); 

    *w1r = __riscv_vget_v_f32m1x2_f32m1(t1, 0);
    *w1i = __riscv_vget_v_f32m1x2_f32m1(t1, 1);

    *w2r = __riscv_vget_v_f32m1x2_f32m1(t2, 0);
    *w2i = __riscv_vget_v_f32m1x2_f32m1(t2, 1);

    *w1_cursor += 2 * vl;  // move cursor to next item in twiddles stream
    *w2_cursor += 2 * vl;
}

#endif