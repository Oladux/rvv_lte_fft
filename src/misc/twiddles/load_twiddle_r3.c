#include "../include/rvv_fft.h"

/* ---------------------------------------------------------------------------
 * r3_cpxt_load_stream — sequential load of radix-3 twiddle factors from stream
 *
 *   tw_cursor         pointer to current position in twiddle table — input/output
 *   vl                vector length — input
 *   w1r, w1i          first twiddle factor pointers (real, imag) — output
 *   w2r, w2i          second twiddle factor pointers (real, imag) — output
 * --------------------------------------------------------------------------*/
void r3_cpxt_load_stream(
    const float* restrict *tw_cursor,
    size_t vl,
    vfloat32m1_t* w1r, vfloat32m1_t* w1i,
    vfloat32m1_t* w2r, vfloat32m1_t* w2i)
{
    vfloat32m1x4_t tw = __riscv_vlseg4e32_v_f32m1x4(*tw_cursor, vl); // segmented load of twiddles

    *w1r = __riscv_vget_v_f32m1x4_f32m1(tw, 0);
    *w1i = __riscv_vget_v_f32m1x4_f32m1(tw, 1);

    *w2r = __riscv_vget_v_f32m1x4_f32m1(tw, 2);
    *w2i = __riscv_vget_v_f32m1x4_f32m1(tw, 3);

    *tw_cursor += 4 * vl; // move cursor to next item in twiddles stream
}


