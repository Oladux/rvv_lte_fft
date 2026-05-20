#include "../include/ofdm_fft.h"

/* ---------------------------------------------------------------------------
 * r4_cpxt_load_stream - sequential load of radix-4 twiddle factors from stream
 *
 *   tw_cursor         pointer to current position in twiddle table - input/output
 *   vl                vector length - input
 *   w1r, w1i          W1 twiddle factor pointers (real, imag) - output
 *   w2r, w2i          W2 twiddle factor pointers (real, imag) - output
 *   w3r, w3i          W3 twiddle factor pointers (real, imag) - output
 * --------------------------------------------------------------------------*/
void r4_cpxt_load_stream(
    const float* restrict *tw_cursor,
    size_t vl,
    vfloat32m1_t* w1r, vfloat32m1_t* w1i,
    vfloat32m1_t* w2r, vfloat32m1_t* w2i,
    vfloat32m1_t* w3r, vfloat32m1_t* w3i)
{
    vfloat32m1x6_t tw = __riscv_vlseg6e32_v_f32m1x6(*tw_cursor, vl); // segmented load of twiddles

    *w1r = __riscv_vget_v_f32m1x6_f32m1(tw, 0);
    *w1i = __riscv_vget_v_f32m1x6_f32m1(tw, 1);

    *w2r = __riscv_vget_v_f32m1x6_f32m1(tw, 2);
    *w2i = __riscv_vget_v_f32m1x6_f32m1(tw, 3);

    *w3r = __riscv_vget_v_f32m1x6_f32m1(tw, 4);
    *w3i = __riscv_vget_v_f32m1x6_f32m1(tw, 5);

    *tw_cursor += 6 * vl; // move cursor to next item in twiddles stream
}

