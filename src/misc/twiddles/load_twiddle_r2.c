#include "../include/rvv_fft.h"

/* ---------------------------------------------------------------------------
 * r2_cpxt_load_stream — sequential load of radix-2 twiddle factors from stream
 *
 *   tw_cursor   pointer to current position in twiddle table — input/output
 *   vl          vector length — input
 *   rw, iw      twiddle factor pointers (real, imag) — output
 * --------------------------------------------------------------------------*/
void r2_cpxt_load_stream(
    const float* restrict *tw_cursor,
    size_t       vl,
    vfloat32m1_t* rw, vfloat32m1_t* iw)
{
    vfloat32m1x2_t tw = __riscv_vlseg2e32_v_f32m1x2(*tw_cursor, vl); // segmented load of twiddles
    *rw = __riscv_vget_v_f32m1x2_f32m1(tw, 0);
    *iw = __riscv_vget_v_f32m1x2_f32m1(tw, 1);
    *tw_cursor += 2 * vl; // move cursor to next item in twiddles stream
}

