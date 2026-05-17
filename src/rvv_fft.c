#include "../include/rvv_fft.h"
 
/* ===========================================================================
 * ofdm_fft — main FFT entry point
 *
 *   vec   data array  — input/output
 *   N     FFT size — input
 *
 *   Special path for N=1536 (= 3*(2 << 9)):
 *     r3_stage with third=512
 *     Three independent r2 cascades of 9 stages on blocks of 512
 *     bit_reverse for N=1536
 *
 *   Standard path for N = power of two:
 *     r2_stage on top if log2N is odd, then r4_stage cascade
 *     bit_reverse at the end
 * ===========================================================================*/
float* ofdm_fft(float* restrict vec, int32_t N)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
 
    if (N == 1536) // special r3 path
    {
        const int32_t third = 512;
        const float* tw_r3;
        get_twiddle_r3(&tw_r3); // get r3-table from outer file

        r3_stage(vec, tw_r3, vlmax); // single r3 stage
 
        size_t log2T;
        const float* tw2;
        const size_t*   off2;
        get_twiddle_r2(third, &tw2, &off2, &log2T); // get r2-table from outer file
 
        for (size_t blk = 0; blk < 3; blk++)
        {
            float* blk_base = vec + 2 * blk * third;
 
            for (int32_t stage = log2T; stage >= 1; stage--) // specific stages dispatch
            {
                const size_t grp_size = 1 << stage;
                if (grp_size == 2)
                {
                    r2_stage_h1(blk_base, third);
                }
                else if (grp_size == 4)
                {
                    r2_stage_h2(blk_base, third);
                }
                else 
                {
                    const float* tw_stage = tw2 + 2 * off2[stage - 1];
                    r2_stage(blk_base, third, grp_size, tw_stage, vlmax);
                }
            }
        }
 
        bit_reverse(vec, N); // bit-reverse of computed data
        return vec;
    }

    // common path

    size_t log2N;
    const float* tw2;
    const size_t*   off2;
    get_twiddle_r2(N, &tw2, &off2, &log2N); // get r2-table from outer file
 
    const float* tw4;
    const size_t*   off4;
    size_t          stages_r4;
    get_twiddle_r4(N, &tw4, &off4, &stages_r4); // get r4-table from outer file
 
    if (log2N & 1) // log2n is odd 
    {
        const float* tw_stage = tw2 + 2 * off2[log2N - 1];
        r2_stage(vec, N, N, tw_stage, vlmax);
    }
 
    for (int32_t r4_idx = stages_r4 - 1; r4_idx >= 0; r4_idx--) // specific stages dispatch
    {
        const size_t    grp_size = 1 << (2 + 2*r4_idx);
        const float* tw_base  = tw4 + 2 * off4[r4_idx];
 
        if (grp_size == 4)
        {
            r4_stage_q1(vec, N);
        }
        else if (grp_size == 16)
        {
            r4_stage_q4(vec, N, tw_base);
        }
        else
        {
            r4_stage(vec, N, grp_size, tw_base, vlmax);
        }
    }
 
    bit_reverse(vec, N); // bit-reverse of computed data
    return vec;
}