#include "../include/ofdm_fft.h"

/* ---------------------------------------------------------------------------
 * r2_stage_h1 - specialized radix-2 stage for grp_size=2, H=1, W≡1
 *
 *   vec      data array - input/output
 *   N        full FFT size - input
 *
 *  In this case twiddle factors are not used
 * --------------------------------------------------------------------------*/
void r2_stage_h1(float* restrict vec, int32_t N)
{
    const size_t ngroups = N >> 1;
    
    float *p = vec;

    const size_t vlmax = __riscv_vsetvlmax_e32m1();  

    for (int32_t i = 0; i < ngroups; )
    {
        size_t vl = (ngroups - i < vlmax) ? (ngroups - i) : vlmax;

        vfloat32m1x4_t seg = __riscv_vlseg4e32_v_f32m1x4(p, vl);
        
        vfloat32m1_t ar = __riscv_vget_v_f32m1x4_f32m1(seg, 0);
        vfloat32m1_t ai = __riscv_vget_v_f32m1x4_f32m1(seg, 1);
        vfloat32m1_t br = __riscv_vget_v_f32m1x4_f32m1(seg, 2);
        vfloat32m1_t bi = __riscv_vget_v_f32m1x4_f32m1(seg, 3);


        vfloat32m1_t sumr = __riscv_vfadd_vv_f32m1(ar, br, vl);
        vfloat32m1_t sumi = __riscv_vfadd_vv_f32m1(ai, bi, vl);

        vfloat32m1_t diffr = __riscv_vfsub_vv_f32m1(ar, br, vl);
        vfloat32m1_t diffi = __riscv_vfsub_vv_f32m1(ai, bi, vl);

       __riscv_vsseg4e32_v_f32m1x4(p,
            __riscv_vcreate_v_f32m1x4(sumr, sumi, diffr, diffi), vl);

        p += 4 * vl;

        i += vl;

    }
}

/* ---------------------------------------------------------------------------
 * r2_stage_h2 - specialized radix-2 stage for grp_size=4, H=2, W[0]=1, W[1]=-i
 *
 *   vec      data array (in-place) - input/output
 *   N        full FFT size - input
 *
 *   In this case twiddle factors are handled without multiplications
 * --------------------------------------------------------------------------*/
void r2_stage_h2(float* restrict vec, int32_t N)
{
    float* p = vec;
    const size_t ngroups = N >> 2;
    const size_t vlmax = __riscv_vsetvlmax_e32m1();  
    
    for (size_t i = 0; i < ngroups;)
    {
        size_t vl = (ngroups - i < vlmax) ? (ngroups - i) : vlmax;
        
        vfloat32m1x8_t seg = __riscv_vlseg8e32_v_f32m1x8(p, vl);
        
        vfloat32m1_t a0r = __riscv_vget_v_f32m1x8_f32m1(seg, 0);
        vfloat32m1_t a0i = __riscv_vget_v_f32m1x8_f32m1(seg, 1);
        vfloat32m1_t a1r = __riscv_vget_v_f32m1x8_f32m1(seg, 2);
        vfloat32m1_t a1i = __riscv_vget_v_f32m1x8_f32m1(seg, 3);
        vfloat32m1_t b0r = __riscv_vget_v_f32m1x8_f32m1(seg, 4);
        vfloat32m1_t b0i = __riscv_vget_v_f32m1x8_f32m1(seg, 5);
        vfloat32m1_t b1r = __riscv_vget_v_f32m1x8_f32m1(seg, 6);
        vfloat32m1_t b1i = __riscv_vget_v_f32m1x8_f32m1(seg, 7);

        vfloat32m1_t y0r = __riscv_vfadd_vv_f32m1(a0r, b0r, vl);
        vfloat32m1_t y0i = __riscv_vfadd_vv_f32m1(a0i, b0i, vl);
        vfloat32m1_t y2r = __riscv_vfsub_vv_f32m1(a0r, b0r, vl);
        vfloat32m1_t y2i = __riscv_vfsub_vv_f32m1(a0i, b0i, vl);
        vfloat32m1_t y1r = __riscv_vfadd_vv_f32m1(a1r, b1r, vl);
        vfloat32m1_t y1i = __riscv_vfadd_vv_f32m1(a1i, b1i, vl);
        
        vfloat32m1_t dr = __riscv_vfsub_vv_f32m1(a1r, b1r, vl);
        vfloat32m1_t di = __riscv_vfsub_vv_f32m1(a1i, b1i, vl);
        vfloat32m1_t y3r = di;
        vfloat32m1_t y3i = __riscv_vfneg_v_f32m1(dr, vl);

        __riscv_vsseg8e32_v_f32m1x8(p,
            __riscv_vcreate_v_f32m1x8(y0r, y0i, y1r, y1i,
                                       y2r, y2i, y3r, y3i), vl);

        p += 8 * vl;
        i += vl;
    }
}

/* ===========================================================================
 * r2_stage - single radix-2 DIF stage with software pipeline
 *
 *   vec       data array  - input/output
 *   N         full FFT size - input
 *   grp_size  group size for this stage - input
 *   tw_stage  twiddle table block for this stage - input
 *   vlmax     maximum vector length - input
 *
 *   Generic case 
 * ===========================================================================*/
void r2_stage(
    float* restrict vec,
    int32_t N, size_t grp_size,
    const float* restrict tw_stage,
    size_t vlmax)
{
    const size_t half = grp_size >> 1;

    for (size_t grp_start = 0; grp_start < N; grp_start += grp_size)
    {
        float* base = &vec[2 * grp_start];

        const float* pa = base; // 2 read and write pointers
        const float* pb = base + 2 * half;

        const float* tw_cursor = tw_stage;  // moving pointer to current position in twiddle table

        float* sa = base;
        float* sb = base + 2 * half;

        size_t j = 0;

        /*  pipeline for halfs >= 2 * vlmax. Overlaps loading of next block with computation of current block */

    #if ENABLE_PIPELINE

        if (half >= 4 * vlmax)
        {
            /* pipeline preload */

            const size_t vl = vlmax;

            vfloat32m1_t a0r, a0i,  // butterfly operands
                         a1r, a1i;

            vfloat32m1_t w1r, w1i; // twiddle factors

            cpx2v_load(pa, pb, j, vl, &a0r, &a0i, &a1r, &a1i); // preload

            r2_cpxt_load_stream(&tw_cursor, vl, &w1r, &w1i);

            j += vl;

            for (; j + vl <= half; j += vl)
            {
                vfloat32m1_t na0r, na0i,  // next operand
                             na1r, na1i;

                vfloat32m1_t nw1r, nw1i; // next twiddle

                cpx2v_load(pa, pb, j, vl, &na0r, &na0i, &na1r, &na1i); // load of next data block

                r2_cpxt_load_stream(&tw_cursor, vl, &nw1r, &nw1i); // load of next twiddles block

                vfloat32m1_t y0r,y0i,  // output vars
                             y1r,y1i;

                r2_cpx_bfly(a0r, a0i, a1r, a1i, 
                            w1r, w1i,
                            &y0r, &y0i, &y1r, &y1i, 
                            vl);

                cpx2v_store(sa, sb, j-vl, y0r, y0i, y1r, y1i, vl); // store of block result

                a0r=na0r; a0i=na0i;  // move next values
                a1r=na1r; a1i=na1i;

                w1r=nw1r; w1i=nw1i;
            }

            /* pipeline drain for last block*/
            vfloat32m1_t y0r,y0i, 
                         y1r,y1i;

            r2_cpx_bfly(a0r, a0i, a1r, a1i, 
                        w1r, w1i,
                        &y0r, &y0i, &y1r, &y1i, 
                        vl);

            cpx2v_store(sa, sb, j-vl, y0r,y0i, y1r, y1i, vl);
        }
#endif
        // section without pipeline for vlmax blocks
        for (; j + vlmax <= half; j += vlmax)
        {
            const size_t vl = vlmax;

            vfloat32m1_t a0r, a0i, 
                         a1r, a1i;

            cpx2v_load(pa, pb, j, vl, &a0r,&a0i, &a1r,&a1i);

            vfloat32m1_t w1r, w1i;

            r2_cpxt_load_stream(&tw_cursor, vl, &w1r, &w1i);


            vfloat32m1_t y0r,y0i, 
                         y1r,y1i;

            r2_cpx_bfly(a0r, a0i, a1r, a1i, 
                        w1r, w1i,
                        &y0r, &y0i, &y1r, &y1i, 
                        vl);

            cpx2v_store(sa, sb, j, y0r,y0i, y1r, y1i, vl);
        }

        // tail for processing remaining elements
        if (j < half)
        {
            const size_t remaining = half - j;
            const size_t vl = (remaining >= 4) ? 4 : remaining;  

            vfloat32m1_t a0r, a0i, 
                         a1r, a1i;

            cpx2v_load(pa, pb, j, vl, &a0r,&a0i, &a1r,&a1i);

            vfloat32m1_t w1r, w1i;

            r2_cpxt_load_stream(&tw_cursor, vl, &w1r, &w1i);

            vfloat32m1_t y0r,y0i, 
                         y1r,y1i;


            r2_cpx_bfly(a0r, a0i, a1r, a1i, 
                        w1r, w1i,
                        &y0r, &y0i, &y1r, &y1i, 
                        vl);

            cpx2v_store(sa, sb, j, y0r,y0i, y1r, y1i, vl);
        }
    }
}
