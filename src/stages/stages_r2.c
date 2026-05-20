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
    
    float *pa = vec;
    float *pb = vec + 2;

    for (int32_t i = 0; i < ngroups; )
    {
        size_t vl = __riscv_vsetvl_e32m1(ngroups - i);


        vfloat32m1_t ar = __riscv_vle32_v_f32m1(pa + 0, vl); // segmented load of 4 vectors  
        vfloat32m1_t ai = __riscv_vle32_v_f32m1(pa + 1, vl);
        vfloat32m1_t br = __riscv_vle32_v_f32m1(pb + 0, vl);
        vfloat32m1_t bi = __riscv_vle32_v_f32m1(pb + 1, vl); 
  

        ar = __riscv_vlse32_v_f32m1(pa + 0, 4 * sizeof(float), vl);
        ai = __riscv_vlse32_v_f32m1(pa + 1, 4 * sizeof(float), vl);

        br = __riscv_vlse32_v_f32m1(pb + 0, 4 * sizeof(float), vl);
        bi = __riscv_vlse32_v_f32m1(pb + 1, 4 * sizeof(float), vl);

        vfloat32m1_t sumr = __riscv_vfadd_vv_f32m1(ar, br, vl);
        vfloat32m1_t sumi = __riscv_vfadd_vv_f32m1(ai, bi, vl);

        vfloat32m1_t diffr = __riscv_vfsub_vv_f32m1(ar, br, vl);
        vfloat32m1_t diffi = __riscv_vfsub_vv_f32m1(ai, bi, vl);

        __riscv_vsse32_v_f32m1(pa + 0, 4 * sizeof(float), sumr, vl);  // segmented store of 4 vectors
        __riscv_vsse32_v_f32m1(pa + 1, 4 * sizeof(float), sumi, vl);

        __riscv_vsse32_v_f32m1(pb + 0, 4 * sizeof(float), diffr, vl);
        __riscv_vsse32_v_f32m1(pb + 1, 4 * sizeof(float), diffi, vl);

        pa += 4 * vl;
        pb += 4 * vl;

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


    float *p0 = vec + 0; /* a0 */
    float *p1 = vec + 2; /* a1 */
    float *p2 = vec + 4; /* b0 */
    float *p3 = vec + 6; /* b1 */

    const ptrdiff_t stride = 8 * sizeof(float);

    for (int32_t i = 0; i < ngroups; )
    {
        size_t vl = __riscv_vsetvl_e32m1(ngroups - i);


        vfloat32m1_t a0r =  __riscv_vlse32_v_f32m1(p0 + 0, stride, vl); // segmented load of 8 vectors
        vfloat32m1_t a0i = __riscv_vlse32_v_f32m1(p0 + 1, stride, vl);

        vfloat32m1_t a1r = __riscv_vlse32_v_f32m1(p1 + 0, stride, vl);
        vfloat32m1_t a1i = __riscv_vlse32_v_f32m1(p1 + 1, stride, vl);

        vfloat32m1_t b0r = __riscv_vlse32_v_f32m1(p2 + 0, stride, vl);
        vfloat32m1_t b0i = __riscv_vlse32_v_f32m1(p2 + 1, stride, vl);

        vfloat32m1_t b1r = __riscv_vlse32_v_f32m1(p3 + 0, stride, vl);
        vfloat32m1_t b1i = __riscv_vlse32_v_f32m1(p3 + 1, stride, vl);


        vfloat32m1_t y0r = __riscv_vfadd_vv_f32m1(a0r, b0r, vl);
        vfloat32m1_t y0i = __riscv_vfadd_vv_f32m1(a0i, b0i, vl);

        vfloat32m1_t y2r = __riscv_vfsub_vv_f32m1(a0r, b0r, vl);
        vfloat32m1_t y2i = __riscv_vfsub_vv_f32m1(a0i, b0i, vl);

        vfloat32m1_t y1r = __riscv_vfadd_vv_f32m1(a1r, b1r, vl);
        vfloat32m1_t y1i = __riscv_vfadd_vv_f32m1(a1i, b1i, vl);

        vfloat32m1_t dr =
            __riscv_vfsub_vv_f32m1(a1r, b1r, vl);
        vfloat32m1_t di =
            __riscv_vfsub_vv_f32m1(a1i, b1i, vl);
       
          vfloat32m1_t y3r = di;
        vfloat32m1_t y3i =
            __riscv_vfneg_v_f32m1(dr, vl);

        __riscv_vsse32_v_f32m1(p0 + 0, stride, y0r, vl);
        __riscv_vsse32_v_f32m1(p0 + 1, stride, y0i, vl);

        __riscv_vsse32_v_f32m1(p1 + 0, stride, y1r, vl);
        __riscv_vsse32_v_f32m1(p1 + 1, stride, y1i, vl);

        __riscv_vsse32_v_f32m1(p2 + 0, stride, y2r, vl);
        __riscv_vsse32_v_f32m1(p2 + 1, stride, y2i, vl);

        __riscv_vsse32_v_f32m1(p3 + 0, stride, y3r, vl);
        __riscv_vsse32_v_f32m1(p3 + 1, stride, y3i, vl);

        p0 += 8 * vl;
        p1 += 8 * vl;
        p2 += 8 * vl;
        p3 += 8 * vl;

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

        if (half >= 2 * vlmax)
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
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(half - j));

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
