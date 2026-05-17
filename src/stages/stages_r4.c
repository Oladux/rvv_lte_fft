#include "../include/rvv_fft.h"

/* ---------------------------------------------------------------------------
 * r4_stage_q1 —  specialized last radix-4 stage: grp_size=4, Q=1, W≡1
 *   vec       data array  — input/output
 *   N         FFT size — input
 *   In this case twiddle factors are not used
 * --------------------------------------------------------------------------*/
void r4_stage_q1(float* restrict vec, int32_t N)
{
    float* p = vec;
    const size_t ngroups = N >> 2;

    for (int32_t i = 0; i < ngroups; ) 
    {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(ngroups - i));

        vfloat32m1x8_t seg = __riscv_vlseg8e32_v_f32m1x8(p, vl); // segmented load of 8 vectors 
        vfloat32m1_t a0r = __riscv_vget_v_f32m1x8_f32m1(seg, 0);
        vfloat32m1_t a0i = __riscv_vget_v_f32m1x8_f32m1(seg, 1);
        vfloat32m1_t a1r = __riscv_vget_v_f32m1x8_f32m1(seg, 2);
        vfloat32m1_t a1i = __riscv_vget_v_f32m1x8_f32m1(seg, 3);
        vfloat32m1_t a2r = __riscv_vget_v_f32m1x8_f32m1(seg, 4);
        vfloat32m1_t a2i = __riscv_vget_v_f32m1x8_f32m1(seg, 5);
        vfloat32m1_t a3r = __riscv_vget_v_f32m1x8_f32m1(seg, 6);
        vfloat32m1_t a3i = __riscv_vget_v_f32m1x8_f32m1(seg, 7);

        vfloat32m1_t t0r = __riscv_vfadd_vv_f32m1(a0r, a2r, vl); // t0 = a0 + a2,  
        vfloat32m1_t t0i = __riscv_vfadd_vv_f32m1(a0i, a2i, vl); 
        vfloat32m1_t t1r = __riscv_vfsub_vv_f32m1(a0r, a2r, vl); // t1 = a0 - a2
        vfloat32m1_t t1i = __riscv_vfsub_vv_f32m1(a0i, a2i, vl);
        vfloat32m1_t t2r = __riscv_vfadd_vv_f32m1(a1r, a3r, vl); // t2 = a1 + a3
        vfloat32m1_t t2i = __riscv_vfadd_vv_f32m1(a1i, a3i, vl);
        vfloat32m1_t dr  = __riscv_vfsub_vv_f32m1(a1r, a3r, vl); // d = a1 - a3
        vfloat32m1_t di  = __riscv_vfsub_vv_f32m1(a1i, a3i, vl);

        vfloat32m1_t y0r = __riscv_vfadd_vv_f32m1(t0r, t2r, vl); // y0 = t0 + t2
        vfloat32m1_t y0i = __riscv_vfadd_vv_f32m1(t0i, t2i, vl);

        vfloat32m1_t s0r = __riscv_vfsub_vv_f32m1(t0r, t2r, vl); // s0 = t0 - t2
        vfloat32m1_t s0i = __riscv_vfsub_vv_f32m1(t0i, t2i, vl);
        vfloat32m1_t s1r = __riscv_vfadd_vv_f32m1(t1r, di, vl);  // s1 = t1 + t3
        vfloat32m1_t s1i = __riscv_vfsub_vv_f32m1(t1i, dr, vl);
        vfloat32m1_t s2r = __riscv_vfsub_vv_f32m1(t1r, di, vl);  // s2 = t1 - t3
        vfloat32m1_t s2i = __riscv_vfadd_vv_f32m1(t1i, dr, vl);

        __riscv_vsseg8e32_v_f32m1x8(p,
            __riscv_vcreate_v_f32m1x8(y0r,y0i, s0r,s0i, s1r,s1i, s2r,s2i), vl);  // segmented store of 8 vectors 

        p += 8 * vl;
        i += vl;
    }
}

/* ---------------------------------------------------------------------------
 * r4_stage_q4 — specialized radix-4 stage for grp_size=16, Q=4, preloaded twiddle
 *
 *   vec      data array (in-place) — input/output
 *   N        full FFT size — input
 *   tw_base  twiddle table block for this stage — input
 *
 *    In this case twiddle factors loaded once, then reused for all groups
 * --------------------------------------------------------------------------*/
void r4_stage_q4(
    float* restrict vec, int32_t N,
    const float* restrict tw_base)
{
    const size_t vl = __riscv_vsetvl_e32m1(4); 
    const size_t quarter = 4; 

    vfloat32m1_t w1r, w1i, w2r, w2i, w3r, w3i; // preload all twiddle factors for this stage
    {
        vfloat32m1x6_t tw = __riscv_vlseg6e32_v_f32m1x6(tw_base, vl); 
        w1r = __riscv_vget_v_f32m1x6_f32m1(tw, 0);
        w1i = __riscv_vget_v_f32m1x6_f32m1(tw, 1);
        w2r = __riscv_vget_v_f32m1x6_f32m1(tw, 2);
        w2i = __riscv_vget_v_f32m1x6_f32m1(tw, 3);
        w3r = __riscv_vget_v_f32m1x6_f32m1(tw, 4);
        w3i = __riscv_vget_v_f32m1x6_f32m1(tw, 5);
    }

    for (size_t g = 0; g < N; g += 16)
    {
        float* base = &vec[2 * g];

        vfloat32m1_t a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i; 
        {
            vfloat32m1x2_t ta = __riscv_vlseg2e32_v_f32m1x2(base,       vl); // segmented load of 4 vectors 
            vfloat32m1x2_t tb = __riscv_vlseg2e32_v_f32m1x2(base + 2*quarter, vl);
            vfloat32m1x2_t tc = __riscv_vlseg2e32_v_f32m1x2(base + 4*quarter, vl);
            vfloat32m1x2_t td = __riscv_vlseg2e32_v_f32m1x2(base + 6*quarter, vl);
            a0r = __riscv_vget_v_f32m1x2_f32m1(ta, 0);
            a0i = __riscv_vget_v_f32m1x2_f32m1(ta, 1);
            a1r = __riscv_vget_v_f32m1x2_f32m1(tb, 0);
            a1i = __riscv_vget_v_f32m1x2_f32m1(tb, 1);
            a2r = __riscv_vget_v_f32m1x2_f32m1(tc, 0);
            a2i = __riscv_vget_v_f32m1x2_f32m1(tc, 1);
            a3r = __riscv_vget_v_f32m1x2_f32m1(td, 0);
            a3i = __riscv_vget_v_f32m1x2_f32m1(td, 1);
        }

        vfloat32m1_t y0r,y0i, y1r,y1i, y2r,y2i, y3r,y3i; // radix-4 butterfly with preloaded twiddles
        r4_cpx_bfly(
            a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i,
            w1r,w1i, w2r,w2i, w3r,w3i,
            &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, &y3r,&y3i, vl);

        __riscv_vsseg2e32_v_f32m1x2(  // segmented store of 4 vectors 
            base,
            __riscv_vcreate_v_f32m1x2(y0r,y0i), 
            vl);
        __riscv_vsseg2e32_v_f32m1x2(
            base + 2*quarter,
            __riscv_vcreate_v_f32m1x2(y1r,y1i), 
            vl);
        __riscv_vsseg2e32_v_f32m1x2(
            base + 4*quarter,
            __riscv_vcreate_v_f32m1x2(y2r,y2i), 
            vl);
        __riscv_vsseg2e32_v_f32m1x2(
            base + 6*quarter,
            __riscv_vcreate_v_f32m1x2(y3r,y3i), 
            vl);
    }
}

/* ===========================================================================
 * r4_stage — single radix-4 DIF stage with software pipeline
 *
 *   vec       data array  — input/output
 *   N         full FFT size — input
 *   grp_size  group size for this stage  — input
 *   tw_base   twiddle table block for this stage — input
 *   vlmax     maximum vector length — input
 *
 *   Generic case 
 * ===========================================================================*/
void r4_stage(
    float* restrict vec,
    int32_t N, size_t grp_size,
    const float* restrict tw_base,
    size_t vlmax)
{
    const size_t quarter = grp_size >> 2;

    for (size_t grp_start = 0; grp_start < N; grp_start += grp_size)
    {
        float* base = &vec[2 * grp_start];

        const float* pa = base; // 4 read and write pointers
        const float* pb = base + 2 * quarter;
        const float* pc = base + 4 * quarter;
        const float* pd = base + 6 * quarter;

        const float* tw_cursor = tw_base; // moving pointer to current position in twiddle table

        float* sa = base;
        float* sb = base + 2 * quarter;
        float* sc = base + 4 * quarter;
        float* sd = base + 6 * quarter;

        size_t j = 0;

        /*  pipeline for quarters >= 2 * vlmax. Overlaps loading of next block with computation of current block */

        if (quarter >= 2 * vlmax)
        {
            /* pipeline preload */

            const size_t vl = vlmax;

            vfloat32m1_t a0r,a0i, // butterfly operands
                         a1r,a1i, 
                         a2r,a2i, 
                         a3r,a3i;

            vfloat32m1_t w1r,w1i, // twiddle factors
                         w2r,w2i, 
                         w3r,w3i;

            cpx2v_load(pa, pb, j, vl, &a0r, &a0i, &a1r, &a1i); // preload
            cpx2v_load(pc, pd, j, vl, &a2r, &a2i, &a3r, &a3i);

            r4_cpxt_load_stream(&tw_cursor, vl,
                                &w1r, &w1i, 
                                &w2r, &w2i, 
                                &w3r, &w3i);

            j += vl;

            /* pipeline kernel */
            for (; j + vl <= quarter; j += vl)
            {
                vfloat32m1_t na0r,na0i, // next operand
                             na1r,na1i, 
                             na2r,na2i, 
                             na3r,na3i;

                vfloat32m1_t nw1r,nw1i, // next twiddle
                             nw2r,nw2i, 
                             nw3r,nw3i;

                cpx2v_load(pa, pb, j, vl, &na0r,&na0i, &na1r,&na1i); // load of next data block
                cpx2v_load(pc, pd, j, vl, &na2r,&na2i, &na3r,&na3i);

                r4_cpxt_load_stream(&tw_cursor, vl,                 // load of next twiddles block
                                 &nw1r,&nw1i, &nw2r,&nw2i, &nw3r,&nw3i);

                vfloat32m1_t y0r,y0i,  // output vars
                             y1r,y1i, 
                             y2r,y2i, 
                             y3r,y3i;

                r4_cpx_bfly(       // compute of this block
                    a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i,
                    w1r,w1i, w2r,w2i, w3r,w3i,
                    &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, &y3r,&y3i, 
                    vl);

                cpx2v_store(sa, sb, j-vl, y0r,y0i, y1r,y1i, vl); // store of block result
                cpx2v_store(sc, sd, j-vl, y2r,y2i, y3r,y3i, vl);

                a0r=na0r; a0i=na0i;  // move next values
                a1r=na1r; a1i=na1i;
                a2r=na2r; a2i=na2i; 
                a3r=na3r; a3i=na3i;

                w1r=nw1r; w1i=nw1i; 
                w2r=nw2r; w2i=nw2i;
                w3r=nw3r; w3i=nw3i;
            }

            /* pipeline drain for last block*/
            {
                vfloat32m1_t y0r,y0i, 
                             y1r,y1i, 
                             y2r,y2i, 
                             y3r,y3i;

                r4_cpx_bfly(
                    a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i,
                    w1r,w1i, w2r,w2i, w3r,w3i,
                    &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, &y3r,&y3i, 
                    vl);

                cpx2v_store(sa, sb, j-vl, y0r,y0i, y1r,y1i, vl);
                cpx2v_store(sc, sd, j-vl, y2r,y2i, y3r,y3i, vl);
            }
        }

        // section without pipeline for vlmax blocks
        for (; j + vlmax <= quarter; j += vlmax)
        {
            const size_t vl = vlmax;

            vfloat32m1_t a0r,a0i, 
                         a1r,a1i, 
                         a2r,a2i, 
                         a3r,a3i;

            cpx2v_load(pa, pb, j, vl, &a0r, &a0i, &a1r, &a1i);
            cpx2v_load(pc, pd, j, vl, &a2r, &a2i, &a3r, &a3i);

            vfloat32m1_t w1r,w1i, 
                         w2r,w2i,
                         w3r,w3i;

            r4_cpxt_load_stream(&tw_cursor, vl,
                             &w1r,&w1i, 
                             &w2r,&w2i, 
                             &w3r,&w3i);

            vfloat32m1_t y0r,y0i, 
                         y1r,y1i, 
                         y2r,y2i, 
                         y3r,y3i;

            r4_cpx_bfly(
                a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i,
                w1r, w1i, w2r, w2i, w3r, w3i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i, 
                vl);

            cpx2v_store(sa, sb, j, y0r, y0i, y1r, y1i, vl);
            cpx2v_store(sc, sd, j, y2r, y2i, y3r, y3i, vl);
        }
        
        // tail for processing remaining elements
        if (j < quarter)
        {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(quarter - j));

            vfloat32m1_t a0r,a0i, 
                         a1r, a1i, 
                         a2r, a2i, 
                         a3r, a3i;

            cpx2v_load(pa, pb, j, vl, &a0r, &a0i, &a1r, &a1i);
            cpx2v_load(pc, pd, j, vl, &a2r, &a2i, &a3r, &a3i);

            vfloat32m1_t w1r,w1i, 
                         w2r,w2i, 
                         w3r,w3i;

            r4_cpxt_load_stream(&tw_cursor, vl,
                             &w1r, &w1i, 
                             &w2r, &w2i, 
                             &w3r, &w3i);

            vfloat32m1_t y0r,y0i, 
                          y1r,y1i, 
                          y2r,y2i, 
                          y3r,y3i;

            r4_cpx_bfly(
                a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i,
                w1r,w1i, w2r,w2i, w3r,w3i,
                &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, &y3r,&y3i, 
                vl);

            cpx2v_store(sa, sb, j, y0r, y0i, y1r, y1i, vl);
            cpx2v_store(sc, sd, j, y2r, y2i, y3r, y3i, vl);
        }
    }
}

