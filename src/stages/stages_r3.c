#include "../include/rvv_fft.h"

/* ===========================================================================
 * r3_stage — single radix-3 DIF stage (for N=1536, third=512)
 *
 *   vec      data array — input/output
 *   tw_r3    twiddle table  — input
 *   vlmax    maximum vector length — input
 *
 *   Generic case 
 * ===========================================================================*/
void r3_stage(
    float* restrict vec,
    const float* restrict tw_r3,   
    size_t vlmax)
{

    const size_t third = 512;
 
    const float* restrict pa = vec; // 3 read and write pointers
    const float* restrict pb = vec + 2*third;
    const float* restrict pc = vec + 4*third;

    const float* tw_cursor = tw_r3; // moving pointer to current position in twiddle table

    float* restrict sa = vec;
    float* restrict sb = vec + 2*third;
    float* restrict sc = vec + 4*third;

    size_t j = 0;
 
    /*  pipeline for thirds >= 2 * vlmax. Overlaps loading of next block with computation of current block */
   
    if (third >= 2 * vlmax)
    {
        /* pipeline preload */

        const size_t vl = vlmax;

        vfloat32m1_t a0r, a0i, // butterfly operands
                     a1r, a1i,
                     a2r, a2i;

        vfloat32m1_t w1r, w1i, // twiddle factors
                     w2r, w2i;
 
        cpx2v_load(pa, pb, j, vl, &a0r,&a0i, &a1r,&a1i); // preload
        cpx1v_load (pc, j, vl, &a2r,&a2i);

        r3_cpxt_load_stream(&tw_cursor, vl, &w1r,&w1i, &w2r,&w2i);

        j += vl;
 
        /* Kernel */
        for (; j + vl <= third; j += vl)
        {
            vfloat32m1_t na0r, na0i, // next operand
                         na1r, na1i,
                         na2r, na2i;

            vfloat32m1_t nw1r, nw1i, // next twiddle
                         nw2r, nw2i;
 
            cpx2v_load(pa, pb, j, vl, &na0r,&na0i, &na1r,&na1i); // load of next data block
            cpx1v_load  (pc, j, vl, &na2r, &na2i);

            r3_cpxt_load_stream(&tw_cursor, vl, // load of next twiddles block
                                &nw1r,&nw1i, 
                                &nw2r,&nw2i);

            vfloat32m1_t y0r,y0i,  // output vars
                         y1r,y1i, 
                         y2r,y2i;

            r3_cpx_bfly(            // compute of this block
                a0r,a0i, a1r,a1i, a2r,a2i,
                w1r,w1i, w2r,w2i,
                &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, 
                vl);
 
            cpx2v_store(sa, sb, j-vl, y0r,y0i, y1r,y1i, vl); // store of block result
            cpx1v_store (sc, j-vl, y2r, y2i, vl);
 
            a0r=na0r; a0i=na0i; // move next values
            a1r=na1r; a1i=na1i; 
            a2r=na2r; a2i=na2i;

            w1r=nw1r; w1i=nw1i; 
            w2r=nw2r; w2i=nw2i;
        }
 
        /* pipeline drain for last block*/
        {
            vfloat32m1_t y0r,y0i, 
                         y1r,y1i, 
                         y2r,y2i;

            r3_cpx_bfly(
                a0r,a0i, a1r,a1i, a2r,a2i,
                w1r,w1i, w2r,w2i,
                &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, 
                vl);
                
            cpx2v_store(sa, sb, j-vl, y0r,y0i, y1r,y1i, vl);
            cpx1v_store (sc, j-vl, y2r, y2i, vl);
        }
    }
 
    // section without pipeline for vlmax blocks
    for (; j + vlmax <= third; j += vlmax)
    {
        const size_t vl = vlmax;

        vfloat32m1_t a0r, a0i,
                     a1r, a1i,
                     a2r, a2i;

        cpx2v_load(pa, pb, j, vl, &a0r, &a0i, &a1r, &a1i);
        cpx1v_load(pc, j, vl, &a2r, &a2i);

        vfloat32m1_t w1r, w1i,
                     w2r, w2i;

        r3_cpxt_load_stream(&tw_cursor, vl, &w1r, &w1i, &w2r, &w2i);

        vfloat32m1_t y0r, y0i,
                        y1r, y1i,
                        y2r, y2i;

        r3_cpx_bfly(
            a0r,a0i, a1r,a1i, a2r,a2i,
            w1r,w1i, w2r,w2i,
            &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, 
            vl);

        cpx2v_store(sa, sb, j, y0r,y0i, y1r,y1i, vl);
        cpx1v_store (sc, j, y2r, y2i, vl);
    }
 
    // tail for processing remaining elements
        if (j < third)
        {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(third - j));

            vfloat32m1_t a0r, a0i,
                         a1r, a1i,
                         a2r, a2i;

            cpx2v_load(pa, pb, j, vl, &a0r, &a0i, &a1r, &a1i);
            cpx1v_load(pc, j, vl, &a2r, &a2i);

            vfloat32m1_t w1r, w1i,
                         w2r, w2i;

            r3_cpxt_load_stream(&tw_cursor, vl, &w1r, &w1i, &w2r, &w2i);

            vfloat32m1_t y0r, y0i,
                         y1r, y1i,
                         y2r, y2i;

            r3_cpx_bfly(a0r, a0i, a1r, a1i, a2r, a2i,
                        w1r, w1i, w2r, w2i,
                        &y0r, &y0i, &y1r, &y1i, &y2r, &y2i,
                        vl);

            cpx2v_store(sa, sb, j, y0r, y0i, y1r, y1i, vl);
            cpx1v_store(sc, j, y2r, y2i, vl);
        }
}
