#include "../include/rvv_fft.h"
#include "../include/rvv_fft_misc.h"
 
/* ---------------------------------------------------------------------------
 * cpx2v_load — load of two interleaved complex vectors
 *
 *   pa, pb   source pointers — input
 *   j        element offset within the group — input
 *   vl       vector length — input
 *   ra, ia   first vector pointers (real, imag) — output
 *   rb, ib   second vector pointers (real, imag) — output
 * --------------------------------------------------------------------------*/
static inline void cpx2v_load(
    const float* restrict pa,
    const float* restrict pb,
    int          j,
    size_t       vl,
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

/* ---------------------------------------------------------------------------
 * cpx2v_store — store of two interleaved complex vectors
 *
 *   pa, pb   destination pointers — input
 *   j        element offset within the group — input
 *   x1r, x1i first vector values (real, imag) — input
 *   x2r, x2i second vector values (real, imag) — input
 *   vl       vector length — input
 * --------------------------------------------------------------------------*/
static inline void cpx2v_store(
    float* restrict pa,
    float* restrict pb,
    int    j,
    vfloat32m1_t x1r, vfloat32m1_t x1i,
    vfloat32m1_t x2r, vfloat32m1_t x2i,
    size_t vl)
{
    __riscv_vsseg2e32_v_f32m1x2(  // segmented store of input vector 
        pa + 2*j,
        __riscv_vcreate_v_f32m1x2(x1r, x1i), vl);

    __riscv_vsseg2e32_v_f32m1x2(
        pb + 2*j,
        __riscv_vcreate_v_f32m1x2(x2r, x2i), vl);
}

/* ---------------------------------------------------------------------------
 * cpx1v_load — load of one interleaved complex vector
 *
 *   pc    source pointer — input
 *   j     element offset within the group — input
 *   vl    vector length — input
 *   rc, ic vector pointers (real, imag) — output
 * --------------------------------------------------------------------------*/
static inline void cpx1v_load(
    const float* restrict pc,
    int          j,
    size_t       vl,
    vfloat32m1_t* rc, vfloat32m1_t* ic)
{
    vfloat32m1x2_t vc = __riscv_vlseg2e32_v_f32m1x2(pc + 2*j, vl); // segmented complex load from source
    *rc = __riscv_vget_v_f32m1x2_f32m1(vc, 0);
    *ic = __riscv_vget_v_f32m1x2_f32m1(vc, 1);
}

/* ---------------------------------------------------------------------------
 * cpx1v_store — store of one interleaved complex vector
 *
 *   pc     destination pointer — input
 *   j      element offset within the group — input
 *   xr, xi vector values (real, imag) — input
 *   vl     vector length — input
 * --------------------------------------------------------------------------*/
static inline void cpx1v_store(
    float* restrict pc,
    int    j,
    vfloat32m1_t xr, vfloat32m1_t xi,
    size_t vl)
{
    __riscv_vsseg2e32_v_f32m1x2( // segmented store of complex vector 
        pc + 2*j,
        __riscv_vcreate_v_f32m1x2(xr, xi), vl);
}

/* ---------------------------------------------------------------------------
 * r2_cpxt_load_stream — sequential load of radix-2 twiddle factors from stream
 *
 *   tw_cursor   pointer to current position in twiddle table — input/output
 *   vl          vector length — input
 *   rw, iw      twiddle factor pointers (real, imag) — output
 * --------------------------------------------------------------------------*/
static inline void r2_cpxt_load_stream(
    const float* restrict *tw_cursor,
    size_t       vl,
    vfloat32m1_t* rw, vfloat32m1_t* iw)
{
    vfloat32m1x2_t tw = __riscv_vlseg2e32_v_f32m1x2(*tw_cursor, vl); // segmented load of twiddles
    *rw = __riscv_vget_v_f32m1x2_f32m1(tw, 0);
    *iw = __riscv_vget_v_f32m1x2_f32m1(tw, 1);
    *tw_cursor += 2 * vl; // move cursor to next item in twiddles stream
}

/* ---------------------------------------------------------------------------
 * r3_cpxt_load_stream — sequential load of radix-3 twiddle factors from stream
 *
 *   tw_cursor         pointer to current position in twiddle table — input/output
 *   vl                vector length — input
 *   w1r, w1i          first twiddle factor pointers (real, imag) — output
 *   w2r, w2i          second twiddle factor pointers (real, imag) — output
 * --------------------------------------------------------------------------*/
static inline void r3_cpxt_load_stream(
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

/* ---------------------------------------------------------------------------
 * r4_cpxt_load_stream — sequential load of radix-4 twiddle factors from stream
 *
 *   tw_cursor         pointer to current position in twiddle table — input/output
 *   vl                vector length — input
 *   w1r, w1i          W1 twiddle factor pointers (real, imag) — output
 *   w2r, w2i          W2 twiddle factor pointers (real, imag) — output
 *   w3r, w3i          W3 twiddle factor pointers (real, imag) — output
 * --------------------------------------------------------------------------*/
static inline void r4_cpxt_load_stream(
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

/* ---------------------------------------------------------------------------
 * cpx_cmul — complex multiplication: out = (ar + i·ai) × (br + i·bi)
 *
 *   ar, ai   first operand (real, imag) — input
 *   br, bi   second operand (real, imag) — input
 *   outr, outi result pointers (real, imag) — output
 *   vl       vector length — input
 * --------------------------------------------------------------------------*/
static inline void cpx_cmul(
    vfloat32m1_t ar, vfloat32m1_t ai,
    vfloat32m1_t br, vfloat32m1_t bi,
    vfloat32m1_t* outr, vfloat32m1_t* outi,
    size_t vl)
{
    vfloat32m1_t arbr = __riscv_vfmul_vv_f32m1(ar, br, vl); // ar * br
    vfloat32m1_t aibi = __riscv_vfmul_vv_f32m1(ai, bi, vl); // i*ai * i*bi
    vfloat32m1_t aibr = __riscv_vfmul_vv_f32m1(ai, br, vl); // i*ai * br
    vfloat32m1_t arbi = __riscv_vfmul_vv_f32m1(ar, bi, vl); // ar * i*bi

    *outr = __riscv_vfsub_vv_f32m1(arbr, aibi, vl); 
    *outi = __riscv_vfadd_vv_f32m1(aibr, arbi, vl);
}

/* ---------------------------------------------------------------------------
 * r4_cpx_bfly — DIF radix-4 butterfly 
 *   a0r..a3i   input quarters (real, imag) — input
 *   w1r..w3i   twiddle factors (real, imag) W1, W2, W3 — input
 *   y0r..y3i   output quarters (real, imag) — output
 *   vl         vector length — input
 * --------------------------------------------------------------------------*/
static inline void r4_cpx_bfly(
    vfloat32m1_t a0r, vfloat32m1_t a0i,
    vfloat32m1_t a1r, vfloat32m1_t a1i,
    vfloat32m1_t a2r, vfloat32m1_t a2i,
    vfloat32m1_t a3r, vfloat32m1_t a3i,
    vfloat32m1_t w1r, vfloat32m1_t w1i,
    vfloat32m1_t w2r, vfloat32m1_t w2i,
    vfloat32m1_t w3r, vfloat32m1_t w3i,
    vfloat32m1_t* y0r, vfloat32m1_t* y0i, 
    vfloat32m1_t* y1r, vfloat32m1_t* y1i, 
    vfloat32m1_t* y2r, vfloat32m1_t* y2i,  
    vfloat32m1_t* y3r, vfloat32m1_t* y3i,  
    size_t vl)
{
    vfloat32m1_t t0r = __riscv_vfadd_vv_f32m1(a0r, a2r, vl); // t0 = a0+a2,
    vfloat32m1_t t0i = __riscv_vfadd_vv_f32m1(a0i, a2i, vl);  

    vfloat32m1_t t1r = __riscv_vfsub_vv_f32m1(a0r, a2r, vl); // t1 = a0−a2
    vfloat32m1_t t1i = __riscv_vfsub_vv_f32m1(a0i, a2i, vl);

    vfloat32m1_t t2r = __riscv_vfadd_vv_f32m1(a1r, a3r, vl); // t2 = a1+a3 
    vfloat32m1_t t2i = __riscv_vfadd_vv_f32m1(a1i, a3i, vl);

    vfloat32m1_t dr  = __riscv_vfsub_vv_f32m1(a1r, a3r, vl); // di − i·dr = (dr+i·di)·(−i) 
    vfloat32m1_t di  = __riscv_vfsub_vv_f32m1(a1i, a3i, vl);
    vfloat32m1_t t3r = di;                                   // t3 = (a1−a3)·(−i)
    vfloat32m1_t t3i = __riscv_vfneg_v_f32m1(dr, vl);

    *y0r = __riscv_vfadd_vv_f32m1(t0r, t2r, vl); // y0 = t0+t2 
    *y0i = __riscv_vfadd_vv_f32m1(t0i, t2i, vl);

    // swap vars
    vfloat32m1_t s0r = __riscv_vfsub_vv_f32m1(t0r, t2r, vl); // s0=t0−t2
    vfloat32m1_t s0i = __riscv_vfsub_vv_f32m1(t0i, t2i, vl);
    vfloat32m1_t s1r = __riscv_vfadd_vv_f32m1(t1r, t3r, vl); // s1=t1+t3
    vfloat32m1_t s1i = __riscv_vfadd_vv_f32m1(t1i, t3i, vl);
    vfloat32m1_t s2r = __riscv_vfsub_vv_f32m1(t1r, t3r, vl); // s2=t1−t3
    vfloat32m1_t s2i = __riscv_vfsub_vv_f32m1(t1i, t3i, vl);

    cpx_cmul(s0r, s0i, w2r, w2i, y1r, y1i, vl); // y1_stored = s0·W2 

    cpx_cmul(s1r, s1i, w1r, w1i, y2r, y2i, vl); // y2_stored = s1·W1 

    cpx_cmul(s2r, s2i, w3r, w3i, y3r, y3i, vl); // y3_stored = s2·W3
}


/* ---------------------------------------------------------------------------
 * r3_cpx_bfly — DIF radix-3 butterfly
 *
 *   a0r..a2i   input thirds (real, imag) — input
 *   w1r..w2i   twiddle factors (real, imag) W1, W2 — input
 *   y0r..y2i   output thirds (real, imag) — output
 *   vl         vector length — input
 * --------------------------------------------------------------------------*/
static inline void r3_cpx_bfly(
    vfloat32m1_t a0r, vfloat32m1_t a0i,
    vfloat32m1_t a1r, vfloat32m1_t a1i,
    vfloat32m1_t a2r, vfloat32m1_t a2i,
    vfloat32m1_t w1r, vfloat32m1_t w1i,   
    vfloat32m1_t w2r, vfloat32m1_t w2i,   
    vfloat32m1_t* y0r, vfloat32m1_t* y0i, 
    vfloat32m1_t* y1r, vfloat32m1_t* y1i, 
    vfloat32m1_t* y2r, vfloat32m1_t* y2i, 
    size_t vl)
{
    vfloat32m1_t pr = __riscv_vfadd_vv_f32m1(a1r, a2r, vl); // p = a1+a2
    vfloat32m1_t pi = __riscv_vfadd_vv_f32m1(a1i, a2i, vl);
    vfloat32m1_t qr = __riscv_vfsub_vv_f32m1(a1r, a2r, vl); // q = a1-a2
    vfloat32m1_t qi = __riscv_vfsub_vv_f32m1(a1i, a2i, vl);
 
    *y0r = __riscv_vfadd_vv_f32m1(a0r, pr, vl); // y0 = a0 + p
    *y0i = __riscv_vfadd_vv_f32m1(a0i, pi, vl);
 
    vfloat32m1_t half_pr = __riscv_vfmul_vf_f32m1(pr, 0.5f, vl); 
    vfloat32m1_t half_pi = __riscv_vfmul_vf_f32m1(pi, 0.5f, vl); 

    vfloat32m1_t base_r = __riscv_vfsub_vv_f32m1(a0r, half_pr, vl); // base_r = a0r - pr*0.5
    vfloat32m1_t base_i = __riscv_vfsub_vv_f32m1(a0i, half_pi, vl); // base_i = a0i - pi*0.5
 
    vfloat32m1_t sq_qi = __riscv_vfmul_vf_f32m1(qi, SQRT3_OVER_2, vl); // sq_qi = qi*sqrt(3/2)
    vfloat32m1_t sq_qr = __riscv_vfmul_vf_f32m1(qr, SQRT3_OVER_2, vl); // sq_qr = qr*sqrt(3/2)

    vfloat32m1_t t1r = __riscv_vfadd_vv_f32m1(base_r, sq_qi, vl); // t1 = (base_r + sq_qi) + i*(base_i - sq_qr) 
    vfloat32m1_t t1i = __riscv_vfsub_vv_f32m1(base_i, sq_qr, vl); 
 
    vfloat32m1_t t2r = __riscv_vfnmsac_vf_f32m1(t1r, 2.0f, sq_qi, vl); // t2 = (base_r - sq_qi) + i*(base_i + sq_qr)
    vfloat32m1_t t2i = __riscv_vfmacc_vf_f32m1 (t1i, 2.0f, sq_qr, vl);
 
    cpx_cmul(t1r, t1i, w1r, w1i, y1r, y1i, vl); // y1_stored = t1·W1
    cpx_cmul(t2r, t2i, w2r, w2i, y2r, y2i, vl); // y2_stored = t2·W2
}


/* ---------------------------------------------------------------------------
 * r2_cpx_bfly — DIF radix-2 butterfly
 *
 *   ra, ia   first operand (real, imag) — input
 *   rb, ib   second operand (real, imag) — input
 *   rw, iw   twiddle factor W (real, imag) — input
 *   x1r, x1i sum result pointers (real, imag) — output
 *   x2r, x2i difference·W result pointers (real, imag) — output
 *   vl       vector length — input
 * --------------------------------------------------------------------------*/
static inline void r2_cpx_bfly(
    vfloat32m1_t ra, vfloat32m1_t ia,
    vfloat32m1_t rb, vfloat32m1_t ib,
    vfloat32m1_t rw, vfloat32m1_t iw,
    vfloat32m1_t* x1r, vfloat32m1_t* x1i,
    vfloat32m1_t* x2r, vfloat32m1_t* x2i,
    size_t vl)
{
    *x1r = __riscv_vfadd_vv_f32m1(ra, rb, vl); // x1r = a + b
    *x1i = __riscv_vfadd_vv_f32m1(ia, ib, vl); // x1i = i*a + i*b

    vfloat32m1_t dr = __riscv_vfsub_vv_f32m1(ra, rb, vl); // dr = a - b
    vfloat32m1_t di = __riscv_vfsub_vv_f32m1(ia, ib, vl); // di = i*a - i*b

    cpx_cmul(dr, di, rw, iw, x2r, x2i, vl);
}
/* ---------------------------------------------------------------------------
 * r4_stage_q1 —  specialized last radix-4 stage: grp_size=4, Q=1, W≡1
 *   vec       data array  — input/output
 *   N         FFT size — input
 *   In this case twiddle factors are not used
 * --------------------------------------------------------------------------*/
static void r4_stage_q1(float* restrict vec, int N)
{
    float* p = vec;
    const int ngroups = N >> 2;

    for (int i = 0; i < ngroups; ) 
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

        p += 8 * (int)vl;
        i += (int)vl;
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
static void r4_stage_q4(
    float* restrict vec, int N,
    const float* restrict tw_base)
{
    const size_t vl = __riscv_vsetvl_e32m1(4); 
    const int quarter = 4; 

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

    for (int g = 0; g < N; g += 16)
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
static void r4_stage(
    float* restrict vec,
    int N, int grp_size,
    const float* restrict tw_base,
    size_t vlmax)
{
    const int quarter = grp_size >> 2;

    for (int grp_start = 0; grp_start < N; grp_start += grp_size)
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

        int j = 0;

        /*  pipeline for quarters >= 2 * vlmax. Overlaps loading of next block with computation of current block */

        if (quarter >= 2 * (int)vlmax)
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

            j += (int)vl;

            /* pipeline kernel */
            for (; j + (int)vl <= quarter; j += (int)vl)
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

                cpx2v_store(sa, sb, j-(int)vl, y0r,y0i, y1r,y1i, vl); // store of block result
                cpx2v_store(sc, sd, j-(int)vl, y2r,y2i, y3r,y3i, vl);

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

                cpx2v_store(sa, sb, j-(int)vl, y0r,y0i, y1r,y1i, vl);
                cpx2v_store(sc, sd, j-(int)vl, y2r,y2i, y3r,y3i, vl);
            }
        }

        // section without pipeline for vlmax blocks
        for (; j + (int)vlmax <= quarter; j += (int)vlmax)
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

/* ---------------------------------------------------------------------------
 * r2_stage_h1 — specialized radix-2 stage for grp_size=2, H=1, W≡1
 *
 *   vec      data array — input/output
 *   N        full FFT size — input
 *
 *  In this case twiddle factors are not used
 * --------------------------------------------------------------------------*/
static void r2_stage_h1(float* restrict vec, int N)
{
    float* p = vec;
    const int ngroups = N >> 1;

    for (int i = 0; i < ngroups; )
    {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(ngroups - i));

        vfloat32m1x4_t seg = __riscv_vlseg4e32_v_f32m1x4(p, vl); // segmented load of 4 vectors 
        vfloat32m1_t ar = __riscv_vget_v_f32m1x4_f32m1(seg, 0);
        vfloat32m1_t ai = __riscv_vget_v_f32m1x4_f32m1(seg, 1);
        vfloat32m1_t br = __riscv_vget_v_f32m1x4_f32m1(seg, 2);
        vfloat32m1_t bi = __riscv_vget_v_f32m1x4_f32m1(seg, 3);

        vfloat32m1_t t0r = __riscv_vfadd_vv_f32m1(ar, br, vl); // t0 = a + b
        vfloat32m1_t t0i = __riscv_vfadd_vv_f32m1(ai, bi, vl);
        vfloat32m1_t t1r = __riscv_vfsub_vv_f32m1(ar, br, vl); // t1 = a - b
        vfloat32m1_t t1i = __riscv_vfsub_vv_f32m1(ai, bi, vl);

        __riscv_vsseg4e32_v_f32m1x4(p,
            __riscv_vcreate_v_f32m1x4(t0r, t0i, t1r, t1i), vl); // segmented store of 4 vectors

        p += 4 * (int)vl;
        i += (int)vl;
    }
}

/* ---------------------------------------------------------------------------
 * r2_stage_h2 — specialized radix-2 stage for grp_size=4, H=2, W[0]=1, W[1]=-i
 *
 *   vec      data array (in-place) — input/output
 *   N        full FFT size — input
 *
 *   In this case twiddle factors are handled without multiplications
 * --------------------------------------------------------------------------*/
static void r2_stage_h2(float* restrict vec, int N)
{
    float* p = vec;
    const int ngroups = N >> 2;

    for (int i = 0; i < ngroups; )
    {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(ngroups - i));

        vfloat32m1x8_t seg = __riscv_vlseg8e32_v_f32m1x8(p, vl); // segmented load of 8 vectors
        vfloat32m1_t a0r = __riscv_vget_v_f32m1x8_f32m1(seg, 0);
        vfloat32m1_t a0i = __riscv_vget_v_f32m1x8_f32m1(seg, 1);
        vfloat32m1_t a1r = __riscv_vget_v_f32m1x8_f32m1(seg, 2);
        vfloat32m1_t a1i = __riscv_vget_v_f32m1x8_f32m1(seg, 3);
        vfloat32m1_t b0r = __riscv_vget_v_f32m1x8_f32m1(seg, 4);
        vfloat32m1_t b0i = __riscv_vget_v_f32m1x8_f32m1(seg, 5);
        vfloat32m1_t b1r = __riscv_vget_v_f32m1x8_f32m1(seg, 6);
        vfloat32m1_t b1i = __riscv_vget_v_f32m1x8_f32m1(seg, 7);

        /* k=0, W=1 */ 
        vfloat32m1_t t0r = __riscv_vfadd_vv_f32m1(a0r, b0r, vl); // t0 = a0 + b0,
        vfloat32m1_t t0i = __riscv_vfadd_vv_f32m1(a0i, b0i, vl);
        vfloat32m1_t t1r = __riscv_vfsub_vv_f32m1(a0r, b0r, vl); // t1 = a0 - b0 
        vfloat32m1_t t1i = __riscv_vfsub_vv_f32m1(a0i, b0i, vl);

        /* k=1, W=-i */
        vfloat32m1_t t2r = __riscv_vfadd_vv_f32m1(a1r, b1r, vl); // t2 = a1 + b1
        vfloat32m1_t t2i = __riscv_vfadd_vv_f32m1(a1i, b1i, vl);
        vfloat32m1_t dr   = __riscv_vfsub_vv_f32m1(a1r, b1r, vl); // d = a1 - b1
        vfloat32m1_t di   = __riscv_vfsub_vv_f32m1(a1i, b1i, vl);
       
        vfloat32m1_t t3r = di;   // t3 = d·(-i) = (di, -dr) 
        vfloat32m1_t t3i = __riscv_vfneg_v_f32m1(dr, vl);

        __riscv_vsseg8e32_v_f32m1x8(p, // segmented store of 8 vectors
            __riscv_vcreate_v_f32m1x8(
            t0r, t0i, t2r, t2i,
            t1r, t1i,
            t3r, t3i), 
            vl);

        p += 8 * (int)vl;
        i += (int)vl;
    }
}

/* ===========================================================================
 * r3_stage — single radix-3 DIF stage (for N=1536, third=512)
 *
 *   vec      data array — input/output
 *   tw_r3    twiddle table  — input
 *   vlmax    maximum vector length — input
 *
 *   Generic case 
 * ===========================================================================*/
static void r3_stage(
    float* restrict vec,
    const float* restrict tw_r3,   
    size_t vlmax)
{

    const int third = 512;
 
    const float* restrict pa = vec; // 3 read and write pointers
    const float* restrict pb = vec + 2*third;
    const float* restrict pc = vec + 4*third;

    const float* tw_cursor = tw_r3; // moving pointer to current position in twiddle table

    float* restrict sa = vec;
    float* restrict sb = vec + 2*third;
    float* restrict sc = vec + 4*third;

    int j = 0;
 
    /*  pipeline for thirds >= 2 * vlmax. Overlaps loading of next block with computation of current block */
   
    if (third >= 2 * (int)vlmax)
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

        j += (int)vl;
 
        /* Kernel */
        for (; j + (int)vl <= third; j += (int)vl)
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
 
            cpx2v_store(sa, sb, j-(int)vl, y0r,y0i, y1r,y1i, vl); // store of block result
            cpx1v_store (sc, j-(int)vl, y2r, y2i, vl);
 
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
                
            cpx2v_store(sa, sb, j-(int)vl, y0r,y0i, y1r,y1i, vl);
            cpx1v_store (sc, j-(int)vl, y2r, y2i, vl);
        }
    }
 
    // section without pipeline for vlmax blocks
    for (; j + (int)vlmax <= third; j += (int)vlmax)
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

/* ===========================================================================
 * r2_stage — single radix-2 DIF stage with software pipeline
 *
 *   vec       data array  — input/output
 *   N         full FFT size — input
 *   grp_size  group size for this stage — input
 *   tw_stage  twiddle table block for this stage — input
 *   vlmax     maximum vector length — input
 *
 *   Generic case 
 * ===========================================================================*/
static void r2_stage(
    float* restrict vec,
    int N, int grp_size,
    const float* restrict tw_stage,
    size_t vlmax)
{
    const int half = grp_size >> 1;

    for (int grp_start = 0; grp_start < N; grp_start += grp_size)
    {
        float* base = &vec[2 * grp_start];

        const float* pa = base; // 2 read and write pointers
        const float* pb = base + 2 * half;

        const float* tw_cursor = tw_stage;  // moving pointer to current position in twiddle table

        float* sa = base;
        float* sb = base + 2 * half;

        int j = 0;

        /*  pipeline for halfs >= 2 * vlmax. Overlaps loading of next block with computation of current block */

        if (half >= 2 * (int)vlmax)
        {
            /* pipeline preload */

            const size_t vl = vlmax;

            vfloat32m1_t a0r, a0i,  // butterfly operands
                         a1r, a1i;

            vfloat32m1_t w1r, w1i; // twiddle factors

            cpx2v_load(pa, pb, j, vl, &a0r, &a0i, &a1r, &a1i); // preload

            r2_cpxt_load_stream(&tw_cursor, vl, &w1r, &w1i);

            j += (int)vl;

            for (; j + (int)vl <= half; j += (int)vl)
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

                cpx2v_store(sa, sb, j-(int)vl, y0r, y0i, y1r, y1i, vl); // store of block result

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

            cpx2v_store(sa, sb, j-(int)vl, y0r,y0i, y1r, y1i, vl);
        }

        // section without pipeline for vlmax blocks
        for (; j + (int)vlmax <= half; j += (int)vlmax)
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

            cpx2v_store(sa, sb, j-(int)vl, y0r,y0i, y1r, y1i, vl);
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
float* ofdm_fft(float* restrict vec, int N)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
 
    if (N == 1536) // special r3 path
    {
        const int third = 512;
        const float* tw_r3;
        get_twiddle_r3(&tw_r3); // get r3-table from outer file

        r3_stage(vec, tw_r3, vlmax); // single r3 stage
 
        int log2T;
        const float* tw2;
        const int*   off2;
        get_twiddle_r2(third, &tw2, &off2, &log2T); // get r2-table from outer file
 
        for (int blk = 0; blk < 3; blk++)
        {
            float* blk_base = vec + 2 * blk * third;
 
            for (int stage = log2T; stage >= 1; stage--) // specific stages dispatch
            {
                const int grp_size = 1 << stage;
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

    int log2N;
    const float* tw2;
    const int*   off2;
    get_twiddle_r2(N, &tw2, &off2, &log2N); // get r2-table from outer file
 
    const float* tw4;
    const int*   off4;
    int          stages_r4;
    get_twiddle_r4(N, &tw4, &off4, &stages_r4); // get r4-table from outer file
 
    if (log2N & 1) // log2n is odd 
    {
        const float* tw_stage = tw2 + 2 * off2[log2N - 1];
        r2_stage(vec, N, N, tw_stage, vlmax);
    }
 
    for (int r4_idx = stages_r4 - 1; r4_idx >= 0; r4_idx--) // specific stages dispatch
    {
        const int    grp_size = 1 << (2 + 2*r4_idx);
        const float* tw_base  = tw4 + 2 * off4[r4_idx];
 
        if (grp_size == 4)
        {
            r4_stage_q1(vec, N, vlmax);
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