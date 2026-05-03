#include "../include/rvv_fft.h"
#include "../include/rvv_fft_misc.h"

static inline void load_twiddles(
    const float* tw,
    int j,
    int stride,
    size_t vl,
    vfloat32m1_t* rw,
    vfloat32m1_t* iw)
{
    if (stride == 1)
    {
        vfloat32m1x2_t vw =
            __riscv_vlseg2e32_v_f32m1x2(&tw[2*j], vl);

        *rw = vw.__val[0];
        *iw = vw.__val[1];
    }
    else
    {
        const uint32_t complex_size = 2 * sizeof(float);

        vuint32m1_t vid = __riscv_vid_v_u32m1(vl);

        vuint32m1_t idx =
            __riscv_vadd_vx_u32m1(vid, j, vl);

        idx = __riscv_vmul_vx_u32m1(idx, stride, vl);

        idx = __riscv_vmul_vx_u32m1(idx, complex_size, vl);

        *rw = __riscv_vluxei32_v_f32m1(tw, idx, vl);

        vuint32m1_t idx_im =
            __riscv_vadd_vx_u32m1(idx, sizeof(float), vl);

        *iw = __riscv_vluxei32_v_f32m1(tw, idx_im, vl);
    }
}

static inline void butterfly(
    vfloat32m1_t ra, vfloat32m1_t ia,
    vfloat32m1_t rb, vfloat32m1_t ib,
    vfloat32m1_t rw, vfloat32m1_t iw,
    vfloat32m1_t* x1r, vfloat32m1_t* x1i,
    vfloat32m1_t* x2r, vfloat32m1_t* x2i,
    size_t vl)
{
    vfloat32m1_t t1 = __riscv_vfmul_vv_f32m1(rb, rw, vl);
    vfloat32m1_t t2 = __riscv_vfmul_vv_f32m1(ib, rw, vl);

    vfloat32m1_t mr =
        __riscv_vfnmsac_vv_f32m1(t1, ib, iw, vl);

    vfloat32m1_t mi =
        __riscv_vfmacc_vv_f32m1(t2, rb, iw, vl);

    *x1r = __riscv_vfadd_vv_f32m1(ra, mr, vl);
    *x1i = __riscv_vfadd_vv_f32m1(ia, mi, vl);

    *x2r = __riscv_vfsub_vv_f32m1(ra, mr, vl);
    *x2i = __riscv_vfsub_vv_f32m1(ia, mi, vl);
}

float* rvv_fft(float* restrict vec, int N, int log2N)
{
    size_t vl;

    vfloat32m1x2_t va, vb;

    vfloat32m1_t  ra, ia,
                  rb, ib,
                  rw, iw,
                  x1r, x1i,
                  x2r, x2i;

    int tw_cnt;
    const float* tw_table = get_twiddle_table(N, &tw_cnt);

    size_t vlmax = __riscv_vsetvlmax_e32m1(); // ?

    for (int stage = 1; stage <= log2N; stage++)
    {
        int grp_size = 1 << stage;
        int grp_half = grp_size >> 1;
        int stride   = N / grp_size;

        for (int grp_start = 0; grp_start < N; grp_start += grp_size)
        {
            float* pa = &vec[2 * grp_start]; // first half of data
            float* pb = &vec[2 * (grp_start + grp_half)]; // second half of data

            int j = 0;

            for (; j + vlmax <= grp_half; j += vlmax)
            {
                vl = vlmax; // set of data len for this group

                va = __riscv_vlseg2e32_v_f32m1x2(pa + 2*j, vl); // load of interlieved data of a 

                vb = __riscv_vlseg2e32_v_f32m1x2(pb + 2*j, vl); // load of interlieved data of b

                ra = va.__val[0]; // real part of interlieved data of a 
                ia = va.__val[1]; // im part of interlieved data of a 
                rb = vb.__val[0]; // real part of interlieved data of b 
                ib = vb.__val[1]; // im part of interlieved data of b 

                rw, iw; // regs for twiddles

                load_twiddles(tw_table, j, stride, vl, &rw, &iw);

                x1r, x1i, x2r, x2i; // regs for result of butterfly

                butterfly(ra, ia, rb, ib, rw, iw,
                          &x1r, &x1i, &x2r, &x2i, vl);

                vfloat32m1x2_t vout1 = {x1r, x1i}; // interlieved out of butterfly
                vfloat32m1x2_t vout2 = {x2r, x2i}; // interlieved out of butterfly

                __riscv_vsseg2e32_v_f32m1x2(pa + 2*j, vout1, vl); // out of A data
                __riscv_vsseg2e32_v_f32m1x2(pb + 2*j, vout2, vl); // out of B data
            }

            if (j < grp_half)
            {
                vl = __riscv_vsetvl_e32m1(grp_half - j);

                va =  __riscv_vlseg2e32_v_f32m1x2(pa + 2*j, vl);

                vb =  __riscv_vlseg2e32_v_f32m1x2(pb + 2*j, vl);

                ra = va.__val[0];
                ia = va.__val[1];
                rb = vb.__val[0];
                ib = vb.__val[1];

                rw, iw;

                load_twiddles(tw_table, j, stride, vl, &rw, &iw);

                x1r, x1i, x2r, x2i;

                butterfly(ra, ia, rb, ib, rw, iw,
                          &x1r, &x1i, &x2r, &x2i, vl);

                vfloat32m1x2_t vout1 = {x1r, x1i};
                vfloat32m1x2_t vout2 = {x2r, x2i};

                __riscv_vsseg2e32_v_f32m1x2(pa + 2*j, vout1, vl);
                __riscv_vsseg2e32_v_f32m1x2(pb + 2*j, vout2, vl);
            }
        }
    }

    return vec;
}