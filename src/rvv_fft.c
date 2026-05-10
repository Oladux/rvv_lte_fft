#include "../include/rvv_fft.h"
#include "../include/rvv_fft_misc.h"
 
/* ---------------------------------------------------------------------------
 * load_data
 *   Загружает два interleaved комплексных вектора из *pa и *pb,
 *   оба указателя сдвигаются вперёд на 2*vl float'ов.
 *   Формат: [Re0, Im0, Re1, Im1, ...] → vlseg2 разделяет Re/Im.
 * --------------------------------------------------------------------------*/
static inline void load_data(
    const float* restrict *pa,
    const float* restrict *pb,
    size_t       vl,
    vfloat32m1_t* ra,
    vfloat32m1_t* ia,
    vfloat32m1_t* rb,
    vfloat32m1_t* ib)
{
    vfloat32m1x2_t va = __riscv_vlseg2e32_v_f32m1x2(*pa, vl);
    vfloat32m1x2_t vb = __riscv_vlseg2e32_v_f32m1x2(*pb, vl);
 
    *ra = __riscv_vget_v_f32m1x2_f32m1(va, 0);
    *ia = __riscv_vget_v_f32m1x2_f32m1(va, 1);
    *rb = __riscv_vget_v_f32m1x2_f32m1(vb, 0);
    *ib = __riscv_vget_v_f32m1x2_f32m1(vb, 1);
 
    *pa += 2 * vl;
    *pb += 2 * vl;
}
 
/* ---------------------------------------------------------------------------
 * store_data
 *   Записывает два комплексных вектора в interleaved формат,
 *   оба указателя сдвигаются вперёд на 2*vl float'ов.
 * --------------------------------------------------------------------------*/
static inline void store_data(
    float* restrict *pa,
    float* restrict *pb,
    vfloat32m1_t x1r, vfloat32m1_t x1i,
    vfloat32m1_t x2r, vfloat32m1_t x2i,
    size_t vl)
{
    vfloat32m1x2_t out1 = __riscv_vcreate_v_f32m1x2(x1r, x1i);
    vfloat32m1x2_t out2 = __riscv_vcreate_v_f32m1x2(x2r, x2i);
 
    __riscv_vsseg2e32_v_f32m1x2(*pa, out1, vl);
    __riscv_vsseg2e32_v_f32m1x2(*pb, out2, vl);
 
    *pa += 2 * vl;
    *pb += 2 * vl;
}
 
/* ---------------------------------------------------------------------------
 * load_twiddles_stream
 *   Последовательная загрузка vl твидл-факторов, курсор сдвигается.
 *   Формат: [Re0, Im0, Re1, Im1, ...]
 * --------------------------------------------------------------------------*/
static inline void load_twiddles_stream(
    const float* restrict *tw_cursor,
    size_t       vl,
    vfloat32m1_t* rw,
    vfloat32m1_t* iw)
{
    vfloat32m1x2_t tw = __riscv_vlseg2e32_v_f32m1x2(*tw_cursor, vl);
 
    *rw = __riscv_vget_v_f32m1x2_f32m1(tw, 0);
    *iw = __riscv_vget_v_f32m1x2_f32m1(tw, 1);
 
    *tw_cursor += 2 * vl;
}
 
/* ---------------------------------------------------------------------------
 * load_twiddles_radix4
 *   Загружает W1, W2, W3 для позиций j..j+vl-1 из таблицы текущей r4-стадии.
 *
 *   Предполагаемый формат таблицы (формируется get_twiddle_r4):
 *     [ W1[0], W1[1], ..., W1[Q-1],       <- Q = quarter записей {Re,Im}
 *       W2[0], W2[1], ..., W2[Q-1],
 *       W3[0], W3[1], ..., W3[Q-1] ]
 *
 *   Смещения:
 *     W1[j] → tw_base + 2*j
 *     W2[j] → tw_base + 2*(quarter + j)
 *     W3[j] → tw_base + 2*(2*quarter + j)
 *
 *   BUG FIX: оригинал использовал j*stride / 2*j*stride / 3*j*stride,
 *   что давало неверные адреса для любого j > 0 и stride > 1.
 *   Параметр stride заменён на quarter.
 * --------------------------------------------------------------------------*/
static inline void load_twiddles_radix4(
    const float* tw_base,
    int j,
    int quarter,
    size_t vl,
    vfloat32m1_t* w1r, vfloat32m1_t* w1i,
    vfloat32m1_t* w2r, vfloat32m1_t* w2i,
    vfloat32m1_t* w3r, vfloat32m1_t* w3i)
{
    const float* w1_base = tw_base;
    const float* w2_base = tw_base + 2 * quarter;
    const float* w3_base = tw_base + 4 * quarter;

    const float* w1 = w1_base + 2 * j;
    const float* w2 = w2_base + 2 * j;
    const float* w3 = w3_base + 2 * j;

    vfloat32m1x2_t t1 = __riscv_vlseg2e32_v_f32m1x2(w1, vl);
    vfloat32m1x2_t t2 = __riscv_vlseg2e32_v_f32m1x2(w2, vl);
    vfloat32m1x2_t t3 = __riscv_vlseg2e32_v_f32m1x2(w3, vl);

    *w1r = __riscv_vget_v_f32m1x2_f32m1(t1, 0);
    *w1i = __riscv_vget_v_f32m1x2_f32m1(t1, 1);

    *w2r = __riscv_vget_v_f32m1x2_f32m1(t2, 0);
    *w2i = __riscv_vget_v_f32m1x2_f32m1(t2, 1);

    *w3r = __riscv_vget_v_f32m1x2_f32m1(t3, 0);
    *w3i = __riscv_vget_v_f32m1x2_f32m1(t3, 1);
}
/* ---------------------------------------------------------------------------
 * butterfly_radix4
 *   DIT radix-4 butterfly.
 *   a0 входит без домножения (W^0 = 1).
 *   a1, a2, a3 домножаются на W1, W2, W3 внутри функции.
 *
 *   Формулы (стандартная DIT radix-4 декомпозиция):
 *     b_m = a_m · W_m
 *     t0  = a0 + b2,  t1 = a0 - b2
 *     t2  = b1 + b3,  t3 = (b1 - b3)·(-i)
 *     y0  = t0 + t2   → позиция k
 *     y1  = t1 + t3   → позиция k+Q
 *     y2  = t0 - t2   → позиция k+2Q
 *     y3  = t1 - t3   → позиция k+3Q
 * --------------------------------------------------------------------------*/
static inline void butterfly_radix4(
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
    /* b1 = a1 · W1 */
    vfloat32m1_t b1r = __riscv_vfmul_vv_f32m1(a1r, w1r, vl);
    b1r = __riscv_vfnmsac_vv_f32m1(b1r, a1i, w1i, vl);
    vfloat32m1_t b1i = __riscv_vfmul_vv_f32m1(a1i, w1r, vl);
    b1i = __riscv_vfmacc_vv_f32m1(b1i, a1r, w1i, vl);
 
    /* b2 = a2 · W2 */
    vfloat32m1_t b2r = __riscv_vfmul_vv_f32m1(a2r, w2r, vl);
    b2r = __riscv_vfnmsac_vv_f32m1(b2r, a2i, w2i, vl);
    vfloat32m1_t b2i = __riscv_vfmul_vv_f32m1(a2i, w2r, vl);
    b2i = __riscv_vfmacc_vv_f32m1(b2i, a2r, w2i, vl);
 
    /* b3 = a3 · W3 */
    vfloat32m1_t b3r = __riscv_vfmul_vv_f32m1(a3r, w3r, vl);
    b3r = __riscv_vfnmsac_vv_f32m1(b3r, a3i, w3i, vl);
    vfloat32m1_t b3i = __riscv_vfmul_vv_f32m1(a3i, w3r, vl);
    b3i = __riscv_vfmacc_vv_f32m1(b3i, a3r, w3i, vl);
 
    /* t0 = a0 + b2,  t1 = a0 - b2 */
    vfloat32m1_t t0r = __riscv_vfadd_vv_f32m1(a0r, b2r, vl);
    vfloat32m1_t t0i = __riscv_vfadd_vv_f32m1(a0i, b2i, vl);
    vfloat32m1_t t1r = __riscv_vfsub_vv_f32m1(a0r, b2r, vl);
    vfloat32m1_t t1i = __riscv_vfsub_vv_f32m1(a0i, b2i, vl);
 
    /* t2 = b1 + b3 */
    vfloat32m1_t t2r = __riscv_vfadd_vv_f32m1(b1r, b3r, vl);
    vfloat32m1_t t2i = __riscv_vfadd_vv_f32m1(b1i, b3i, vl);
 
    /* t3 = (b1 - b3) · (-i) = (d_r + i·d_i)·(-i) = d_i - i·d_r */
    vfloat32m1_t d_r = __riscv_vfsub_vv_f32m1(b1r, b3r, vl);
    vfloat32m1_t d_i = __riscv_vfsub_vv_f32m1(b1i, b3i, vl);
    vfloat32m1_t t3r = d_i;
    vfloat32m1_t t3i = __riscv_vfneg_v_f32m1(d_r, vl);
 
    /* outputs */
    *y0r = __riscv_vfadd_vv_f32m1(t0r, t2r, vl);
    *y0i = __riscv_vfadd_vv_f32m1(t0i, t2i, vl);
    *y2r = __riscv_vfsub_vv_f32m1(t0r, t2r, vl);
    *y2i = __riscv_vfsub_vv_f32m1(t0i, t2i, vl);
    *y1r = __riscv_vfadd_vv_f32m1(t1r, t3r, vl);
    *y1i = __riscv_vfadd_vv_f32m1(t1i, t3i, vl);
    *y3r = __riscv_vfsub_vv_f32m1(t1r, t3r, vl);
    *y3i = __riscv_vfsub_vv_f32m1(t1i, t3i, vl);
}
 
/* ---------------------------------------------------------------------------
 * core_radix2
 *   DIT radix-2 butterfly с комплексным умножением через FMA:
 *     mr = rb·rw − ib·iw
 *     mi = ib·rw + rb·iw
 *     x1 = a + m,   x2 = a − m
 * --------------------------------------------------------------------------*/
static inline void core_radix2(
    vfloat32m1_t ra, vfloat32m1_t ia,
    vfloat32m1_t rb, vfloat32m1_t ib,
    vfloat32m1_t rw, vfloat32m1_t iw,
    vfloat32m1_t* x1r, vfloat32m1_t* x1i,
    vfloat32m1_t* x2r, vfloat32m1_t* x2i,
    size_t vl)
{
    vfloat32m1_t mr = __riscv_vfmul_vv_f32m1(rb, rw, vl);
    mr = __riscv_vfnmsac_vv_f32m1(mr, ib, iw, vl);
 
    vfloat32m1_t mi = __riscv_vfmul_vv_f32m1(ib, rw, vl);
    mi = __riscv_vfmacc_vv_f32m1(mi, rb, iw, vl);
 
    *x1r = __riscv_vfadd_vv_f32m1(ra, mr, vl);
    *x1i = __riscv_vfadd_vv_f32m1(ia, mi, vl);
    *x2r = __riscv_vfsub_vv_f32m1(ra, mr, vl);
    *x2i = __riscv_vfsub_vv_f32m1(ia, mi, vl);
}
 
/* ===========================================================================
 * rvv_fft
 *
 * In-place Cooley-Tukey DIT FFT, mixed radix-4 / radix-2.
 * vec  — float-массив длиной 2·N (interleaved Re/Im).
 * N    — размер БПФ, степень двойки.
 *
 * Порядок стадий для LTE-размеров:
 *   log2N чётный (N=256,1024):  все стадии radix-4
 *   log2N нечётный (N=128,512,2048): stages_r4 стадий r4 + 1 стадия r2
 *
 * Исправленные баги:
 *   BUG1 — stage-счётчик после r4-loop был на +1 выше нужного значения,
 *           что приводило к пропуску последней r2-стадии для нечётных log2N.
 *           Фикс: stage = 2*stages_r4 + 1 перед r2-loop.
 *
 *   BUG2 — load_twiddles_radix4 вычислял смещения через j*stride / 2*j*stride,
 *           что давало неверные адреса для всех j>0 при stride>1.
 *           Фикс: W1→tw_base+2*j, W2→tw_base+2*(Q+j), W3→tw_base+2*(2*Q+j).
 * ===========================================================================*/
float* rvv_fft(float* restrict vec, int N)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
 
    int log2N;
    const float* tw2;
    const int*   off2;
    get_twiddle_r2(N, &tw2, &off2, &log2N);
 
    const float* tw4;
    const int*   off4;
    int          stages_r4;

    get_twiddle_r4(N, &tw4, &off4, &stages_r4);
 
    /* stage — номер эквивалентной radix-2 стадии для grp_size = 1<<stage.
     * r4 начинает с grp_size=4 (stage=2) и каждую итерацию потребляет
     * две radix-2 стадии (stage += 2). */
   // Временная замена radix-4 стадий на динамические твидлы
int stage = 2;
for (int s = 0; s < stages_r4; s++) {
    int grp_size = 1 << stage;
    int quarter  = grp_size >> 2;
    int num_groups = N / grp_size;

    for (int g = 0; g < num_groups; g++) {
        float* base = &vec[2 * g * grp_size];

        const float* pa = base;
        const float* pb = base + 2 * quarter;
        const float* pc = base + 4 * quarter;
        const float* pd = base + 6 * quarter;
        float* sa = base;
        float* sb = base + 2 * quarter;
        float* sc = base + 4 * quarter;
        float* sd = base + 6 * quarter;

        for (int k = 0; k < quarter; ) {
            size_t vl = __riscv_vsetvl_e32m1(quarter - k);
            vfloat32m1_t a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
            load_data(&pa, &pb, vl, &a0r, &a0i, &a1r, &a1i);
            load_data(&pc, &pd, vl, &a2r, &a2i, &a3r, &a3i);

            // Динамические твидлы, совместимые с бит‑реверсом
            float theta_base = -2.0f * M_PI * (g * grp_size) / N;
            float theta_step = -2.0f * M_PI / grp_size;

            float w1r_buf[vlmax], w1i_buf[vlmax];
            float w2r_buf[vlmax], w2i_buf[vlmax];
            float w3r_buf[vlmax], w3i_buf[vlmax];

            for (size_t i = 0; i < vl; i++) {
                float theta = theta_base + theta_step * (k + i);
                // W1 = exp(-j * theta)
                w1r_buf[i] = cosf(theta);
                w1i_buf[i] = sinf(theta);
                // W2 = exp(-j * 2*theta)
                w2r_buf[i] = cosf(2.0f * theta);
                w2i_buf[i] = sinf(2.0f * theta);
                // W3 = exp(-j * 3*theta)
                w3r_buf[i] = cosf(3.0f * theta);
                w3i_buf[i] = sinf(3.0f * theta);
            }

            vfloat32m1_t w1r = __riscv_vle32_v_f32m1(w1r_buf, vl);
            vfloat32m1_t w1i = __riscv_vle32_v_f32m1(w1i_buf, vl);
            vfloat32m1_t w2r = __riscv_vle32_v_f32m1(w2r_buf, vl);
            vfloat32m1_t w2i = __riscv_vle32_v_f32m1(w2i_buf, vl);
            vfloat32m1_t w3r = __riscv_vle32_v_f32m1(w3r_buf, vl);
            vfloat32m1_t w3i = __riscv_vle32_v_f32m1(w3i_buf, vl);

            vfloat32m1_t y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            butterfly_radix4(a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i,
                             w1r, w1i, w2r, w2i, w3r, w3i,
                             &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i, vl);

            store_data(&sa, &sc, y0r, y0i, y2r, y2i, vl);
            store_data(&sb, &sd, y1r, y1i, y3r, y3i, vl);
            k += vl;
        }
    }
    stage += 2;
}

    stage = 2 * stages_r4 + 1;  

    for (; stage <= log2N; stage++)
    {
        const int grp_size = 1 << stage;
        const int grp_half = grp_size >> 1;
 
        const float* tw_stage = tw2 + 2 * off2[stage - 1];
 
        for (int grp_start = 0; grp_start < N; grp_start += grp_size)
        {
            float* pa_base = &vec[2 * grp_start];
            float* pb_base = &vec[2 * (grp_start + grp_half)];
 
            const float* pa = pa_base;
            const float* pb = pb_base;
            float*       sa = pa_base;
            float*       sb = pb_base;
 
            const float* tw_ptr = tw_stage;
            int j = 0;
 
            /* Pipeline: активен когда есть минимум 2 полных блока */
            if (grp_half >= 2 * (int)vlmax)
            {
                const size_t vl = vlmax;
                vfloat32m1_t ra,ia,rb,ib,rw,iw;
 
                load_data(&pa, &pb, vl, &ra, &ia, &rb, &ib);
                load_twiddles_stream(&tw_ptr, vl, &rw, &iw);
                j += (int)vl;
 
                for (; j + (int)vl <= grp_half; j += (int)vl)
                {
                    vfloat32m1_t ra_n,ia_n,rb_n,ib_n,rw_n,iw_n;
                    load_data(&pa, &pb, vl, &ra_n, &ia_n, &rb_n, &ib_n);
                    load_twiddles_stream(&tw_ptr, vl, &rw_n, &iw_n);
 
                    vfloat32m1_t x1r,x1i,x2r,x2i;
                    core_radix2(ra,ia,rb,ib,rw,iw,
                                &x1r,&x1i,&x2r,&x2i, vl);
                    store_data(&sa, &sb, x1r,x1i,x2r,x2i, vl);
 
                    ra=ra_n; ia=ia_n; rb=rb_n; ib=ib_n;
                    rw=rw_n; iw=iw_n;
                }
 
                /* Drain */
                vfloat32m1_t x1r,x1i,x2r,x2i;
                core_radix2(ra,ia,rb,ib,rw,iw,
                            &x1r,&x1i,&x2r,&x2i, vl);
                store_data(&sa, &sb, x1r,x1i,x2r,x2i, vl);
            }
 
            /* Normal: полные vlmax-блоки без pipeline */
            for (; j + (int)vlmax <= grp_half; j += (int)vlmax)
            {
                const size_t vl = vlmax;
                vfloat32m1_t ra,ia,rb,ib,rw,iw,x1r,x1i,x2r,x2i;
 
                load_data(&pa, &pb, vl, &ra, &ia, &rb, &ib);
                load_twiddles_stream(&tw_ptr, vl, &rw, &iw);
                core_radix2(ra,ia,rb,ib,rw,iw,
                            &x1r,&x1i,&x2r,&x2i, vl);
                store_data(&sa, &sb, x1r,x1i,x2r,x2i, vl);
            }
 
            /* Tail */
            if (j < grp_half)
            {
                const size_t vl = __riscv_vsetvl_e32m1(
                    (size_t)(grp_half - j));
                vfloat32m1_t ra,ia,rb,ib,rw,iw,x1r,x1i,x2r,x2i;
 
                load_data(&pa, &pb, vl, &ra, &ia, &rb, &ib);
                load_twiddles_stream(&tw_ptr, vl, &rw, &iw);
                core_radix2(ra,ia,rb,ib,rw,iw,
                            &x1r,&x1i,&x2r,&x2i, vl);
                store_data(&sa, &sb, x1r,x1i,x2r,x2i, vl);
            }
        }
    }
 
    return vec;
}
