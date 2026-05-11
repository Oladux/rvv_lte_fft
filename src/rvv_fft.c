#include "../include/rvv_fft.h"
#include "../include/rvv_fft_misc.h"
 

/*
* ===========================================================================
 * DIF mixed radix-4 / radix-2 FFT
 *
 * Структура алгоритма:
 *   Вход:  натуральный порядок (не нужен bit-reversal на входе)
 *   Выход: bit-reversed порядок → bit_reverse_permutation() в конце
 *
 * Порядок стадий (от крупных групп к мелким):
 *   log2N нечётный (N=128,512,2048):  r2(G=N) → r4(G=N/2) → … → r4(G=4)
 *   log2N чётный  (N=256,1024):       r4(G=N) → r4(G=N/4) → … → r4(G=4)
 *
 * Butterfly DIF r4 — вывод хранится в bit-compatible порядке:
 *   Стандартный DIF r4 даёт y0,y1,y2,y3.
 *   Для совместимости со стандартным bit-reversal на выходе
 *   y1 и y2 меняются местами при записи:
 *     pa[k] = y0 = t0+t2             (quarter 0, нет twiddle)
 *     pb[k] = y2 = (t0−t2)·W2        (quarter 1, W2)  ← swap
 *     pc[k] = y1 = (t1+t3)·W1        (quarter 2, W1)  ← swap
 *     pd[k] = y3 = (t1−t3)·W3        (quarter 3, W3)
 *
 * Twiddle-таблица: идентична DIT r4 — get_twiddle_r4() без изменений.
 * ===========================================================================*/
/* ---------------------------------------------------------------------------
 * load_data  —  загружает два interleaved комплексных вектора
 * --------------------------------------------------------------------------*/
static inline void load_data(
    const float* restrict pa,
    const float* restrict pb,
    int          j,
    size_t       vl,
    vfloat32m1_t* ra, vfloat32m1_t* ia,
    vfloat32m1_t* rb, vfloat32m1_t* ib)
{
    vfloat32m1x2_t va = __riscv_vlseg2e32_v_f32m1x2(pa + 2*j, vl);
    vfloat32m1x2_t vb = __riscv_vlseg2e32_v_f32m1x2(pb + 2*j, vl);

    *ra = __riscv_vget_v_f32m1x2_f32m1(va, 0);
    *ia = __riscv_vget_v_f32m1x2_f32m1(va, 1);
    *rb = __riscv_vget_v_f32m1x2_f32m1(vb, 0);
    *ib = __riscv_vget_v_f32m1x2_f32m1(vb, 1);
}

/* ---------------------------------------------------------------------------
 * store_data  —  записывает два комплексных вектора в interleaved формат
 * --------------------------------------------------------------------------*/
static inline void store_data(
    float* restrict pa,
    float* restrict pb,
    int    j,
    vfloat32m1_t x1r, vfloat32m1_t x1i,
    vfloat32m1_t x2r, vfloat32m1_t x2i,
    size_t vl)
{
    __riscv_vsseg2e32_v_f32m1x2(pa + 2*j,
        __riscv_vcreate_v_f32m1x2(x1r, x1i), vl);
    __riscv_vsseg2e32_v_f32m1x2(pb + 2*j,
        __riscv_vcreate_v_f32m1x2(x2r, x2i), vl);
}

/* ---------------------------------------------------------------------------
 * load_twiddles_stream  —  последовательная загрузка twiddle-факторов
 * --------------------------------------------------------------------------*/
static inline void load_twiddles_stream(
    const float* restrict *tw_cursor,
    size_t       vl,
    vfloat32m1_t* rw, vfloat32m1_t* iw)
{
    vfloat32m1x2_t tw = __riscv_vlseg2e32_v_f32m1x2(*tw_cursor, vl);
    *rw = __riscv_vget_v_f32m1x2_f32m1(tw, 0);
    *iw = __riscv_vget_v_f32m1x2_f32m1(tw, 1);
    *tw_cursor += 2 * vl;
}

/* ---------------------------------------------------------------------------
 * load_twiddles_r4
 *   Загружает W1, W2, W3 для позиций j..j+vl-1.
 *   Формат блока таблицы: [W1[0..Q-1], W2[0..Q-1], W3[0..Q-1]]
 *   Тот же формат и те же значения что у DIT r4.
 * --------------------------------------------------------------------------*/
static inline void load_twiddles_r4(
    const float* restrict tw_base,
    int   j, int quarter, size_t vl,
    vfloat32m1_t* w1r, vfloat32m1_t* w1i,
    vfloat32m1_t* w2r, vfloat32m1_t* w2i,
    vfloat32m1_t* w3r, vfloat32m1_t* w3i)
{
    const float* p1 = tw_base + 2 * j;
    const float* p2 = tw_base + 2 * (quarter + j);
    const float* p3 = tw_base + 2 * (2*quarter + j);

    vfloat32m1x2_t t1 = __riscv_vlseg2e32_v_f32m1x2(p1, vl);
    vfloat32m1x2_t t2 = __riscv_vlseg2e32_v_f32m1x2(p2, vl);
    vfloat32m1x2_t t3 = __riscv_vlseg2e32_v_f32m1x2(p3, vl);

    *w1r = __riscv_vget_v_f32m1x2_f32m1(t1, 0);
    *w1i = __riscv_vget_v_f32m1x2_f32m1(t1, 1);
    *w2r = __riscv_vget_v_f32m1x2_f32m1(t2, 0);
    *w2i = __riscv_vget_v_f32m1x2_f32m1(t2, 1);
    *w3r = __riscv_vget_v_f32m1x2_f32m1(t3, 0);
    *w3i = __riscv_vget_v_f32m1x2_f32m1(t3, 1);
}

/* ---------------------------------------------------------------------------
 * cmul  —  комплексное умножение через FMA: out = (ar+i·ai)·(br+i·bi)
 * --------------------------------------------------------------------------*/
static inline void cmul(
    vfloat32m1_t ar, vfloat32m1_t ai,
    vfloat32m1_t br, vfloat32m1_t bi,
    vfloat32m1_t* outr, vfloat32m1_t* outi,
    size_t vl)
{
    *outr = __riscv_vfmul_vv_f32m1(ar, br, vl);
    *outr = __riscv_vfnmsac_vv_f32m1(*outr, ai, bi, vl);
    *outi = __riscv_vfmul_vv_f32m1(ai, br, vl);
    *outi = __riscv_vfmacc_vv_f32m1(*outi, ar, bi, vl);
}

/* ---------------------------------------------------------------------------
 * butterfly_r4_dif
 *
 *   Входы: a0 (quarter 0), a1 (quarter 1), a2 (quarter 2), a3 (quarter 3)
 *
 *   Вычисление:
 *     t0 = a0 + a2,   t1 = a0 − a2
 *     t2 = a1 + a3,   t3 = (a1 − a3)·(−i)   [т.е. t3 = (di − i·dr)]
 *
 *   Выходы в bit-compatible порядке (swap y1↔y2):
 *     *y0 = t0 + t2            → quarter 0  (нет twiddle)
 *     *y1 = (t0 − t2)·W2       → quarter 1  ← здесь W2, не W1
 *     *y2 = (t1 + t3)·W1       → quarter 2  ← здесь W1, не W2
 *     *y3 = (t1 − t3)·W3       → quarter 3
 *
 *   Почему swap: стандартный DIF r4 записывает {y0,y1,y2,y3} в позиции
 *   {0,Q,2Q,3Q}, что требует digit-reversal base-4 в конце. Swap позиций
 *   y1 и y2 эквивалентен bit-reversal внутри 2-битного индекса (01↔10),
 *   после чего финальный стандартный bit-reversal даёт правильный порядок.
 * --------------------------------------------------------------------------*/
static inline void butterfly_r4_dif(
    vfloat32m1_t a0r, vfloat32m1_t a0i,
    vfloat32m1_t a1r, vfloat32m1_t a1i,
    vfloat32m1_t a2r, vfloat32m1_t a2i,
    vfloat32m1_t a3r, vfloat32m1_t a3i,
    vfloat32m1_t w1r, vfloat32m1_t w1i,
    vfloat32m1_t w2r, vfloat32m1_t w2i,
    vfloat32m1_t w3r, vfloat32m1_t w3i,
    vfloat32m1_t* y0r, vfloat32m1_t* y0i,  /* → quarter 0 (pa) */
    vfloat32m1_t* y1r, vfloat32m1_t* y1i,  /* → quarter 1 (pb), умножено на W2 */
    vfloat32m1_t* y2r, vfloat32m1_t* y2i,  /* → quarter 2 (pc), умножено на W1 */
    vfloat32m1_t* y3r, vfloat32m1_t* y3i,  /* → quarter 3 (pd), умножено на W3 */
    size_t vl)
{
    /* t0 = a0+a2,  t1 = a0−a2 */
    vfloat32m1_t t0r = __riscv_vfadd_vv_f32m1(a0r, a2r, vl);
    vfloat32m1_t t0i = __riscv_vfadd_vv_f32m1(a0i, a2i, vl);
    vfloat32m1_t t1r = __riscv_vfsub_vv_f32m1(a0r, a2r, vl);
    vfloat32m1_t t1i = __riscv_vfsub_vv_f32m1(a0i, a2i, vl);

    /* t2 = a1+a3 */
    vfloat32m1_t t2r = __riscv_vfadd_vv_f32m1(a1r, a3r, vl);
    vfloat32m1_t t2i = __riscv_vfadd_vv_f32m1(a1i, a3i, vl);

    /* t3 = (a1−a3)·(−i):  (dr+i·di)·(−i) = di − i·dr */
    vfloat32m1_t dr  = __riscv_vfsub_vv_f32m1(a1r, a3r, vl);
    vfloat32m1_t di  = __riscv_vfsub_vv_f32m1(a1i, a3i, vl);
    vfloat32m1_t t3r = di;
    vfloat32m1_t t3i = __riscv_vfneg_v_f32m1(dr, vl);

    /* y0 = t0+t2  (no twiddle → quarter 0) */
    *y0r = __riscv_vfadd_vv_f32m1(t0r, t2r, vl);
    *y0i = __riscv_vfadd_vv_f32m1(t0i, t2i, vl);

    /* Временные: s0=t0−t2, s1=t1+t3, s2=t1−t3 */
    vfloat32m1_t s0r = __riscv_vfsub_vv_f32m1(t0r, t2r, vl);
    vfloat32m1_t s0i = __riscv_vfsub_vv_f32m1(t0i, t2i, vl);
    vfloat32m1_t s1r = __riscv_vfadd_vv_f32m1(t1r, t3r, vl);
    vfloat32m1_t s1i = __riscv_vfadd_vv_f32m1(t1i, t3i, vl);
    vfloat32m1_t s2r = __riscv_vfsub_vv_f32m1(t1r, t3r, vl);
    vfloat32m1_t s2i = __riscv_vfsub_vv_f32m1(t1i, t3i, vl);

    /* y1_stored = s0·W2 → quarter 1 (pb)   [swap: W2 идёт в позицию 1] */
    cmul(s0r, s0i, w2r, w2i, y1r, y1i, vl);

    /* y2_stored = s1·W1 → quarter 2 (pc)   [swap: W1 идёт в позицию 2] */
    cmul(s1r, s1i, w1r, w1i, y2r, y2i, vl);

    /* y3 = s2·W3 → quarter 3 (pd) */
    cmul(s2r, s2i, w3r, w3i, y3r, y3i, vl);
}

/* ---------------------------------------------------------------------------
 * butterfly_core_dif  —  DIF radix-2 butterfly
 *   x1 = a + b           (сумма, нет twiddle)
 *   x2 = (a − b)·W      (разность, умноженная на twiddle)
 * --------------------------------------------------------------------------*/
static inline void butterfly_core_dif(
    vfloat32m1_t ra, vfloat32m1_t ia,
    vfloat32m1_t rb, vfloat32m1_t ib,
    vfloat32m1_t rw, vfloat32m1_t iw,
    vfloat32m1_t* x1r, vfloat32m1_t* x1i,
    vfloat32m1_t* x2r, vfloat32m1_t* x2i,
    size_t vl)
{
    *x1r = __riscv_vfadd_vv_f32m1(ra, rb, vl);
    *x1i = __riscv_vfadd_vv_f32m1(ia, ib, vl);

    vfloat32m1_t dr = __riscv_vfsub_vv_f32m1(ra, rb, vl);
    vfloat32m1_t di = __riscv_vfsub_vv_f32m1(ia, ib, vl);

    cmul(dr, di, rw, iw, x2r, x2i, vl);
}

/* ===========================================================================
 * r4_stage  —  одна radix-4 DIF стадия
 *
 * Параметры:
 *   vec      — массив данных (in-place)
 *   N        — полный размер FFT
 *   grp_size — размер группы для этой стадии (= 4·quarter)
 *   tw_base  — указатель на блок twiddle-таблицы для этой стадии
 *   vlmax    — vsetvlmax_e32m1
 *
 * Внутренний цикл по j:
 *   [PIPELINE] grp_half ≥ 2·vlmax
 *   [NORMAL]   полные vlmax-блоки
 *   [TAIL]     остаток < vlmax
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

        /* Четыре read-pointer и четыре write-pointer.
         * Начальные позиции совпадают — операция in-place. */
        const float* pa = base;
        const float* pb = base + 2 * quarter;
        const float* pc = base + 4 * quarter;
        const float* pd = base + 6 * quarter;

        float* sa = base;
        float* sb = base + 2 * quarter;
        float* sc = base + 4 * quarter;
        float* sd = base + 6 * quarter;

        int j = 0;

        /* =================================================================
         * PIPELINE-СЕКЦИЯ
         * Активна когда quarter ≥ 2·vlmax.
         * Схема: preload[j] → compute[j−vl] → store[j−vl]
         * ================================================================*/
        if (quarter >= 2 * (int)vlmax)
        {
            const size_t vl = vlmax;

            vfloat32m1_t a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i;
            vfloat32m1_t w1r,w1i, w2r,w2i, w3r,w3i;

            /* Prolog: загрузить j=0 */
            load_data(pa, pb, j, vl, &a0r,&a0i, &a1r,&a1i);
            load_data(pc, pd, j, vl, &a2r,&a2i, &a3r,&a3i);
            load_twiddles_r4(tw_base, j, quarter, vl,
                             &w1r,&w1i, &w2r,&w2i, &w3r,&w3i);

            j += (int)vl;

            /* Kernel */
            for (; j + (int)vl <= quarter; j += (int)vl)
            {
                vfloat32m1_t na0r,na0i, na1r,na1i, na2r,na2i, na3r,na3i;
                vfloat32m1_t nw1r,nw1i, nw2r,nw2i, nw3r,nw3i;

                /* Загружаем следующий блок */
                load_data(pa, pb, j, vl, &na0r,&na0i, &na1r,&na1i);
                load_data(pc, pd, j, vl, &na2r,&na2i, &na3r,&na3i);
                load_twiddles_r4(tw_base, j, quarter, vl,
                                 &nw1r,&nw1i, &nw2r,&nw2i, &nw3r,&nw3i);

                /* Вычисляем и сохраняем текущий блок */
                vfloat32m1_t y0r,y0i, y1r,y1i, y2r,y2i, y3r,y3i;
                butterfly_r4_dif(
                    a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i,
                    w1r,w1i, w2r,w2i, w3r,w3i,
                    &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, &y3r,&y3i, vl);

                store_data(sa, sb, j-(int)vl, y0r,y0i, y1r,y1i, vl);
                store_data(sc, sd, j-(int)vl, y2r,y2i, y3r,y3i, vl);

                /* Сдвигаем конвейер */
                a0r=na0r; a0i=na0i; a1r=na1r; a1i=na1i;
                a2r=na2r; a2i=na2i; a3r=na3r; a3i=na3i;

                w1r=nw1r; w1i=nw1i; 
                w2r=nw2r; w2i=nw2i;
                w3r=nw3r; w3i=nw3i;
            }

            /* Drain */
            {
                vfloat32m1_t y0r,y0i, y1r,y1i, y2r,y2i, y3r,y3i;
                butterfly_r4_dif(
                    a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i,
                    w1r,w1i, w2r,w2i, w3r,w3i,
                    &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, &y3r,&y3i, vl);
                store_data(sa, sb, j-(int)vl, y0r,y0i, y1r,y1i, vl);
                store_data(sc, sd, j-(int)vl, y2r,y2i, y3r,y3i, vl);
            }
        }

        /* =================================================================
         * NORMAL-СЕКЦИЯ  —  полные vlmax-блоки без pipeline
         * ================================================================*/
        for (; j + (int)vlmax <= quarter; j += (int)vlmax)
        {
            const size_t vl = vlmax;

            vfloat32m1_t a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i;
            load_data(pa, pb, j, vl, &a0r,&a0i, &a1r,&a1i);
            load_data(pc, pd, j, vl, &a2r,&a2i, &a3r,&a3i);

            vfloat32m1_t w1r,w1i, w2r,w2i, w3r,w3i;
            load_twiddles_r4(tw_base, j, quarter, vl,
                             &w1r,&w1i, &w2r,&w2i, &w3r,&w3i);

            vfloat32m1_t y0r,y0i, y1r,y1i, y2r,y2i, y3r,y3i;
            butterfly_r4_dif(
                a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i,
                w1r,w1i, w2r,w2i, w3r,w3i,
                &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, &y3r,&y3i, vl);

            store_data(sa, sb, j, y0r,y0i, y1r,y1i, vl);
            store_data(sc, sd, j, y2r,y2i, y3r,y3i, vl);
        }

        /* =================================================================
         * TAIL-СЕКЦИЯ  —  остаток < vlmax элементов
         * ================================================================*/
        if (j < quarter)
        {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(quarter - j));

            vfloat32m1_t a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i;
            load_data(pa, pb, j, vl, &a0r,&a0i, &a1r,&a1i);
            load_data(pc, pd, j, vl, &a2r,&a2i, &a3r,&a3i);

            vfloat32m1_t w1r,w1i, w2r,w2i, w3r,w3i;
            load_twiddles_r4(tw_base, j, quarter, vl,
                             &w1r,&w1i, &w2r,&w2i, &w3r,&w3i);

            vfloat32m1_t y0r,y0i, y1r,y1i, y2r,y2i, y3r,y3i;
            butterfly_r4_dif(
                a0r,a0i, a1r,a1i, a2r,a2i, a3r,a3i,
                w1r,w1i, w2r,w2i, w3r,w3i,
                &y0r,&y0i, &y1r,&y1i, &y2r,&y2i, &y3r,&y3i, vl);

            store_data(sa, sb, j, y0r,y0i, y1r,y1i, vl);
            store_data(sc, sd, j, y2r,y2i, y3r,y3i, vl);
        }
    }
}

/* ===========================================================================
 * r2_stage  —  одна radix-2 DIF стадия с software pipeline
 * ===========================================================================*/
static void r2_stage(
    float* restrict vec,
    int N, int grp_size,
    const float* restrict tw_stage,
    size_t vlmax)
{
    const int grp_half = grp_size >> 1;

    for (int grp_start = 0; grp_start < N; grp_start += grp_size)
    {
        float* restrict pa = &vec[2 *  grp_start];
        float* restrict pb = &vec[2 * (grp_start + grp_half)];

        int j = 0;
        const float* tw_cursor = tw_stage;

        /* Pipeline */
        if (grp_half >= 2 * (int)vlmax)
        {
            const size_t vl = vlmax;
            vfloat32m1_t ra,ia, rb,ib, rw,iw;

            load_data(pa, pb, j, vl, &ra,&ia, &rb,&ib);
            load_twiddles_stream(&tw_cursor, vl, &rw, &iw);
            j += (int)vl;

            for (; j + (int)vl <= grp_half; j += (int)vl)
            {
                vfloat32m1_t ra_n,ia_n, rb_n,ib_n, rw_n,iw_n;
                load_data(pa, pb, j, vl, &ra_n,&ia_n, &rb_n,&ib_n);
                load_twiddles_stream(&tw_cursor, vl, &rw_n, &iw_n);

                vfloat32m1_t x1r,x1i, x2r,x2i;
                butterfly_core_dif(ra,ia, rb,ib, rw,iw,
                                   &x1r,&x1i, &x2r,&x2i, vl);
                store_data(pa, pb, j-(int)vl, x1r,x1i, x2r,x2i, vl);

                ra=ra_n; ia=ia_n; rb=rb_n; ib=ib_n;
                rw=rw_n; iw=iw_n;
            }

            /* Drain */
            vfloat32m1_t x1r,x1i, x2r,x2i;
            butterfly_core_dif(ra,ia, rb,ib, rw,iw,
                               &x1r,&x1i, &x2r,&x2i, vl);
            store_data(pa, pb, j-(int)vl, x1r,x1i, x2r,x2i, vl);
        }

        /* Normal */
        for (; j + (int)vlmax <= grp_half; j += (int)vlmax)
        {
            const size_t vl = vlmax;
            vfloat32m1_t ra,ia, rb,ib, rw,iw, x1r,x1i, x2r,x2i;

            load_data(pa, pb, j, vl, &ra,&ia, &rb,&ib);
            load_twiddles_stream(&tw_cursor, vl, &rw, &iw);
            butterfly_core_dif(ra,ia, rb,ib, rw,iw,
                               &x1r,&x1i, &x2r,&x2i, vl);
            store_data(pa, pb, j, x1r,x1i, x2r,x2i, vl);
        }

        /* Tail */
        if (j < grp_half)
        {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(grp_half - j));
            vfloat32m1_t ra,ia, rb,ib, rw,iw, x1r,x1i, x2r,x2i;

            load_data(pa, pb, j, vl, &ra,&ia, &rb,&ib);
            load_twiddles_stream(&tw_cursor, vl, &rw, &iw);
            butterfly_core_dif(ra,ia, rb,ib, rw,iw,
                               &x1r,&x1i, &x2r,&x2i, vl);
            store_data(pa, pb, j, x1r,x1i, x2r,x2i, vl);
        }
    }
}

/* ===========================================================================
 * rvv_fft_dif
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

    /* =========================================================================
     * Если log2N нечётный: одна r2-стадия ПЕРВОЙ (grp_size = N).
     *
     * Почему первой: в DIF стадии идут от крупных групп к мелким.
     * При нечётном log2N вершина дерева — r2, всё остальное — r4.
     * tw_offsets[log2N-1] указывает на W[0..N/2−1] для grp_size=N.
     * ========================================================================*/
    if (log2N & 1)
    {
        const float* tw_stage = tw2 + 2 * off2[log2N - 1];
        r2_stage(vec, N, N, tw_stage, vlmax);
    }

    /* =========================================================================
     * Radix-4 стадии: от наибольшего grp_size к наименьшему.
     *
     * В DIT-таблице r4_idx=0 → grp_size=4, r4_idx=stages_r4-1 → наибольший.
     * DIF обходит в обратном порядке: stages_r4-1 → 0.
     *
     * Для каждого r4_idx:
     *   grp_size = 1 << (2 + 2*r4_idx)
     *   tw_base  = tw4 + 2*off4[r4_idx]   (тот же блок что и в DIT)
     * ========================================================================*/
    for (int r4_idx = stages_r4 - 1; r4_idx >= 0; r4_idx--)
    {
        const int   grp_size = 1 << (2 + 2*r4_idx);
        const float* tw_base = tw4 + 2 * off4[r4_idx];

        r4_stage(vec, N, grp_size, tw_base, vlmax);
    }

    /* =========================================================================
     * Bit-reversal на выходе.
     * DIF оставляет результат в bit-reversed порядке.
     * bit_reverse_permutation выполняет in-place перестановку за O(N).
     * ========================================================================*/
    bit_reverse(vec, N);

    return vec;
}