#include "../include/rvv_fft.h"
#include "../include/rvv_fft_misc.h"
 
/* ---------------------------------------------------------------------------
 * load_data
 *   Загружает два интерливленных комплексных вектора из pa[2j..] и pb[2j..]
 *   Формат данных: [Re0, Im0, Re1, Im1, ...]  -> vlseg2 разделяет Re / Im
 * --------------------------------------------------------------------------*/
static inline void load_data(
    const float* restrict pa,
    const float* restrict pb,
    int          j,
    size_t       vl,
    vfloat32m1_t* ra,
    vfloat32m1_t* ia,
    vfloat32m1_t* rb,
    vfloat32m1_t* ib)
{
    vfloat32m1x2_t va = __riscv_vlseg2e32_v_f32m1x2(pa + 2*j, vl);
    vfloat32m1x2_t vb = __riscv_vlseg2e32_v_f32m1x2(pb + 2*j, vl);
 
    *ra = __riscv_vget_v_f32m1x2_f32m1(va, 0);
    *ia = __riscv_vget_v_f32m1x2_f32m1(va, 1);
    *rb = __riscv_vget_v_f32m1x2_f32m1(vb, 0);
    *ib = __riscv_vget_v_f32m1x2_f32m1(vb, 1);
}
 
/* ---------------------------------------------------------------------------
 * load_twiddles_stream
 *   Читает vl твидл-факторов последовательно из *tw_cursor и двигает курсор.
 *   Формат таблицы: [Re0, Im0, Re1, Im1, ...]
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
 * butterfly_core
 *   DIT radix-2 butterfly с комплексным умножением через FMA:
 *     m  = rb * rw - ib * iw   (Re часть произведения)
 *     mi = ib * rw + rb * iw   (Im часть произведения)
 *     x1 = a + m,  x2 = a - m
 * --------------------------------------------------------------------------*/
static inline void butterfly_core(
    vfloat32m1_t ra, vfloat32m1_t ia,
    vfloat32m1_t rb, vfloat32m1_t ib,
    vfloat32m1_t rw, vfloat32m1_t iw,
    vfloat32m1_t* x1r, vfloat32m1_t* x1i,
    vfloat32m1_t* x2r, vfloat32m1_t* x2i,
    size_t vl)
{
    /* mr = rb*rw - ib*iw */
    vfloat32m1_t mr = __riscv_vfmul_vv_f32m1(rb, rw, vl);
    mr = __riscv_vfnmsac_vv_f32m1(mr, ib, iw, vl);
 
    /* mi = ib*rw + rb*iw */
    vfloat32m1_t mi = __riscv_vfmul_vv_f32m1(ib, rw, vl);
    mi = __riscv_vfmacc_vv_f32m1(mi, rb, iw, vl);
 
    *x1r = __riscv_vfadd_vv_f32m1(ra, mr, vl);
    *x1i = __riscv_vfadd_vv_f32m1(ia, mi, vl);
    *x2r = __riscv_vfsub_vv_f32m1(ra, mr, vl);
    *x2i = __riscv_vfsub_vv_f32m1(ia, mi, vl);
}
 
/* ---------------------------------------------------------------------------
 * store_data
 *   Записывает два комплексных вектора обратно в interleaved формат.
 * --------------------------------------------------------------------------*/
static inline void store_data(
    float* restrict pa,
    float* restrict pb,
    int    j,
    vfloat32m1_t x1r, vfloat32m1_t x1i,
    vfloat32m1_t x2r, vfloat32m1_t x2i,
    size_t vl)
{
    vfloat32m1x2_t out1 = __riscv_vcreate_v_f32m1x2(x1r, x1i);
    vfloat32m1x2_t out2 = __riscv_vcreate_v_f32m1x2(x2r, x2i);
 
    __riscv_vsseg2e32_v_f32m1x2(pa + 2*j, out1, vl);
    __riscv_vsseg2e32_v_f32m1x2(pb + 2*j, out2, vl);
}
 
/* ---------------------------------------------------------------------------
 * rvv_fft
 *   In-place Cooley-Tukey DIT FFT, radix-2.
 *   vec  — указатель на float-массив длиной 2*N (interleaved Re/Im).
 *   N    — размер БПФ, степень двойки.
 *
 *   Структура внутреннего цикла по j:
 *     [PIPELINE] — software-pipeline с preload+compute+store
 *                  активен когда grp_half >= 2*vlmax (есть смысл прогревать)
 *     [NORMAL]   — обычный векторный цикл для оставшихся полных векторов
 *     [TAIL]     — хвостовой цикл с vsetvl для остатка < vlmax
 *
 *   Все три секции не пересекаются: j монотонно растёт через все три.
 * --------------------------------------------------------------------------*/
float* rvv_fft(float* restrict vec, int N)
{
    /* FIX: убрана объявление «size_t vl» на уровне функции.
     * В оригинале она никогда не инициализировалась и сразу же
     * перекрывалась (shadow) локальными vl внутри каждого блока,
     * создавая потенциал для UB при любом рефакторинге.
     * Каждая секция объявляет свой vl в собственном блоке. */
 
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
 
    int log2N;
    const float* tw_table;
    const int*   tw_offsets;
 
    get_twiddle(N, &tw_table, &tw_offsets, &log2N);
 
    for (int stage = 1; stage <= log2N; stage++)
    {
        const int grp_size = 1 << stage;
        const int grp_half = grp_size >> 1;
 
        /* Начало блока твидл-факторов для данной стадии.
         * tw_cursor сбрасывается на начало блока для каждой группы. */
        const float* restrict tw_stage = tw_table + 2 * tw_offsets[stage - 1];
 
        for (int grp_start = 0; grp_start < N; grp_start += grp_size)
        {
            float* restrict pa = &vec[2 * grp_start];
            float* restrict pb = &vec[2 * (grp_start + grp_half)];
 
            int j = 0;
            const float* tw_cursor = tw_stage;
 
            /* =============================================================
             * PIPELINE-СЕКЦИЯ
             * Используется когда группа достаточно большая, чтобы
             * скрыть задержку загрузки: grp_half >= 2*vlmax гарантирует
             * минимум одну итерацию pipeline-цикла + drain.
             *
             * Схема: preload[j] -> compute[j-vl] -> store[j-vl]
             *   Шаг 0: загрузить j=0, j+=vl
             *   Цикл: загрузить j (следующий), вычислить и сохранить j-vl (текущий)
             *   Drain: вычислить и сохранить последний загруженный блок
             * ============================================================*/
            if (grp_half >= 2 * (int)vlmax)
            {
                /* FIX: vl объявлена внутри if-блока, а не на уровне функции.
                 * Область видимости строго ограничена этой секцией. */
                const size_t vl = vlmax;
 
                vfloat32m1_t ra, ia, rb, ib, rw, iw;
 
                /* Шаг 0: загрузить первый блок */
                load_data(pa, pb, j, vl, &ra, &ia, &rb, &ib);
                load_twiddles_stream(&tw_cursor, vl, &rw, &iw);
                j += (int)vl;
 
                /* Pipeline-цикл: на каждой итерации обрабатываем предыдущий
                 * блок пока следующий блок уже в регистрах. */
                for (; j + (int)vl <= grp_half; j += (int)vl)
                {
                    vfloat32m1_t ra_next, ia_next, rb_next, ib_next;
                    vfloat32m1_t rw_next, iw_next;
 
                    /* Загружаем следующий блок */
                    load_data(pa, pb, j, vl,
                              &ra_next, &ia_next, &rb_next, &ib_next);
                    load_twiddles_stream(&tw_cursor, vl, &rw_next, &iw_next);
 
                    /* FIX: убран вызов prefetch_data().
                     * Оригинал: (void)vlseg2e32(...) — dead load, компилятор
                     * удалял целиком. На SPIKE кэш не моделируется.
                     * Для реального железа: __builtin_prefetch(pa + 2*(j+vl), 0, 1)
                     *                       __builtin_prefetch(pb + 2*(j+vl), 0, 1) */
 
                    /* Вычисляем и сохраняем текущий (предыдущий) блок */
                    vfloat32m1_t x1r, x1i, x2r, x2i;
                    butterfly_core(ra, ia, rb, ib, rw, iw,
                                   &x1r, &x1i, &x2r, &x2i, vl);
                    store_data(pa, pb, j - (int)vl, x1r, x1i, x2r, x2i, vl);
 
                    /* Сдвигаем регистровый конвейер */
                    ra = ra_next; ia = ia_next;
                    rb = rb_next; ib = ib_next;
                    rw = rw_next; iw = iw_next;
                }
 
                /* Drain: вычисляем последний загруженный блок.
                 * Позиция: j - vl (последний j из цикла перед j+=vl). */
                {
                    vfloat32m1_t x1r, x1i, x2r, x2i;
                    butterfly_core(ra, ia, rb, ib, rw, iw,
                                   &x1r, &x1i, &x2r, &x2i, vl);
                    store_data(pa, pb, j - (int)vl, x1r, x1i, x2r, x2i, vl);
                }
 
                /* j теперь указывает на первый ещё не обработанный элемент */
            }
 
            /* =============================================================
             * NORMAL-СЕКЦИЯ
             * Обрабатывает полные векторы vlmax без pipeline.
             * Активна если после pipeline остались полные векторы,
             * либо если grp_half < 2*vlmax (pipeline не запускался).
             * ============================================================*/
            for (; j + (int)vlmax <= grp_half; j += (int)vlmax)
            {
                const size_t vl = vlmax;
 
                vfloat32m1_t ra, ia, rb, ib, rw, iw;
                vfloat32m1_t x1r, x1i, x2r, x2i;
 
                load_data(pa, pb, j, vl, &ra, &ia, &rb, &ib);
                load_twiddles_stream(&tw_cursor, vl, &rw, &iw);
                butterfly_core(ra, ia, rb, ib, rw, iw,
                               &x1r, &x1i, &x2r, &x2i, vl);
                store_data(pa, pb, j, x1r, x1i, x2r, x2i, vl);
            }
 
            /* =============================================================
             * TAIL-СЕКЦИЯ
             * Обрабатывает хвост [j, grp_half) если grp_half не кратен vlmax.
             * vsetvl вернёт точное количество оставшихся элементов.
             * ============================================================*/
            if (j < grp_half)
            {
                const size_t vl = __riscv_vsetvl_e32m1(
                    (size_t)(grp_half - j));
 
                vfloat32m1_t ra, ia, rb, ib, rw, iw;
                vfloat32m1_t x1r, x1i, x2r, x2i;
 
                load_data(pa, pb, j, vl, &ra, &ia, &rb, &ib);
                load_twiddles_stream(&tw_cursor, vl, &rw, &iw);
                butterfly_core(ra, ia, rb, ib, rw, iw,
                               &x1r, &x1i, &x2r, &x2i, vl);
                store_data(pa, pb, j, x1r, x1i, x2r, x2i, vl);
            }
        }
    }
 
    return vec;
}
