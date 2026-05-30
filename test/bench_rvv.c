
 
#include <stdint.h>
#include <math.h>
 
#include "../include/ofdm_fft.h"
#include "fft_scalar.c"
 
#define BENCH_WARMUP   3
#define BENCH_REPEAT  11
 
static const int LTE_SIZES[] = { 128, 256, 512, 1024, 1536, 2048 };
#define N_SIZES ((int)(sizeof(LTE_SIZES) / sizeof(LTE_SIZES[0])))
 

#define MAX_N      2048
#define RVV_ALIGN    64
 
static volatile float g_debug_ref[16];
static volatile float g_debug_rvv[16];


static int ilog2i(int x) {
    int r = 0;
    while (x >>= 1) r++;
    return r;
}
 
static float vec[2 * MAX_N] __attribute__((aligned(RVV_ALIGN)));
 
typedef struct
{
    uint64_t N;
    uint64_t cycles_min;
    uint64_t instrs_min;
    uint64_t cycles_med;
    uint64_t instrs_med;
    uint64_t done;     
} bench_result_t;
 

volatile bench_result_t g_results[N_SIZES];

 
static inline uint64_t csr_mcycle(void)
{
    uint64_t v;
    __asm__ volatile ("csrr %0, mcycle"   : "=r"(v));
    return v;
}
 
static inline uint64_t csr_minstret(void)
{
    uint64_t v;
    __asm__ volatile ("csrr %0, minstret" : "=r"(v));
    return v;
}
 
static void fill_random(int N)
{
    uint32_t s = 0xABCD1234u;
    for (int i = 0; i < N; i++)
    {
        s = s * 1664525u + 1013904223u;
        vec[2*i]   = (float)(int32_t)s * (1.0f / 2147483648.0f);
        s = s * 1664525u + 1013904223u;
        vec[2*i+1] = (float)(int32_t)s * (1.0f / 2147483648.0f);
    }
}

typedef struct {
    double max_abs_error;
    double max_rel_error;
    double snr_db;
} error_metrics_t;


error_metrics_t g_errors[N_SIZES];

static error_metrics_t compare_fft_interleaved(
    float* ref_re,
    float* ref_im,
    float* vec,
    int N)
{
    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_sq_err = 0.0;
    double sum_sq_sig = 0.0;

    for (int i = 0; i < N; i++)
    {
        double re_ref = ref_re[i];
        double im_ref = ref_im[i];

        double re_test = vec[2*i];
        double im_test = vec[2*i + 1];

        double abs_err = hypot(
            re_test - re_ref,
            im_test - im_ref
        );

        double abs_sig = hypot(re_ref, im_ref);

        if (abs_err > max_abs)
            max_abs = abs_err;

        if (abs_sig > 1e-12)
        {
            double rel_err = abs_err / abs_sig;

            if (rel_err > max_rel)
                max_rel = rel_err;
        }

        sum_sq_err += abs_err * abs_err;
        sum_sq_sig += abs_sig * abs_sig;
    }

    double snr =
        10.0 * log10(sum_sq_sig / (sum_sq_err + 1e-12));

    error_metrics_t m = {
        .max_abs_error = max_abs,
        .max_rel_error = max_rel,
        .snr_db = snr
    };

    return m;
}
 
static void sort_u64(uint64_t* a, int n)
{
    for (int i = 1; i < n; i++)
    {
        uint64_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j+1] = a[j]; j--; }
        a[j+1] = key;
    }
}
 
static void run_bench(int idx, int N)
{
    int LogN = ilog2i(N); 


    static float ref_re[MAX_N] __attribute__((aligned(RVV_ALIGN)));
    static float ref_im[MAX_N] __attribute__((aligned(RVV_ALIGN)));


    for (int w = 0; w < BENCH_WARMUP; w++) {
        fill_random(N);
        // Копируем входные данные в reference
        for (int i = 0; i < N; i++) {
            ref_re[i] = vec[2*i];
            ref_im[i] = vec[2*i+1];
        }
        FFT(ref_re, ref_im, N, LogN, FT_DIRECT); 
        ofdm_fft(vec, N);                    
    }
    uint64_t cyc_buf[BENCH_REPEAT];
    uint64_t ins_buf[BENCH_REPEAT];
    error_metrics_t err_buf[BENCH_REPEAT];

    for (int r = 0; r < BENCH_REPEAT; r++) {
        fill_random(N);

        for (int i = 0; i < N; i++) {
            ref_re[i] = vec[2*i];
            ref_im[i] = vec[2*i+1];
        }

        __asm__ volatile ("fence" ::: "memory");
        uint64_t c0 = csr_mcycle();
        uint64_t i0 = csr_minstret();
        __asm__ volatile ("fence" ::: "memory");

        ofdm_fft(vec, N);

        __asm__ volatile ("fence" ::: "memory");
        cyc_buf[r] = csr_mcycle() - c0;
        ins_buf[r] = csr_minstret() - i0;
        __asm__ volatile ("fence" ::: "memory");
        FFT(ref_re, ref_im, N, LogN, FT_DIRECT);


        for (int i = 0; i < 8; i++)
        {
            g_debug_ref[2*i]     = ref_re[i];
            g_debug_ref[2*i +1 ] = ref_im[i];

            g_debug_rvv[2*i]     = vec[2*i];
            g_debug_rvv[2*i +1 ] = vec[2*i +1];
        }

        // Сравниваем результаты
     err_buf[r] =
    compare_fft_interleaved(
        ref_re,
        ref_im,
        vec,
        N
    );
    }

    sort_u64(cyc_buf, BENCH_REPEAT);
    sort_u64(ins_buf, BENCH_REPEAT);

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double snr_min = 1e9;
    for (int r = 0; r < BENCH_REPEAT; r++) {
        if (err_buf[r].max_abs_error > max_abs_err)
            max_abs_err = err_buf[r].max_abs_error;
        if (err_buf[r].max_rel_error > max_rel_err)
            max_rel_err = err_buf[r].max_rel_error;
        if (err_buf[r].snr_db < snr_min)
            snr_min = err_buf[r].snr_db;
    }

    g_results[idx].N          = N;
    g_results[idx].cycles_min = cyc_buf[0];
    g_results[idx].instrs_min = ins_buf[0];
    g_results[idx].cycles_med = cyc_buf[BENCH_REPEAT / 2];
    g_results[idx].instrs_med = ins_buf[BENCH_REPEAT / 2];
    g_errors[idx].max_abs_error = max_abs_err;
    g_errors[idx].max_rel_error = max_rel_err;
    g_results[idx].done       = 1;

}
int main(void)
{
    for (int i = 0; i < N_SIZES; i++)
        g_results[i].done = 0;
 

    for (int i = 0; i < N_SIZES; i++)
        run_bench(i, LTE_SIZES[i]);

    __asm__ volatile ("ebreak");
    while (1);
 
    return 0;
}
