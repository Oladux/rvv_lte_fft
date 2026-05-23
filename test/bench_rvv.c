
 
#include <stdint.h>
#include <math.h>
 
#include "../include/ofdm_fft.h"
 
 
#define BENCH_WARMUP   3
#define BENCH_REPEAT  11
 
static const int LTE_SIZES[] = { 128, 256, 512, 1024, 1536, 2048 };
#define N_SIZES ((int)(sizeof(LTE_SIZES) / sizeof(LTE_SIZES[0])))
 
#define MAX_N      2048
#define RVV_ALIGN    64
 
 
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
    for (int w = 0; w < BENCH_WARMUP; w++)
    {
        fill_random(N);
        ofdm_fft(vec, N);
    }
 
    uint64_t cyc_buf[BENCH_REPEAT];
    uint64_t ins_buf[BENCH_REPEAT];
 
    for (int r = 0; r < BENCH_REPEAT; r++)
    {
        fill_random(N);
        __asm__ volatile ("fence" ::: "memory");
        uint64_t c0 = csr_mcycle();
        uint64_t i0 = csr_minstret();
        __asm__ volatile ("fence" ::: "memory");
 
        ofdm_fft(vec, N);
 
        __asm__ volatile ("fence" ::: "memory");
        cyc_buf[r] = csr_mcycle()   - c0;
        ins_buf[r] = csr_minstret() - i0;
        __asm__ volatile ("fence" ::: "memory");
    }
 
    sort_u64(cyc_buf, BENCH_REPEAT);
    sort_u64(ins_buf, BENCH_REPEAT);
 
    g_results[idx].N          = (uint64_t)N;
    g_results[idx].cycles_min = cyc_buf[0];
    g_results[idx].instrs_min = ins_buf[0];
    g_results[idx].cycles_med = cyc_buf[BENCH_REPEAT / 2];
    g_results[idx].instrs_med = ins_buf[BENCH_REPEAT / 2];
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
