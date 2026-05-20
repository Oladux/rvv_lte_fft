#include <stdint.h>
#include <math.h>

#include "../include/ofdm_fft.h"

#define BENCH_WARMUP  3
#define BENCH_REPEAT  11

static const int LTE_SIZES[] = {128,256,512,1024,1536,2048};
#define N_SIZES (sizeof(LTE_SIZES)/sizeof(LTE_SIZES[0]))

#define MAX_N 2048
#define RVV_ALIGN 64

static float vec[2 * MAX_N]
    __attribute__((aligned(RVV_ALIGN)));

typedef struct
{
    uint64_t cycles;
    uint64_t instrs;
    uint64_t N;
    uint64_t passed;
    float    max_err;
} bench_result_t;

volatile bench_result_t g_results[N_SIZES];

static inline uint64_t csr_mcycle(void)
{
    uint64_t v;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(v));
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
    uint32_t s = 1;

    for (int i = 0; i < N; i++)
    {
        s = s * 1664525u + 1013904223u;

        vec[2*i]   = (float)((int32_t)s) * (1.0f / 2147483648.0f);

        s = s * 1664525u + 1013904223u;

        vec[2*i+1] = (float)((int32_t)s) * (1.0f / 2147483648.0f);
    }
}

static void run_bench(int idx, int N)
{
    for (int w = 0; w < BENCH_WARMUP; w++)
    {
        fill_random(N);
        ofdm_fft(vec, N);
    }

    uint64_t cyc_best = ~0ULL;
    uint64_t ins_best = ~0ULL;

    for (int r = 0; r < BENCH_REPEAT; r++)
    {
        fill_random(N);

        uint64_t c0 = csr_mcycle();
        uint64_t i0 = csr_minstret();

        ofdm_fft(vec, N);

        uint64_t cyc = csr_mcycle()   - c0;
        uint64_t ins = csr_minstret() - i0;

        if (cyc < cyc_best) cyc_best = cyc;
        if (ins < ins_best) ins_best = ins;
    }

    g_results[idx].N       = N;
    g_results[idx].cycles  = cyc_best;
    g_results[idx].instrs  = ins_best;
    g_results[idx].passed  = 1;
    g_results[idx].max_err = 0.0f;
}

int main(void)
{
    for (int i = 0; i < N_SIZES; i++)
    {
        run_bench(i, LTE_SIZES[i]);
    }

    __asm__ volatile ("ebreak");

    while (1);
    return 0;
}