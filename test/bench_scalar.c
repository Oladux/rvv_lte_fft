#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "fft_scalar.c" 

// Прототип вашей скалярной FFT
bool FFT(float *Rdat, float *Idat, int N, int LogN, int Ft_Flag);
 
#define FT_DIRECT  -1
#define FT_INVERSE  1
 
/* ── Настройки ────────────────────────────────────────────────────────── */
#define BENCH_WARMUP   3
#define BENCH_REPEAT  11
 
static const int LTE_SIZES[] = { 128, 256, 512, 1024, 2048 };
#define N_SIZES ((int)(sizeof(LTE_SIZES) / sizeof(LTE_SIZES[0])))
 
#define MAX_N      2048
#define ALIGN      64
 
/* ── Буферы ───────────────────────────────────────────────────────────── */
static float Rdat[MAX_N] __attribute__((aligned(ALIGN)));
static float Idat[MAX_N] __attribute__((aligned(ALIGN)));
 
/* ── Таблица Log2 ─────────────────────────────────────────────────────── */
static int get_log2(int N) {
    int log = 0;
    while (N > 1) { N >>= 1; log++; }
    return log;
}
 
/* ── Структура результата ─────────────────────────────────────────────── */
typedef struct {
    uint64_t N;
    uint64_t cycles_min;
    uint64_t instrs_min;
    uint64_t cycles_med;
    uint64_t instrs_med;
    uint64_t done;
} bench_result_t;
 
volatile bench_result_t g_results[N_SIZES];
volatile double g_errors[N_SIZES][3];  // Заглушка для GDB
 
/* ── CSR счётчики ─────────────────────────────────────────────────────── */
static inline uint64_t csr_mcycle(void) {
    uint64_t v;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(v));
    return v;
}
 
static inline uint64_t csr_minstret(void) {
    uint64_t v;
    __asm__ volatile ("csrr %0, minstret" : "=r"(v));
    return v;
}
 
/* ── Генератор псевдослучайного входа ────────────────────────────────── */
static void fill_random(int N) {
    uint32_t s = 0xABCD1234u;
    for (int i = 0; i < N; i++) {
        s = s * 1664525u + 1013904223u;
        Rdat[i] = (float)(int32_t)s * (1.0f / 2147483648.0f);
        s = s * 1664525u + 1013904223u;
        Idat[i] = (float)(int32_t)s * (1.0f / 2147483648.0f);
    }
}
 
/* ── Сортировка для медианы ───────────────────────────────────────────── */
static void sort_u64(uint64_t* a, int n) {
    for (int i = 1; i < n; i++) {
        uint64_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j+1] = a[j]; j--; }
        a[j+1] = key;
    }
}
 
/* ── Обёртка для FFT с interleaved → split форматом ──────────────────── */
static void fft_wrapper(float* vec, int N) {
    // Конвертация interleaved [re0,im0,re1,im1,...] → split [Rdat, Idat]
    for (int i = 0; i < N; i++) {
        Rdat[i] = vec[2*i];
        Idat[i] = vec[2*i+1];
    }
    
    int LogN = get_log2(N);
    FFT(Rdat, Idat, N, LogN, FT_DIRECT);
    
    // Конвертация обратно
    for (int i = 0; i < N; i++) {
        vec[2*i]   = Rdat[i];
        vec[2*i+1] = Idat[i];
    }
}
 
/* ── Бенчмарк одного размера ─────────────────────────────────────────── */
static void run_bench(int idx, int N) {
    // Прогрев
    for (int w = 0; w < BENCH_WARMUP; w++) {
        fill_random(N);
        fft_wrapper((float*)Rdat, N);  // Используем Rdat как временный буфер
    }
 
    uint64_t cyc_buf[BENCH_REPEAT];
    uint64_t ins_buf[BENCH_REPEAT];
    
    // Временный interleaved буфер на стеке (осторожно с размером!)
    float vec[2 * MAX_N] __attribute__((aligned(ALIGN)));
 
    for (int r = 0; r < BENCH_REPEAT; r++) {
        fill_random(N);
        
        // Копируем в interleaved формат
        for (int i = 0; i < N; i++) {
            vec[2*i]   = Rdat[i];
            vec[2*i+1] = Idat[i];
        }
 
        __asm__ volatile ("fence" ::: "memory");
        uint64_t c0 = csr_mcycle();
        uint64_t i0 = csr_minstret();
        __asm__ volatile ("fence" ::: "memory");
 
        fft_wrapper(vec, N);
 
        __asm__ volatile ("fence" ::: "memory");
        cyc_buf[r] = csr_mcycle() - c0;
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
    
    g_errors[idx][0] = 0.0;
    g_errors[idx][1] = 0.0;
    g_errors[idx][2] = 0.0;
}
 
/* ── main ─────────────────────────────────────────────────────────────── */
int main(void) {
    for (int i = 0; i < N_SIZES; i++) {
        g_results[i].done = 0;
        g_errors[i][0] = 0.0;
        g_errors[i][1] = 0.0;
        g_errors[i][2] = 0.0;
    }
 
    for (int i = 0; i < N_SIZES; i++) {
        run_bench(i, LTE_SIZES[i]);
    }
 
    __asm__ volatile ("ebreak");
    while (1);
 
    return 0;
}