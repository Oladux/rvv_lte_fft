
#include <stdint.h>
#include <stdbool.h>
#include <math.h>


#define NUMBER_IS_2_POW_K(x)   ((!((x)&((x)-1)))&&((x)>1))
#define FT_DIRECT   -1
#define FT_INVERSE   1

bool FFT(float *Rdat, float *Idat, int N, int LogN, int Ft_Flag)
{
    if((Rdat == NULL) || (Idat == NULL))                  return false;
    if((N > 16384) || (N < 1))                            return false;
    if(!NUMBER_IS_2_POW_K(N))                             return false;
    if((LogN < 2) || (LogN > 14))                         return false;
    if((Ft_Flag != FT_DIRECT) && (Ft_Flag != FT_INVERSE)) return false;

    register int  i, j, n, k, io, ie, in, nn;
    float ru, iu, rtp, itp, rtq, itq, rw, iw, sr;

    static const float Rcoef[14] =
    {
        -1.0000000000000000F,  0.0000000000000000F,
         0.7071067811865475F,  0.9238795325112867F,
         0.9807852804032304F,  0.9951847266721969F,
         0.9987954562051724F,  0.9996988186962042F,
         0.9999247018391445F,  0.9999811752826011F,
         0.9999952938095761F,  0.9999988234517018F,
         0.9999997058628822F,  0.9999999264657178F
    };

    static const float Icoef[14] =
    {
         0.0000000000000000F, -1.0000000000000000F,
        -0.7071067811865474F, -0.3826834323650897F,
        -0.1950903220161282F, -0.0980171403295606F,
        -0.0490676743274180F, -0.0245412285229122F,
        -0.0122715382857199F, -0.0061358846491544F,
        -0.0030679567629659F, -0.0015339801862847F,
        -0.0007669903187427F, -0.0003834951875714F
    };

    nn = N >> 1;
    ie = N;

    for(n = 1; n <= LogN; n++)
    {
        rw = Rcoef[LogN - n];
        iw = Icoef[LogN - n];

        if(Ft_Flag == FT_INVERSE)
            iw = -iw;

        in = ie >> 1;
        ru = 1.0F;
        iu = 0.0F;

        for(j = 0; j < in; j++)
        {
            for(i = j; i < N; i += ie)
            {
                io       = i + in;

                rtp      = Rdat[i]  + Rdat[io];
                itp      = Idat[i]  + Idat[io];

                rtq      = Rdat[i]  - Rdat[io];
                itq      = Idat[i]  - Idat[io];

                Rdat[io] = rtq * ru - itq * iu;
                Idat[io] = itq * ru + rtq * iu;

                Rdat[i]  = rtp;
                Idat[i]  = itp;
            }

            sr = ru;
            ru = ru * rw - iu * iw;
            iu = iu * rw + sr * iw;
        }

        ie >>= 1;
    }

    for(j = i = 1; i < N; i++)
    {
        if(i < j)
        {
            io       = i - 1;
            in       = j - 1;

            rtp      = Rdat[in];
            itp      = Idat[in];

            Rdat[in] = Rdat[io];
            Idat[in] = Idat[io];

            Rdat[io] = rtp;
            Idat[io] = itp;
        }

        k = nn;

        while(k < j)
        {
            j   = j - k;
            k >>= 1;
        }

        j = j + k;
    }

    if(Ft_Flag == FT_DIRECT)
        return true;

    rw = 1.0F / N;

    for(i = 0; i < N; i++)
    {
        Rdat[i] *= rw;
        Idat[i] *= rw;
    }

    return true;
}


#define BENCH_WARMUP   3
#define BENCH_REPEAT  11

static const int FFT_SIZES[] = { 128, 256, 512, 1024, 2048 };
#define N_SIZES ((int)(sizeof(FFT_SIZES) / sizeof(FFT_SIZES[0])))

#define MAX_N 2048
#define ALIGN 64

static float Re[MAX_N] __attribute__((aligned(ALIGN)));
static float Im[MAX_N] __attribute__((aligned(ALIGN)));

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
    __asm__ volatile ("csrr %0, mcycle" : "=r"(v));
    return v;
}

static inline uint64_t csr_minstret(void)
{
    uint64_t v;
    __asm__ volatile ("csrr %0, minstret" : "=r"(v));
    return v;
}


static int ilog2i(int x)
{
    int r = 0;
    while (x >>= 1)
        r++;
    return r;
}

static void fill_random(int N)
{
    uint32_t s = 0x12345678u;

    for (int i = 0; i < N; i++)
    {
        s = s * 1664525u + 1013904223u;
        Re[i] = (float)(int32_t)s * (1.0f / 2147483648.0f);

        s = s * 1664525u + 1013904223u;
        Im[i] = (float)(int32_t)s * (1.0f / 2147483648.0f);
    }
}

static void sort_u64(uint64_t* a, int n)
{
    for (int i = 1; i < n; i++)
    {
        uint64_t key = a[i];
        int j = i - 1;

        while (j >= 0 && a[j] > key)
        {
            a[j + 1] = a[j];
            j--;
        }

        a[j + 1] = key;
    }
}



static void run_bench(int idx, int N)
{
    int LogN = ilog2i(N);

    for (int w = 0; w < BENCH_WARMUP; w++)
    {
        fill_random(N);
        FFT(Re, Im, N, LogN, FT_DIRECT);
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

        FFT(Re, Im, N, LogN, FT_DIRECT);

        __asm__ volatile ("fence" ::: "memory");

        cyc_buf[r] = csr_mcycle() - c0;
        ins_buf[r] = csr_minstret() - i0;

        __asm__ volatile ("fence" ::: "memory");
    }

    sort_u64(cyc_buf, BENCH_REPEAT);
    sort_u64(ins_buf, BENCH_REPEAT);

    g_results[idx].N          = N;
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
        run_bench(i, FFT_SIZES[i]);

    __asm__ volatile ("ebreak");

    while (1);

    return 0;
}