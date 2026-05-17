#include "../include/rvv_fft.h"

/* ---------------------------------------------------------------------------
 * interleave — interleave two separate real and imag arrays into one array
 *
 *   re      real part source array — input
 *   im      imaginary part source array — input
 *   data    destination interleaved array (re, im, re, im, ...) — output
 *   N       number of complex elements — input
 * --------------------------------------------------------------------------*/
void interleave(float *re, float *im, float *data, int32_t N) {
    int32_t i = 0;
 
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i); // seting vector len
 
        vfloat32m1_t v_re = __riscv_vle32_v_f32m1(&re[i], vl); // load Re and Im data to vec regs
        vfloat32m1_t v_im = __riscv_vle32_v_f32m1(&im[i], vl);

        vfloat32m1x2_t v_pair = __riscv_vcreate_v_f32m1x2(v_re, v_im); // form a Re and Im pair 
        
        __riscv_vsseg2e32_v_f32m1x2(&data[2 * i], v_pair, vl); // segmented load to interlieved format
 
        i += vl;
    }
}

/* ---------------------------------------------------------------------------
 * bit_reverse — bit-reverse permutation for FFT output ordering
 *
 *   vec     interleaved complex data array (in-place) — input/output
 *   N       FFT size (power of two) — input
 *
 *   Uses precomputed bit-reversal table from get_bitrev_table()
 *   Swaps elements only once (i < j condition prevents double swap)
 * --------------------------------------------------------------------------*/
void bit_reverse(float *vec, int32_t N) {
    int32_t count;
    const int32_t* bitrev = get_bitrev_table(N, &count);

    for (int32_t i = 0; i < N; i++)
    {
        int32_t j = bitrev[i];

        if (i < j)
        {
            int32_t i2 = 2 * i;
            int32_t j2 = 2 * j;

            float tmp_r = vec[i2]; // swap real and imaginary parts
            float tmp_i = vec[i2 + 1];

            vec[i2]     = vec[j2];
            vec[i2 + 1] = vec[j2 + 1];

            vec[j2]     = tmp_r;
            vec[j2 + 1] = tmp_i;
        }
    }
}

/* ---------------------------------------------------------------------------
 * prepare_fft_vector — prepare FFT input by interleaving real and imag arrays
 *
 *   Rvec    real part source array — input
 *   Ivec    imaginary part source array — input
 *   vec     destination interleaved array (re, im, re, im, ...) — output
 *   N       number of complex elements — input
 * --------------------------------------------------------------------------*/
void prepare_fft_vector(float *Rvec, float *Ivec, float *vec, int32_t N) {
    interleave(Rvec, Ivec, vec, N);
}

/* ---------------------------------------------------------------------------
 * cpx_cmul — complex multiplication: out = (ar + i·ai) × (br + i·bi)
 *
 *   ar, ai   first operand (real, imag) — input
 *   br, bi   second operand (real, imag) — input
 *   outr, outi result pointers (real, imag) — output
 *   vl       vector length — input
 * --------------------------------------------------------------------------*/
void cpx_cmul(
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

