#include "../include/ofdm_fft.h"

/* ============================================================================
 * cpx_copy
 *
 * Copy of interleaved complex data
 * 
 * Vectorized copy of interleaved complex float32:
 * 
 *   dst destination pointer - input
 *   src        source pointer - input/output
 *   n_complex  number of complex samples - input
 * ==========================================================================*/
static inline void cpx_copy(
    float* restrict dst,
    const float* restrict src,
    size_t n_complex){

    size_t i = 0;

    while (i < n_complex)
    {
        size_t vl =  __riscv_vsetvl_e32m1(n_complex - i);

        vfloat32m1x2_t v = __riscv_vlseg2e32_v_f32m1x2(
                src + 2 * i,
                vl);

        __riscv_vsseg2e32_v_f32m1x2(
            dst + 2 * i,
            v,
            vl);

        i += vl;
    }
}

/* ============================================================================
 * ofdm_add_cp
 *
 * Adds OFDM cyclic prefix
 *
 *   dst       output buffer - input/output
 *   src       FFT output buffer - output
 *   fft_size  FFT size - input
 *   cp_len    cyclic prefix length - input
 * ==========================================================================*/
void ofdm_add_cp(
    float* restrict dst,
    const float* restrict src,
    uint32_t fft_size,
    uint32_t cp_len)
{
    cpx_copy(
        dst,
        src + 2 * (fft_size - cp_len),
        cp_len);


    cpx_copy(
        dst + 2 * cp_len,
        src,
        fft_size);
}

/* ============================================================================
 * ofdm_remove_cp
 *
 * Removes cyclic prefix
 *   dst       output FFT input buffer - output 
 *   src       received OFDM symbol with CP - input
 *   fft_size  FFT size - input
 *   cp_len    cyclic prefix length - input
 * ==========================================================================*/
void ofdm_remove_cp(
    float* restrict dst,
    const float* restrict src,
    uint32_t fft_size,
    uint32_t cp_len)
{
    cpx_copy(
        dst,
        src + 2 * cp_len,
        fft_size);
}