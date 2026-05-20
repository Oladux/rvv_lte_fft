#ifndef UTLS_H
#define UTLS_H

extern void bit_reverse(float*, int32_t);
extern void interleave(float*, float*, float*, int32_t);
extern void prepare_fft_vector(float*, float*, float*, int32_t);

extern void cpx_cmul(
    vfloat32m1_t, vfloat32m1_t,
    vfloat32m1_t, vfloat32m1_t,
    vfloat32m1_t*, vfloat32m1_t*,
    size_t vl);


extern void reorder_1536(float*);

extern float* ofdm_scale(float* restrict, int32_t, float);

#endif