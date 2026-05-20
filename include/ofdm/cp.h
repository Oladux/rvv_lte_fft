#ifndef CP_H
#define CP_H


#include "../include/ofdm_fft.h"

static inline void cpx_copy(
    float* restrict,
    const float* restrict,
    size_t);

void ofdm_add_cp(
    float* restrict,
    const float* restrict,
    uint32_t,
    uint32_t);

void ofdm_remove_cp(
    float* restrict,
    const float* restrict,
    uint32_t,
    uint32_t);
#endif