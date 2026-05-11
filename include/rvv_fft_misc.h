#ifndef MISC_H
#define MISC_H

#include <stddef.h>
#include <riscv_vector.h>

void bit_reverse(float*, int);
void interleave(float*, float*, float*, int);
void prepare_fft_vector(float*, float*, float*, int);

#endif