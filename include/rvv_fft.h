#ifndef RVV_FFT_H
#define RVV_FFT_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef SQRT3_OVER_2
#define SQRT3_OVER_2 3.14159265358979323846
#endif

#include <stddef.h>
#include <riscv_vector.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "tables/twiddles_radix2.h"
#include "tables/twiddles_radix3.h"
#include "tables/twiddles_radix4.h"
#include "tables/bitrev.h"

float* rvv_fft(float* vec, int N);

#endif