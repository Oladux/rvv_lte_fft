#ifndef OFDM_FFT_H
#define OFDM_FFT_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef SQRT3_OVER_2
#define SQRT3_OVER_2 0.86602540378443864676f  
#endif

#include <stddef.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <riscv_vector.h>

#include "butterflies/butterfly_r2.h"
#include "butterflies/butterfly_r3.h"
#include "butterflies/butterfly_r4.h"

#include "misc/twiddles/load_twiddle_r2.h"
#include "misc/twiddles/load_twiddle_r3.h"
#include "misc/twiddles/load_twiddle_r4.h"

#include "misc/loads/load1.h"
#include "misc/loads/load2.h"

#include "misc/stores/store1.h"
#include "misc/stores/store2.h"

#include "misc/utils/ofdm_fft_utils.h"

#include "tables/twiddles_radix2.h"
#include "tables/twiddles_radix3.h"
#include "tables/twiddles_radix4.h"
#include "tables/bitrev.h"


#include "stages/stages_r2.h"
#include "stages/stages_r3.h"
#include "stages/stages_r4.h"

#include "ofdm/cp.h"


extern float* ofdm_fft(float*, int32_t);
extern float* ofdm_ifft(float*, int32_t);


#endif