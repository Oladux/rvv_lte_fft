#ifndef TWIDDLES4_H
#define TWIDDLES4_H

#include <stddef.h>

// =============================
// N = 128
// =============================

extern const float twiddles_r4_128[] __attribute__((aligned(64)));
extern const int tw_offsets_r4_128[];
extern const int stages_r4_128 ;


extern const float twiddles_r4_256[] __attribute__((aligned(64)));
extern const int tw_offsets_r4_256[];
extern const int stages_r4_256;

extern const float twiddles_r4_512[] __attribute__((aligned(64)));
extern const int tw_offsets_r4_512[];
extern const int stages_r4_512;

extern const float twiddles_r4_1024[] __attribute__((aligned(64)));
extern const int tw_offsets_r4_1024[];
extern const int stages_r4_1024;

extern const float twiddles_r4_2048[] __attribute__((aligned(64)));
extern const int tw_offsets_r4_2048[];
extern const int stages_r4_2048;


extern const void get_twiddle_r4(int, const float**, const int**, int* );


#endif