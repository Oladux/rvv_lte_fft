#ifndef TWIDDLES_H
#define TWIDDLES_H

#include <stddef.h>

// =============================
// N = 128
// =============================

extern const float twiddles_128[] __attribute__((aligned(64)));
extern const int tw_offsets_128[];
extern const int log2_128;


extern const float twiddles_256[] __attribute__((aligned(64)));
extern const int tw_offsets_256[];
extern const int log2_256;

extern const float twiddles_512[] __attribute__((aligned(64)));
extern const int tw_offsets_512[];
extern const int log2_512;

extern const float twiddles_1024[] __attribute__((aligned(64)));
extern const int tw_offsets_1024[];
extern const int log2_1024;

extern const float twiddles_2048[] __attribute__((aligned(64))) ;
extern const int tw_offsets_2048[];
extern const int log2_2048;


extern const void get_twiddle(int, const float**, const int**, int*);
   
#endif