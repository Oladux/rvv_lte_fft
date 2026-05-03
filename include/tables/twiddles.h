#ifndef TWIDDLES_H
#define TWIDDLES_H

#include <stddef.h>

extern const float twiddles_128[128];
extern const float twiddles_256[256];
extern const float twiddles_512[512];
extern const float twiddles_1024[1024];
extern const float twiddles_1536[3072];
extern const float twiddles_2048[2048];
extern const size_t twiddles_count_128;
extern const size_t twiddles_count_256;
extern const size_t twiddles_count_512;
extern const size_t twiddles_count_1024;
extern const size_t twiddles_count_1536;
extern const size_t twiddles_count_2048;

extern const float* get_twiddle_table(int N, int* num_complex);

#endif