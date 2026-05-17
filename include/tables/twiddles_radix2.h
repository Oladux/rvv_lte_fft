#ifndef TWIDDLES2_H
#define TWIDDLES2_H

extern const float twiddles_128[] __attribute__((aligned(64)));
extern const size_t tw_offsets_128[];
extern const size_t log2_128;


extern const float twiddles_256[] __attribute__((aligned(64)));
extern const size_t tw_offsets_256[];
extern const size_t log2_256;

extern const float twiddles_512[] __attribute__((aligned(64)));
extern const size_t tw_offsets_512[];
extern const size_t log2_512;

extern const float twiddles_1024[] __attribute__((aligned(64)));
extern const size_t tw_offsets_1024[];
extern const size_t log2_1024;

extern const float twiddles_2048[] __attribute__((aligned(64))) ;
extern const size_t tw_offsets_2048[];
extern const size_t log2_2048;

extern const void get_twiddle_r2(int32_t, const float**, const size_t**, size_t*);
   
#endif