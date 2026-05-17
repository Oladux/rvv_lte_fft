#ifndef TWIDDLES4_H
#define TWIDDLES4_H

extern const float twiddles_r4_128[] __attribute__((aligned(64)));
extern const size_t tw_offsets_r4_128[];
extern const size_t stages_r4_128 ;


extern const float twiddles_r4_256[] __attribute__((aligned(64)));
extern const size_t tw_offsets_r4_256[];
extern const size_t stages_r4_256;

extern const float twiddles_r4_512[] __attribute__((aligned(64)));
extern const size_t tw_offsets_r4_512[];
extern const size_t stages_r4_512;

extern const float twiddles_r4_1024[] __attribute__((aligned(64)));
extern const size_t tw_offsets_r4_1024[];
extern const size_t stages_r4_1024;

extern const float twiddles_r4_2048[] __attribute__((aligned(64)));
extern const size_t tw_offsets_r4_2048[];
extern const size_t stages_r4_2048;


extern void get_twiddle_r4(int32_t N, const float** tw, const size_t** off, size_t* stages);


#endif