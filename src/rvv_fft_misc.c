#include "../include/rvv_fft_misc.h"
#include "../include/tables/bitrev.h"

inline void prepare_fft_vector(float *Rvec, float *Ivec, float *vec, int N) {
   // bit_reverse(Rvec, Ivec, N);
    interleave(Rvec, Ivec, vec, N);
}

inline void interleave(float *re, float *im, float *data, int N) {
    int i = 0;
 
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i); // seting vector len
 
        vfloat32m1_t v_re = __riscv_vle32_v_f32m1(&re[i], vl); // load Re and Im data to vec regs
        vfloat32m1_t v_im = __riscv_vle32_v_f32m1(&im[i], vl);

        vfloat32m1x2_t v_pair = __riscv_vcreate_v_f32m1x2(v_re, v_im); // form a Re and Im pair 
        
        __riscv_vsseg2e32_v_f32m1x2(&data[2 * i], v_pair, vl); // segmented load to interlieved format
 
        i += vl;
    }
}
inline void bit_reverse(float *vec, int N) {
    int count;
    const int* bitrev = get_bitrev_table(N, &count);

    for (int i = 0; i < N; i++)
    {
        int j = bitrev[i];

        if (i < j)
        {
            int i2 = 2 * i;
            int j2 = 2 * j;

            float tmp_r = vec[i2];
            float tmp_i = vec[i2 + 1];

            vec[i2]     = vec[j2];
            vec[i2 + 1] = vec[j2 + 1];

            vec[j2]     = tmp_r;
            vec[j2 + 1] = tmp_i;
        }
    }
}