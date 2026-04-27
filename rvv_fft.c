#include <riscv_vector.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "twiddles/twiddles.h"
#include "bitrev/bitrev.h"


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
void bit_reverse(float *re, float *im, int N) {
    int count;
    float tmp_r;
    float tmp_i;
    const int* bitrev = get_bitrev_table(N, &count);
    
    for (int i = 0; i < N; i++) {
        int j = bitrev[i];
    
        if (i < j) {
            tmp_r = re[i];
            re[i] = re[j];
            re[j] = tmp_r;
            
            tmp_i = im[i];
            im[i] = im[j];
            im[j] = tmp_i;
        }
    }
}

float* rvv_fft(float* Rvec, float* Ivec, int N, int log2N) {
    int tw_cnt;
    const float* tw_table = get_twiddle_table(N, &tw_cnt);

    // Бит-реверс
    bit_reverse(Rvec, Ivec, N);
    
    // Подготовка чередованного массива
    float* vec = aligned_alloc(16, 2 * N * sizeof(float));
    interleave(Rvec, Ivec, vec, N);
        
    // Butterfly operation: X1 = A + W*B,  X2 = A - W*B
    // where A = (re1 + i*im1), B = (re2 + i*im2), W = (rw + i*iw)

    for (int stage = 1; stage <= log2N; stage++) {
        int grp_size = 1 << stage; // step = 2 ^ stage
        int grp_half = grp_size >> 1; // half = step / 2
        int stride = N / grp_size; // distance between twiddles in table
        
        for (int grp_start = 0; grp_start < N; grp_start += grp_size) { // split data into blocks with size of step N [0, 1], [2, 3], [4, 5] ...

            for (int j = 0; j < grp_half; j++) {
                    float* ptr_a = grp_start + j; // indexes for first half of group
                    float* ptr_b = i1 + grp_half; // indexes for second half of group
                    
                    float ra = vec[2 * i1]; // load of real part of number from firlst half of group
                    float ia = vec[2 * i1 + 1]; // load of im part of number from firlst half of group
                    float rb = vec[2 * i2]; // load of real part of number from second half of group
                    float ib = vec[2 * i2 + 1]; // load of im part of number from second half of group

                    float rw = tw_table[2 * j * stride]; // load of twiddle  for real part
                    float iw = tw_table[2 * j * stride + 1]; // load of twiddle  for im part
                    
                    // Комплексное умножение и сложение
                    float mr = rb * rw - ib * iw; // real part of complex multiply of B
                    float mi = rb * iw + ib * rw; // im part of complex multiply of B
                    
                    vec[2 * i1]     = ra + mr; // store real part of X1
                    vec[2 * i1 + 1] = ia + mi; // store im part of X1
                    vec[2 * i2]     = ra - mr; // store real part of X2
                    vec[2 * i2 + 1] = ia - mi; // store im part of X2
                }
        }
    }
    
    return vec;
}
int fft_test(){
    static const int N = 128;
    float Re[N];
    float Im[N];
    float  p = 2 * 3.141592653589 / N; 

    int i;
    // формируем сигнал
    for(i=0; i<N; i++)
    {
        Re[i] = cos(p * i);  
        Im[i] = 0.0;     
    }

    float *vec = rvv_fft(Re, Im, N, 7);


    
    for (int i = 0; i < N; i++) {
        printf("Re[%d] = %f, Im[%d] = %f\n", i, vec[2 * i], i, vec[2 * i + 1]);
    }

    free(vec);

    return 1;
}


int main(void){
    fft_test();
    return 0;
}