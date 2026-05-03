#include "../include/rvv_fft.h"
#include "../include/rvv_fft_misc.h"


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


    float* vec = aligned_alloc(16, 2 * N * sizeof(float));

    prepare_fft_vector(Re, Im, vec, N);

    vec = rvv_fft(vec, N, 7);


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