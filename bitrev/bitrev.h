#ifndef BITREV_H
#define BITREV_H

#include <stddef.h>


extern const int bitrev_128[128];
extern const int bitrev_256[256];
extern const int bitrev_512[512];
extern const int bitrev_1024[1024];
extern const int bitrev_1536[1536];
extern const int bitrev_2048[2048];
extern const size_t bitrev_count_128;
extern const size_t bitrev_count_256;
extern const size_t bitrev_count_512;
extern const size_t bitrev_count_1024;
extern const size_t bitrev_count_1536;
extern const size_t bitrev_count_2048;


extern const int* get_bitrev_table(int N, int* count);

#endif // BITREV_H
