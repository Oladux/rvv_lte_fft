#ifndef BITREV_H
#define BITREV_H

extern const int32_t bitrev_128[128];
extern const int32_t bitrev_256[256];
extern const int32_t bitrev_512[512];
extern const int32_t bitrev_1024[1024];
extern const int32_t bitrev_1536[1536];
extern const int32_t bitrev_2048[2048];
extern const size_t bitrev_count_128;
extern const size_t bitrev_count_256;
extern const size_t bitrev_count_512;
extern const size_t bitrev_count_1024;
extern const size_t bitrev_count_1536;
extern const size_t bitrev_count_2048;


extern const int32_t* get_bitrev_table(int32_t N, int32_t* count);

#endif // BITREV_H
