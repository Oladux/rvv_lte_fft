CC = /opt/nuclei/gcc/bin/riscv64-unknown-elf-gcc-14.2.1
CFLAGS = -march=rv64gcv -mabi=lp64d -O3 -ftree-vectorize -specs=nano.specs

TWIDDLES = twiddles/twiddles.c
BITREV = bitrev/bitrev.c

SOURCES = rvv_fft.c $(TWIDDLES) $(BITREV)

compile: $(SOURCES)
	$(CC) $(CFLAGS) -o rvv_fft  $^ -lm -u _printf_float

clean:
	rm -f compile