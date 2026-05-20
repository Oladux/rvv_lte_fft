MAKEFILE_DIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT = $(realpath $(MAKEFILE_DIR)/..)

CC = /opt/nuclei/gcc/bin/riscv64-unknown-elf-gcc-14.2.1
CFLAGS = -march=rv64gcv -mabi=lp64d -mcmodel=medany -O3 -ftree-vectorize  -I$(PROJECT_ROOT)/include -nostartfiles -nostdlib  -lgcc
LDFLAGS = -lm -T baremetal_build/linker.ld baremetal_build/syscalls.c baremetal_build/crt0.S

INCDIR = $(PROJECT_ROOT)/include
SRCDIR = $(PROJECT_ROOT)/src
TESTDIR = $(PROJECT_ROOT)/test
TABLESDIR = $(SRCDIR)/tables
MISCDIR = $(SRCDIR)/misc
OFDMDIR = $(SRCDIR)/ofdm
BUTTERFLIESDIR = $(SRCDIR)/butterflies
STAGESDIR = $(SRCDIR)/stages
UTILSDIR = $(MISCDIR)/utils
TWIDDLESDIR = $(MISCDIR)/twiddles
LOADSDIR = $(MISCDIR)/loads
STORESDIR = $(MISCDIR)/stores
BUILDDIR = $(PROJECT_ROOT)/build
OBJDIR = $(BUILDDIR)/obj
LIBDIR = $(BUILDDIR)/lib

DIRS = $(OBJDIR) \
       $(OBJDIR)/tables \
       $(OBJDIR)/utils \
       $(OBJDIR)/butterflies \
       $(OBJDIR)/stages \
       $(OBJDIR)/twiddles \
       $(OBJDIR)/loads \
       $(OBJDIR)/stores \
       $(OBJDIR)/ofdm \
       $(LIBDIR)

LIB_SOURCES = $(PROJECT_ROOT)/ofdm_fft.c \
              $(UTILSDIR)/ofdm_fft_utils.c \
              $(BUTTERFLIESDIR)/butterfly_r2.c \
              $(BUTTERFLIESDIR)/butterfly_r3.c \
              $(BUTTERFLIESDIR)/butterfly_r4.c \
              $(STAGESDIR)/stages_r2.c \
              $(STAGESDIR)/stages_r3.c \
              $(STAGESDIR)/stages_r4.c \
              $(TABLESDIR)/twiddles_radix2.c \
              $(TABLESDIR)/twiddles_radix3.c \
              $(TABLESDIR)/twiddles_radix4.c \
              $(TWIDDLESDIR)/load_twiddle_r2.c \
              $(TWIDDLESDIR)/load_twiddle_r3.c \
              $(TWIDDLESDIR)/load_twiddle_r4.c \
              $(LOADSDIR)/load1.c \
              $(LOADSDIR)/load2.c \
              $(STORESDIR)/store1.c \
              $(STORESDIR)/store2.c \
              $(OFDMDIR)/cp.c \
              $(TABLESDIR)/bitrev.c	

LIB_OBJECTS = $(OBJDIR)/ofdm_fft.o \
              $(OBJDIR)/utils/ofdm_fft_utils.o \
              $(OBJDIR)/tables/twiddles_radix2.o \
              $(OBJDIR)/tables/twiddles_radix3.o \
              $(OBJDIR)/tables/twiddles_radix4.o \
              $(OBJDIR)/butterflies/butterfly_r2.o \
              $(OBJDIR)/butterflies/butterfly_r3.o \
              $(OBJDIR)/butterflies/butterfly_r4.o \
              $(OBJDIR)/stages/stages_r2.o \
              $(OBJDIR)/stages/stages_r3.o \
              $(OBJDIR)/stages/stages_r4.o \
              $(OBJDIR)/twiddles/load_twiddle_r2.o \
              $(OBJDIR)/twiddles/load_twiddle_r3.o \
              $(OBJDIR)/twiddles/load_twiddle_r4.o \
              $(OBJDIR)/loads/load1.o \
              $(OBJDIR)/loads/load2.o \
              $(OBJDIR)/stores/store1.o \
              $(OBJDIR)/stores/store2.o \
              $(OBJDIR)/ofdm/cp.o \
              $(OBJDIR)/tables/bitrev.o

LIBRARY = $(LIBDIR)/librvv_fft.a

TEST_SOURCE = $(TESTDIR)/test_suite.c
TEST_TARGET = test_suite

all: $(DIRS) $(LIBRARY) $(TEST_TARGET)


$(DIRS):
	mkdir -p $@


# rvv_fft.c
$(OBJDIR)/ofdm_fft.o: $(PROJECT_ROOT)/ofdm_fft.c | $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# rvv_fft_utils.c (utils)
$(OBJDIR)/utils/ofdm_fft_utils.o: $(UTILSDIR)/ofdm_fft_utils.c | $(OBJDIR)/utils
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@



$(OBJDIR)/tables/%.o: $(TABLESDIR)/%.c | $(OBJDIR)/tables
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@



$(OBJDIR)/butterflies/%.o: $(BUTTERFLIESDIR)/%.c | $(OBJDIR)/butterflies
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@



$(OBJDIR)/stages/%.o: $(STAGESDIR)/%.c | $(OBJDIR)/stages
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@



$(OBJDIR)/twiddles/%.o: $(TWIDDLESDIR)/%.c | $(OBJDIR)/twiddles
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@



$(OBJDIR)/loads/%.o: $(LOADSDIR)/%.c | $(OBJDIR)/loads
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@


$(OBJDIR)/stores/%.o: $(STORESDIR)/%.c | $(OBJDIR)/stores
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/ofdm/%.o: $(OFDMDIR)/%.c | $(OBJDIR)/ofdm
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(LIBRARY): $(LIB_OBJECTS) | $(LIBDIR)
	ar rcs $@ $^

$(TEST_TARGET): $(TEST_SOURCE) $(LIBRARY)
	$(CC) $(CFLAGS) -I$(INCDIR) -o $@ $< -L$(LIBDIR) -lrvv_fft $(LDFLAGS)



clean:
	rm -f $(TEST_TARGET)
	rm -rf $(LIBDIR)
	rm -rf $(OBJDIR)

distclean: clean

.PHONY: all clean distclean