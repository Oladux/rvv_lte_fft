MAKEFILE_DIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT = $(realpath $(MAKEFILE_DIR)/..)

ENABLE_PIPELINE ?= 0

ifeq ($(ENABLE_PIPELINE),1)
PIPELINE_FLAGS = -DENABLE_PIPELINE=1
else
PIPELINE_FLAGS = -DENABLE_PIPELINE=0
endif

OPTIMIZATION_FLAGS = \
	-ffast-math \
	-ffp-contract=fast \
	-funroll-loops \
	-fomit-frame-pointer \
	-fno-signed-zeros \
	-fno-trapping-math

CC = /opt/nuclei/gcc/bin/riscv64-unknown-elf-gcc-14.2.1

CFLAGS = \
	-march=rv64gcv \
	-mabi=lp64d \
	-mcmodel=medany \
	-O3 \
	-ftree-vectorize \
	-I$(PROJECT_ROOT)/include \
	-nostartfiles \
	-nostdlib \
	-lgcc \
	$(OPTIMIZATION_FLAGS) \
	$(PIPELINE_FLAGS)

LDFLAGS = \
	-lm \
	-T baremetal_build/linker.ld \
	baremetal_build/syscalls.c \
	baremetal_build/crt0.S

INCDIR         = $(PROJECT_ROOT)/include
SRCDIR         = $(PROJECT_ROOT)/src
TESTDIR        = $(PROJECT_ROOT)/test
TABLESDIR      = $(SRCDIR)/tables
OFDMDIR        = $(SRCDIR)/ofdm
BUTTERFLIESDIR = $(SRCDIR)/butterflies
STAGESDIR      = $(SRCDIR)/stages
UTILSDIR       = $(SRCDIR)/utils

BUILDDIR = $(PROJECT_ROOT)/build
OBJDIR   = $(BUILDDIR)/obj
LIBDIR   = $(BUILDDIR)/lib

DIRS = \
	$(OBJDIR) \
	$(OBJDIR)/tables \
	$(OBJDIR)/utils \
	$(OBJDIR)/stages \
	$(OBJDIR)/ofdm \
	$(LIBDIR)

LIB_SOURCES = \
	$(PROJECT_ROOT)/ofdm_fft.c \
	$(UTILSDIR)/ofdm_fft_utils.c \
	$(STAGESDIR)/stages_r2.c \
	$(STAGESDIR)/stages_r3.c \
	$(STAGESDIR)/stages_r4.c \
	$(TABLESDIR)/twiddles_radix2.c \
	$(TABLESDIR)/twiddles_radix3.c \
	$(TABLESDIR)/twiddles_radix4.c \
	$(OFDMDIR)/cp.c \
	$(TABLESDIR)/bitrev.c

LIB_OBJECTS = \
	$(OBJDIR)/ofdm_fft.o \
	$(OBJDIR)/utils/ofdm_fft_utils.o \
	$(OBJDIR)/tables/twiddles_radix2.o \
	$(OBJDIR)/tables/twiddles_radix3.o \
	$(OBJDIR)/tables/twiddles_radix4.o \
	$(OBJDIR)/stages/stages_r2.o \
	$(OBJDIR)/stages/stages_r3.o \
	$(OBJDIR)/stages/stages_r4.o \
	$(OBJDIR)/ofdm/cp.o \
	$(OBJDIR)/tables/bitrev.o

LIBRARY = $(LIBDIR)/librvv_fft.a


all: $(DIRS) $(LIBRARY) $(RVV_TARGET) $(SCALAR_TARGET)


$(DIRS):
	mkdir -p $@


$(OBJDIR)/ofdm_fft.o: $(PROJECT_ROOT)/ofdm_fft.c | $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/utils/ofdm_fft_utils.o: $(UTILSDIR)/ofdm_fft_utils.c | $(OBJDIR)/utils
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/tables/%.o: $(TABLESDIR)/%.c | $(OBJDIR)/tables
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/stages/%.o: $(STAGESDIR)/%.c | $(OBJDIR)/stages
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/ofdm/%.o: $(OFDMDIR)/%.c | $(OBJDIR)/ofdm
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@


$(LIBRARY): $(LIB_OBJECTS) | $(LIBDIR)
	ar rcs $@ $^


$(RVV_TARGET): $(RVV_SOURCE) $(LIBRARY)
	$(CC) $(CFLAGS) \
	-I$(INCDIR) \
	-o $@ \
	$< \
	-L$(LIBDIR) \
	-lrvv_fft \
	$(LDFLAGS)


$(SCALAR_TARGET): $(SCALAR_SOURCE)
	$(CC) $(CFLAGS) \
	-o $@ \
	$< \
	$(LDFLAGS)


bench-rvv: $(RVV_TARGET)
	./bencmark_script/bench.sh --elf bench_rvv

bench-scalar: $(SCALAR_TARGET)
	./bencmark_script/bench.sh --elf bench_scalar


clean:
	rm -f $(RVV_TARGET)
	rm -f $(SCALAR_TARGET)
	rm -rf $(LIBDIR)
	rm -rf $(OBJDIR)

.PHONY: \
	all \
	bench-rvv \
	bench-scalar \
	clean \