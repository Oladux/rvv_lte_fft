MAKEFILE_DIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT = $(realpath $(MAKEFILE_DIR)/..)

CC = /opt/nuclei/gcc/bin/riscv64-unknown-elf-gcc-14.2.1
CFLAGS = -march=rv64gcv -mabi=lp64d -O3 -ftree-vectorize -specs=nano.specs -I$(PROJECT_ROOT)/include
LDFLAGS = -lm -u _printf_float

INCDIR = $(PROJECT_ROOT)/include
SRCDIR = $(PROJECT_ROOT)/src
TESTDIR = $(PROJECT_ROOT)/test
TABLESDIR = $(SRCDIR)/tables
BUILDDIR = $(PROJECT_ROOT)/build
OBJDIR = $(BUILDDIR)/obj
LIBDIR = $(BUILDDIR)/lib

DIRS = $(OBJDIR) $(OBJDIR)/tables $(LIBDIR)

# Исходные файлы библиотеки
LIB_SOURCES = $(SRCDIR)/rvv_fft.c \
              $(SRCDIR)/rvv_fft_misc.c \
              $(TABLESDIR)/twiddles_radix2.c \
			  $(TABLESDIR)/twiddles_radix3.c \
			  $(TABLESDIR)/twiddles_radix4.c \
              $(TABLESDIR)/bitrev.c

# Объектные файлы
LIB_OBJECTS = $(OBJDIR)/rvv_fft.o \
              $(OBJDIR)/rvv_fft_misc.o \
              $(OBJDIR)/tables/twiddles_radix2.o \
			  $(OBJDIR)/tables/twiddles_radix3.o \
			  $(OBJDIR)/tables/twiddles_radix4.o \
              $(OBJDIR)/tables/bitrev.o

LIBRARY = $(LIBDIR)/librvv_fft.a

TEST_SOURCE = $(TESTDIR)/rvv_test_fft.c
TEST_TARGET = rvv_test_fft

all: $(DIRS) $(LIBRARY) $(TEST_TARGET)

$(DIRS):
	mkdir -p $@

$(OBJDIR)/rvv_fft.o: $(SRCDIR)/rvv_fft.c | $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/rvv_fft_misc.o: $(SRCDIR)/rvv_fft_misc.c | $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/tables/%.o: $(TABLESDIR)/%.c | $(OBJDIR)/tables
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(LIBRARY): $(LIB_OBJECTS) | $(LIBDIR)
	ar rcs $@ $^

$(TEST_TARGET): $(TEST_SOURCE) $(LIBRARY)
	$(CC) $(CFLAGS) -I$(INCDIR) -o $@ $< -L$(LIBDIR) -lrvv_fft $(LDFLAGS)

# Очистка
clean:
	rm -f $(TEST_TARGET)
	rm -rf $(LIBDIR)
	rm -rf $(OBJDIR)

.PHONY: all clean distclean