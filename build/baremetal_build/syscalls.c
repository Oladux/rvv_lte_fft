#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <stddef.h>

volatile uint64_t tohost
__attribute__((section(".tohost"), aligned(64))) = 0;

volatile uint64_t fromhost
__attribute__((section(".fromhost"), aligned(64))) = 0;

static int errno_storage;


static void tc_putchar(char c)
{
    while (tohost)
        ;

    tohost =
        (1ULL << 56) |
        ((uint64_t)(unsigned char)c << 48) |
        1ULL;
}

static void tc_puts(const char *s)
{
    while (*s)
        tc_putchar(*s++);
}

static void tc_put_hex(uint64_t v)
{
    static const char hex[] = "0123456789abcdef";

    tc_puts("0x");

    for (int i = 60; i >= 0; i -= 4)
    {
        tc_putchar(hex[(v >> i) & 0xF]);
    }
}



static uint64_t htif_syscall(
    uint64_t num,
    uint64_t arg0,
    uint64_t arg1,
    uint64_t arg2)
{
    static volatile uint64_t magic_mem[8];

    magic_mem[0] = num;
    magic_mem[1] = arg0;
    magic_mem[2] = arg1;
    magic_mem[3] = arg2;
    magic_mem[4] = 0;
    magic_mem[5] = 0;
    magic_mem[6] = 0;
    magic_mem[7] = 0;

    tohost = (uintptr_t)magic_mem;

    while (fromhost == 0)
        ;

    fromhost = 0;

    return magic_mem[0];
}



int *__errno(void)
{
    return &errno_storage;
}


void __attribute__((noreturn)) _exit(int code)
{
    tohost = ((uint64_t)code << 1) | 1ULL;

    while (1)
        ;
}


ssize_t _write(int fd, const void* buf, size_t count)
{
    (void)fd;

    const char* p = (const char*)buf;

    for(size_t i = 0; i < count; i++)
        tc_putchar(p[i]);

    return count;
}

ssize_t _read(int fd, void *buf, size_t len)
{
    return (ssize_t)htif_syscall(
        63,
        (uint64_t)fd,
        (uintptr_t)buf,
        (uint64_t)len);
}


void *_sbrk(ptrdiff_t incr)
{
    extern char __heap_start;
    extern char __heap_end;

    static char *heap = NULL;

    if (!heap)
        heap = &__heap_start;

    char *prev = heap;

    if (heap + incr > &__heap_end)
        return (void *)-1;

    heap += incr;

    return prev;
}


int _close(int fd)
{
    (void)fd;
    return -1;
}

int _fstat(int fd, struct stat *st)
{
    (void)fd;
    (void)st;
    return 0;
}

int _isatty(int fd)
{
    (void)fd;
    return 1;
}

int _lseek(int fd, int ptr, int dir)
{
    (void)fd;
    (void)ptr;
    (void)dir;
    return 0;
}

int _open(const char *name, int flags, ...)
{
    (void)name;
    (void)flags;
    return -1;
}

int _kill(int pid, int sig)
{
    (void)pid;
    (void)sig;
    return -1;
}

int _getpid(void)
{
    return 1;
}



void trap_c_handler(
    uint64_t mcause,
    uint64_t mepc,
    uint64_t mtval)
{
    tc_puts("\n*** TRAP ***\n");

    tc_puts("mcause = ");
    tc_put_hex(mcause);
    tc_puts("\n");

    tc_puts("mepc   = ");
    tc_put_hex(mepc);
    tc_puts("\n");

    tc_puts("mtval  = ");
    tc_put_hex(mtval);
    tc_puts("\n");

    _exit(1);
}