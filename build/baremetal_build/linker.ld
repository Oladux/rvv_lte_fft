OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
    . = ORIGIN(RAM);

    .text : {
        *(.text.init)
        *(.text*)
    } > RAM

    .rodata : {
        *(.rodata*)
    } > RAM

    .data : {
        *(.data*)
    } > RAM

    .bss : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    . = ALIGN(16);

    __stack_top = ORIGIN(RAM) + LENGTH(RAM);

    . = ALIGN(64);

    .stack (NOLOAD) :
    {
        . = ALIGN(64);
        _stack_bottom = .;

        . += 64K;

        . = ALIGN(64);
        _stack_top = .;
    }

    .tohost : {
        tohost = .;
        QUAD(0);

        . = ALIGN(64);

        fromhost = .;
        QUAD(0);
    } > RAM

    __heap_start = .;
    __heap_end   = ORIGIN(RAM) + LENGTH(RAM) - 0x10000;
}