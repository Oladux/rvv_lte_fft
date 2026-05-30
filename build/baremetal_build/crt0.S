   .section .text.init
    .globl _start
    .globl trap_handler


_start:

    /* -------------------------------------------------------------------------
     * Stack pointer
     * ------------------------------------------------------------------------- */
    la      sp, __stack_top

    /* -------------------------------------------------------------------------
     * Clear .bss
     * ------------------------------------------------------------------------- */
    la      t0, __bss_start
    la      t1, __bss_end

1:
    bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b

2:

    /* -------------------------------------------------------------------------
     * Install trap vector
     * ------------------------------------------------------------------------- */
    la      t0, trap_handler
    csrw    mtvec, t0

    /* -------------------------------------------------------------------------
     * Call main()
     * ------------------------------------------------------------------------- */

    li t0, (3 << 13) | (3 << 9)
    csrs mstatus, t0
    
    call    main

    /* a0 already contains return value */
    call    _exit


trap_handler:

    addi    sp, sp, -128

    sd      ra,   0(sp)
    sd      t0,   8(sp)
    sd      t1,  16(sp)
    sd      t2,  24(sp)
    sd      a0,  32(sp)
    sd      a1,  40(sp)
    sd      a2,  48(sp)
    sd      a3,  56(sp)
    sd      a4,  64(sp)
    sd      a5,  72(sp)
    sd      a6,  80(sp)
    sd      a7,  88(sp)


    csrr    a0, mcause
    csrr    a1, mepc
    csrr    a2, mtval


    call    trap_c_handler


    ld      ra,   0(sp)
    ld      t0,   8(sp)
    ld      t1,  16(sp)
    ld      t2,  24(sp)
    ld      a0,  32(sp)
    ld      a1,  40(sp)
    ld      a2,  48(sp)
    ld      a3,  56(sp)
    ld      a4,  64(sp)
    ld      a5,  72(sp)
    ld      a6,  80(sp)
    ld      a7,  88(sp)

    addi    sp, sp, 128

    mret