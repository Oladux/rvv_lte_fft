# OFDM FFT

## RISC-V FFT Library Optimized with the RISC-V Vector Extension (RVV) for LTE OFDM

### Features

- LTE-compatible FFT sizes:
  - 128
  - 256
  - 512
  - 1024
  - 1536
  - 2048
- Mixed-radix decomposition:
  - Radix-2 + Radix-4
  - Radix-3 + Radix-4
- Interleaved complex input and output format
- Optimized for the RISC-V Vector Extension (RVV)

### FFT Parameters

| Parameter | Value |
|------------|---------|
| Supported sizes | 128–2048 |
| Architecture | RISC-V RV64 |
| Vector ISA | RVV |
| Algorithm | Mixed-radix FFT |
| Input format | Interleaved complex samples |
| Output format | Interleaved complex samples |
| Target application | LTE OFDM PHY layer |

### Requirements

To build and run the benchmark, the following software must be installed:

- RISC-V GNU Toolchain
- Spike RISC-V ISA Simulator
- OpenOCD with RISC-V support
- GDB (RISC-V)

### Build

```bash
cd build
make bench-rvv
```
