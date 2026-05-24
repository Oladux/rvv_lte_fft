shell sleep 1

set pagination off
set confirm off
set height 0
set width 0

set architecture riscv:rv64

set remotetimeout 120
set tcp connect-timeout 120

target remote :3333

echo Connected to OpenOCD.\n

monitor reset halt

shell sleep 1

echo Loading ELF sections...\n

load

echo ELF loaded.\n

hb main

echo Starting target...\n

continue

delete breakpoints

echo main() reached.\n

echo Running benchmark...\n

continue

echo Benchmark finished.\n
echo Benchmark finished.\n
 
python
import math
import struct
import gdb

RESULT_STRUCT_SIZE = 48
N_SIZES            = 6

inf = gdb.selected_inferior()
if inf is None:
    raise gdb.GdbError("No inferior selected")

try:
    sym_results = gdb.parse_and_eval('&g_results')
except gdb.error:
    raise gdb.GdbError("Symbol g_results not found")

addr_results = int(sym_results)

raw_results = bytes(inf.read_memory(addr_results, RESULT_STRUCT_SIZE * N_SIZES))

results = []
for i in range(N_SIZES):
    off = i * RESULT_STRUCT_SIZE
    chunk = raw_results[off : off + RESULT_STRUCT_SIZE]
    N, cyc_min, ins_min, cyc_med, ins_med, done = struct.unpack_from('<6Q', chunk)
    results.append({
        'N': N, 'cyc_min': cyc_min, 'ins_min': ins_min,
        'cyc_med': cyc_med, 'ins_med': ins_med, 'done': done,
    })

try:
    sym_errors = gdb.parse_and_eval('&g_errors')
    addr_errors = int(sym_errors)
    raw_errors = bytes(inf.read_memory(addr_errors, 24 * N_SIZES))
    errors = []
    for i in range(N_SIZES):
        off = i * 24
        chunk = raw_errors[off : off + 24]
        a, r, s = struct.unpack_from('<3d', chunk)
        errors.append({'max_abs_error': a, 'max_rel_error': r, 'snr_db': s})
    HAS_ERRORS = True
except:
    errors = [{'max_abs_error': 0.0, 'max_rel_error': 0.0, 'snr_db': 0.0}] * N_SIZES
    HAS_ERRORS = False

def theoretical_flops(N):
    return 5.0 * N * math.log2(N)

def safe_div(a, b):
    return a / b if b != 0 else 0.0
 
def flop_per_ins(flops, ins):
    return safe_div(flops, ins)
 
def ipc(ins, cyc):
    return safe_div(ins, cyc)
 
def cpi(cyc, ins):
    return safe_div(cyc, ins)
 
def cyc_per_work(cyc, N):
    work = N * math.log2(N)
    return safe_div(cyc, work)
 
def samples_per_cycle(N, cyc):
    return safe_div(N, cyc)
 
SEP = (
    '+' + '-'*6  +
    '+' + '-'*12 +
    '+' + '-'*12 +
    '+' + '-'*12 +
    '+' + '-'*10 +
    '+' + '-'*10 +
    '+' + '-'*10 +
    '+' + '-'*10 +
    '+'
)
 
if HAS_ERRORS:
    HDR = (
        '| {:4s} | {:>10s} | {:>10s} | {:>10s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>12s} | {:>12s} |'
        .format('N', 'cyc(min)', 'ins(min)', 'cyc/NlogN', 'IPC', 'CPI', 'FPI', 's/cycle', 'max_abs', 'max_rel')
    )
    SEP = SEP[:-1] + '+' + '-'*14 + '+' + '-'*14 + '+'
else:
    HDR = (
        '| {:4s} | {:>10s} | {:>10s} | {:>10s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} |'
        .format('N', 'cyc(min)', 'ins(min)', 'cyc/NlogN', 'IPC', 'CPI', 'FPI', 's/cycle')
    )
 
print()
print('  ofdm_fft benchmark results')
print('  SPIKE bare-metal / mcycle + minstret CSR')
print('  warmup=3  repeat=11')
print()
print('  ' + SEP)
print('  ' + HDR)
print('  ' + SEP)
 
for idx, r in enumerate(results):
    if not r['done']:
        if HAS_ERRORS:
            print('  | {:>4} | {:>10s} | {:>10s} | {:>10s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>12s} | {:>12s} |'
                  .format(str(r['N']), '---', '---', '---', '---', '---', '---', '---', '---', '---'))
        else:
            print('  | {:>4} | {:>10s} | {:>10s} | {:>10s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} |'
                  .format(str(r['N']), '---', '---', '---', '---', '---', '---', '---'))
        continue
 
    N   = r['N']
    cyc = r['cyc_min']
    ins = r['ins_min']
    err = errors[idx]
 
    flops = theoretical_flops(N)
    metric_fpi = flop_per_ins(flops, ins)
    metric_ipc = ipc(ins, cyc)
    metric_cpi = cpi(cyc, ins)
    metric_cw  = cyc_per_work(cyc, N)
    metric_spc = samples_per_cycle(N, cyc)
 
    if HAS_ERRORS:
        print('  | {:>4} | {:>10} | {:>10} | {:>10.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.5f} | {:>12.3e} | {:>12.3e} |'
              .format(int(N), cyc, ins, metric_cw, metric_ipc, metric_cpi, metric_fpi, metric_spc,
                      err['max_abs_error'], err['max_rel_error']))
    else:
        print('  | {:>4} | {:>10} | {:>10} | {:>10.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.5f} |'
              .format(int(N), cyc, ins, metric_cw, metric_ipc, metric_cpi, metric_fpi, metric_spc))
 
print('  ' + SEP)
print()
print('  Metrics:')
print('    cyc/NlogN : FFT normalized cycle cost')
print('    IPC       : instructions per cycle')
print('    CPI       : cycles per instruction')
print('    FPI       : theoretical FFT FLOPs / instruction')
print('    s/cycle   : FFT samples processed per cycle')
if HAS_ERRORS:
    print('    max_abs   : maximum absolute error')
    print('    max_rel   : maximum relative error')
print()
 
end
 
echo Checking machine state...\n
print/x $mcause
print/x $mepc
x/10i $mepc
 
echo GDB benchmark session finished.\n
 
quit
 