#!/usr/bin/env bash

 
set -euo pipefail

RBB_PORT=9824     
GDB_PORT=3333    
ISA=rv64gcv
ELF=test_suite
GDB_SCRIPT=bencmark_script/bench.gdb
OPENOCD_CFG=bencmark_script/openocd.cfg
NO_BUILD=0
BENCH=$0
 

SPIKE=${SPIKE:-spike}
OPENOCD=${OPENOCD:-openocd}
GDB=${GDB:-riscv64-unknown-elf-gdb}
 

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build)    NO_BUILD=1;     shift ;;
        --rbb-port)    RBB_PORT=$2;    shift 2 ;;
        --gdb-port)    GDB_PORT=$2;    shift 2 ;;
        --isa)         ISA=$2;         shift 2 ;;
        --elf)         ELF=$2;         shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
 

log()  { echo "[bench] $*"; }
err()  { echo "[bench] ERROR: $*" >&2; exit 1; }
 

wait_port() {
    local port="$1"
    local timeout="$2"
    local name="$3"
    local elapsed=0

    printf "[bench] Waiting for %s on :%s" "$name" "$port"

    while ! nc -z 127.0.0.1 "$port" 2>/dev/null; do
        printf "."
        sleep 1
        elapsed=$((elapsed + 1))

        if [ "$elapsed" -ge "$timeout" ]; then
            echo
            err "${name} did not open port ${port} within ${timeout}s"
        fi
    done

    echo " OK"
}

 

for tool in "${SPIKE}" "${OPENOCD}" "${GDB}"; do
    command -v "${tool}" &>/dev/null \
        || err "'${tool}' not found. Install or set env var."
done
 

if [[ $NO_BUILD -eq 0 ]]; then
    log "Building ${ELF}..."
    make clean
    make $(BENCH) ISA="${ISA}"
    log "Build OK"
fi
 
[[ -f "${ELF}" ]] || err "${ELF} not found. Run without --no-build."
 

SPIKE_LOG=$(mktemp /tmp/bench_spike_XXXXXX.log)
OPENOCD_LOG=$(mktemp /tmp/bench_openocd_XXXXXX.log)
PATCHED_CFG=$(mktemp /tmp/bench_ocd_XXXXXX.cfg)
PATCHED_GDB=$(mktemp /tmp/bench_gdb_XXXXXX.gdb)
 

sed \
    -e "s/remote_bitbang port [0-9]*/remote_bitbang port ${RBB_PORT}/" \
    -e "s/gdb_port[ ]*[0-9]*/gdb_port     ${GDB_PORT}/" \
    "${OPENOCD_CFG}" > "${PATCHED_CFG}"
 
sed "s/:3333/:${GDB_PORT}/g" "${GDB_SCRIPT}" > "${PATCHED_GDB}"

SPIKE_PID=""
OPENOCD_PID=""
 
cleanup() {
    local rc=$?
    [[ -n "${OPENOCD_PID}" ]] && \
        kill "${OPENOCD_PID}" 2>/dev/null; wait "${OPENOCD_PID}" 2>/dev/null || true
    [[ -n "${SPIKE_PID}" ]]   && \
        kill "${SPIKE_PID}"   2>/dev/null; wait "${SPIKE_PID}"   2>/dev/null || true
    rm -f "${PATCHED_CFG}" "${PATCHED_GDB}"
    if [[ $rc -ne 0 ]]; then
        echo
        log "=== SPIKE log (last 20 lines) ==="
        tail -20 "${SPIKE_LOG}"  2>/dev/null || true
        log "=== OpenOCD log (last 20 lines) ==="
        tail -20 "${OPENOCD_LOG}" 2>/dev/null || true
    fi
    rm -f "${SPIKE_LOG}" "${OPENOCD_LOG}"
}
trap cleanup EXIT INT TERM
 

log "Starting SPIKE (ISA=${ISA}, rbb-port=${RBB_PORT})..."

"${SPIKE}" \
    --isa="${ISA}" \
    --rbb-port="${RBB_PORT}" \
    "${ELF}" \
    > "${SPIKE_LOG}" 2>&1 &

SPIKE_PID=$!

wait_port "${RBB_PORT}" 5 "SPIKE"
 

log "Starting OpenOCD (rbb→:${RBB_PORT}, gdb→:${GDB_PORT})..."

"${OPENOCD}" \
    -f "${PATCHED_CFG}" \
    > "${OPENOCD_LOG}" 2>&1 &

OPENOCD_PID=$!

wait_port "${GDB_PORT}" 10 "OpenOCD GDB server"
 
sleep 0.5
 
log "Running GDB..."
echo
 
"${GDB}" \
    -batch \
    -ex "source ${PATCHED_GDB}" \
    "${ELF}"
 
GDB_RC=$?
 
echo
[[ $GDB_RC -eq 0 ]] \
    && log "Done (GDB exit 0)." \
    || log "WARNING: GDB exited with code ${GDB_RC}."
