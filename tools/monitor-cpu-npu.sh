#!/bin/bash
# CPU + NPU utilization monitor for Intel Core Ultra X7 358H (Panther Lake)
# Topology:
#   CPU: P-cores cpu0-3 (4.7-4.8 GHz) | E-cores cpu4-11 (3.5 GHz) | LPE-cores cpu12-15 (3.3 GHz)
#   NPU: intel_vpu driver via /sys/bus/pci/devices/*/npu_busy_time_us
#        (monotonically increasing μs the NPU spent on work; readable without sudo)
#
# Usage:
#   bash scripts/monitor-cpu-npu.sh                 # 1s interval, runs until Ctrl+C
#   bash scripts/monitor-cpu-npu.sh 2               # 2s interval
#   bash scripts/monitor-cpu-npu.sh 1 60            # 1s interval, auto-stop after 60s
#   bash scripts/monitor-cpu-npu.sh -v              # show per-group cols (P/E/LPE)
#   bash scripts/monitor-cpu-npu.sh --verbose 1 60
#
# By default the live table shows only CPU-all + NPU columns. The on-exit
# summary always prints every group (incl. P/E/LPE peaks).

set -u

# Parse --verbose / -v; other args are positional (INTERVAL, DURATION).
VERBOSE=0
args=()
for arg in "$@"; do
  case "$arg" in
    -v|--verbose) VERBOSE=1 ;;
    *)            args+=("$arg") ;;
  esac
done
INTERVAL="${args[0]:-1}"
DURATION="${args[1]:-0}"   # 0 = run forever

P_CORES=(0 1 2 3)
E_CORES=(4 5 6 7 8 9 10 11)
LPE_CORES=(12 13 14 15)

# NPU: locate busy-time counter (may be absent on non-Intel-NPU hosts).
NPU_BUSY_PATH=""
for p in /sys/bus/pci/devices/*/npu_busy_time_us; do
  [[ -r "$p" ]] && NPU_BUSY_PATH="$p" && break
done
NPU_PREV_BUSY=""       # last μs reading
NPU_PREV_WALL=""       # monotonic seconds at last reading

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

declare -A PREV_TOTAL PREV_IDLE
ALL_CORES=("${P_CORES[@]}" "${E_CORES[@]}" "${LPE_CORES[@]}")
PEAK_ALL_AVG=0; PEAK_ALL_MAX=0
PEAK_P_AVG=0; PEAK_P_MAX=0
PEAK_E_AVG=0; PEAK_E_MAX=0
PEAK_LPE_AVG=0; PEAK_LPE_MAX=0
PEAK_NPU_AVG=0; PEAK_NPU_MAX=0
SAMPLE_COUNT=0
START_TS=$(date +%s)

# NPU sub-sampling cadence (seconds). Every tick we call sample_npu at this
# rate and report avg + max of the sub-samples.
NPU_SUBSAMPLE_INTERVAL=0.1

# Read /proc/stat once into associative arrays.
# cpu<N> total = user+nice+system+idle+iowait+irq+softirq+steal
# idle counts idle+iowait.
read_cpu_stats() {
  local -n _total=$1
  local -n _idle=$2
  while read -r line; do
    [[ $line =~ ^cpu([0-9]+)[[:space:]]+(.*) ]] || continue
    local id=${BASH_REMATCH[1]}
    local fields=(${BASH_REMATCH[2]})
    local user=${fields[0]} nice=${fields[1]} sys=${fields[2]} idle=${fields[3]}
    local iowait=${fields[4]:-0} irq=${fields[5]:-0} soft=${fields[6]:-0} steal=${fields[7]:-0}
    _total[$id]=$((user + nice + sys + idle + iowait + irq + soft + steal))
    _idle[$id]=$((idle + iowait))
  done < /proc/stat
}

# group_stats "P" 0 1 2 3 -> echoes "avg max"
compute_group() {
  local -n cur_total=$1
  local -n cur_idle=$2
  shift 2
  local cores=("$@")
  local sum=0 max=0 n=0
  for id in "${cores[@]}"; do
    [[ -n "${PREV_TOTAL[$id]:-}" ]] || continue
    local dt=$(( cur_total[$id] - PREV_TOTAL[$id] ))
    local di=$(( cur_idle[$id] - PREV_IDLE[$id] ))
    (( dt > 0 )) || continue
    local util=$(( (dt - di) * 100 / dt ))
    (( util < 0 )) && util=0
    (( util > 100 )) && util=100
    sum=$(( sum + util ))
    (( util > max )) && max=$util
    n=$((n + 1))
  done
  local avg=0
  (( n > 0 )) && avg=$(( sum / n ))
  echo "$avg $max"
}

bar() {
  # $1 = percent 0-100, $2 = width
  local pct=$1 width=$2
  local filled=$(( pct * width / 100 ))
  local i s=""
  for (( i=0; i<filled; i++ )); do s+="█"; done
  for (( i=filled; i<width; i++ )); do s+="░"; done
  echo -n "$s"
}

# Sample NPU utilization as a delta on the busy-time counter over the
# interval since the last call.
# Echoes an integer percent 0-100, or "-1" if NPU is unavailable / first call.
sample_npu() {
  [[ -z "$NPU_BUSY_PATH" ]] && { echo "-1"; return; }
  local busy_us wall_ns wall
  busy_us=$(cat "$NPU_BUSY_PATH" 2>/dev/null || echo "")
  [[ -z "$busy_us" || ! "$busy_us" =~ ^[0-9]+$ ]] && { echo "-1"; return; }
  wall_ns=$(date +%s%N)

  if [[ -z "$NPU_PREV_BUSY" ]]; then
    NPU_PREV_BUSY="$busy_us"
    NPU_PREV_WALL="$wall_ns"
    echo "-1"
    return
  fi

  local d_busy_us=$(( busy_us - NPU_PREV_BUSY ))
  local d_wall_ns=$(( wall_ns - NPU_PREV_WALL ))
  NPU_PREV_BUSY="$busy_us"
  NPU_PREV_WALL="$wall_ns"
  (( d_wall_ns <= 0 )) && { echo "-1"; return; }

  # pct = d_busy_us / d_wall_us * 100 = d_busy_us * 100 * 1000 / d_wall_ns
  local pct=$(( d_busy_us * 100000 / d_wall_ns ))
  (( pct < 0 )) && pct=0
  (( pct > 100 )) && pct=100
  echo "$pct"
}

# Sleep for $INTERVAL seconds while sub-sampling NPU at NPU_SUBSAMPLE_INTERVAL.
# Echoes "avg max n" of the valid sub-samples (both ints 0-100; n = count).
# When NPU is unavailable or we collected no valid samples, echoes "-1 -1 0".
sleep_and_sample_npu() {
  local interval="$1"
  # How many sub-ticks fit in `interval` at NPU_SUBSAMPLE_INTERVAL.
  local n_ticks
  n_ticks=$(awk -v i="$interval" -v s="$NPU_SUBSAMPLE_INTERVAL" \
               'BEGIN { n = int(i / s); if (n < 1) n = 1; print n }')
  local sum=0 max=0 n=0 i pct
  for (( i=0; i<n_ticks; i++ )); do
    sleep "$NPU_SUBSAMPLE_INTERVAL"
    pct=$(sample_npu)
    if (( pct >= 0 )); then
      sum=$(( sum + pct ))
      (( pct > max )) && max=$pct
      n=$(( n + 1 ))
    fi
  done
  if (( n > 0 )); then
    echo "$(( sum / n )) $max $n"
  else
    echo "-1 -1 0"
  fi
}

on_exit() {
  local elapsed=$(( $(date +%s) - START_TS ))
  echo ""
  echo -e "${BOLD}==========================================${NC}"
  echo -e "${BOLD}  CPU+NPU Monitor Summary (${elapsed}s, ${SAMPLE_COUNT} samples)${NC}"
  echo -e "${BOLD}==========================================${NC}"
  printf "  ${CYAN}%-12s${NC} peak avg: ${BOLD}%3d%%${NC}   peak single-core: ${BOLD}%3d%%${NC}\n" \
    "All 16 cores" "$PEAK_ALL_AVG" "$PEAK_ALL_MAX"
  printf "  ${CYAN}%-12s${NC} peak avg: ${BOLD}%3d%%${NC}   peak single-core: ${BOLD}%3d%%${NC}\n" \
    "P-core (4)"  "$PEAK_P_AVG"  "$PEAK_P_MAX"
  printf "  ${CYAN}%-12s${NC} peak avg: ${BOLD}%3d%%${NC}   peak single-core: ${BOLD}%3d%%${NC}\n" \
    "E-core (8)"  "$PEAK_E_AVG"  "$PEAK_E_MAX"
  printf "  ${CYAN}%-12s${NC} peak avg: ${BOLD}%3d%%${NC}   peak single-core: ${BOLD}%3d%%${NC}\n" \
    "LPE-core (4)" "$PEAK_LPE_AVG" "$PEAK_LPE_MAX"
  if [[ -n "$NPU_BUSY_PATH" ]]; then
    printf "  ${CYAN}%-12s${NC} peak avg: ${BOLD}%3d%%${NC}   peak sub-sample: ${BOLD}%3d%%${NC}\n" \
      "NPU"  "$PEAK_NPU_AVG"  "$PEAK_NPU_MAX"
  else
    printf "  ${CYAN}%-12s${NC} ${YELLOW}not detected${NC} (intel_vpu sysfs missing)\n" "NPU"
  fi
  echo ""
  local npu_msg=""
  [[ -n "$NPU_BUSY_PATH" ]] && npu_msg=", NPU avg ~${PEAK_NPU_AVG}% / peak ~${PEAK_NPU_MAX}%"
  echo -e "  ${BOLD}Conclusion:${NC} peak CPU ~${PEAK_ALL_AVG}% (P ${PEAK_P_AVG}% / E ${PEAK_E_AVG}% / LPE ${PEAK_LPE_AVG}%)${npu_msg}"
  echo -e "${BOLD}==========================================${NC}"
  exit 0
}
trap on_exit INT TERM

echo -e "${BOLD}"
echo "=========================================="
echo "  CPU+NPU Monitor — Intel Core Ultra X7 358H"
echo "=========================================="
echo -e "${NC}"
echo "Topology: P-core 0-3 (4.7-4.8GHz) | E-core 4-11 (3.5GHz) | LPE-core 12-15 (3.3GHz)"
if [[ -n "$NPU_BUSY_PATH" ]]; then
  echo "NPU: intel_vpu counter at ${NPU_BUSY_PATH}"
else
  echo -e "${YELLOW}NPU: not detected (intel_vpu sysfs missing); NPU column will show '-'${NC}"
fi
echo "Sampling every ${INTERVAL}s. Press Ctrl+C to stop and see peaks."
echo ""
if (( VERBOSE )); then
  printf "${BOLD}%-10s  %-24s  %-24s  %-24s  %-24s  %-18s${NC}\n" \
    "Time" "CPU-all avg/max" "P-core avg/max" "E-core avg/max" "LPE-core avg/max" "NPU avg/max"
  echo "-------------------------------------------------------------------------------------------------------------------------------------"
else
  printf "${BOLD}%-10s  %-24s  %-18s${NC}\n" \
    "Time" "CPU-all avg/max" "NPU avg/max"
  echo "------------------------------------------------------------"
fi

# Prime previous counters
declare -A CUR_TOTAL CUR_IDLE
read_cpu_stats PREV_TOTAL PREV_IDLE
# Prime NPU snapshot (first call returns -1; next call starts real deltas).
sample_npu > /dev/null

while true; do
  # Sleep for INTERVAL while sub-sampling NPU every NPU_SUBSAMPLE_INTERVAL.
  # This replaces the plain `sleep $INTERVAL` + one-shot NPU sample so we
  # can surface avg/max over short bursty NPU workloads.
  read npu_avg npu_max npu_n <<< "$(sleep_and_sample_npu "$INTERVAL")"

  read_cpu_stats CUR_TOTAL CUR_IDLE

  read all_avg all_max <<< "$(compute_group CUR_TOTAL CUR_IDLE "${ALL_CORES[@]}")"
  read p_avg p_max <<< "$(compute_group CUR_TOTAL CUR_IDLE "${P_CORES[@]}")"
  read e_avg e_max <<< "$(compute_group CUR_TOTAL CUR_IDLE "${E_CORES[@]}")"
  read lpe_avg lpe_max <<< "$(compute_group CUR_TOTAL CUR_IDLE "${LPE_CORES[@]}")"

  (( all_avg > PEAK_ALL_AVG )) && PEAK_ALL_AVG=$all_avg
  (( all_max > PEAK_ALL_MAX )) && PEAK_ALL_MAX=$all_max
  (( p_avg   > PEAK_P_AVG ))   && PEAK_P_AVG=$p_avg
  (( p_max   > PEAK_P_MAX ))   && PEAK_P_MAX=$p_max
  (( e_avg   > PEAK_E_AVG ))   && PEAK_E_AVG=$e_avg
  (( e_max   > PEAK_E_MAX ))   && PEAK_E_MAX=$e_max
  (( lpe_avg > PEAK_LPE_AVG )) && PEAK_LPE_AVG=$lpe_avg
  (( lpe_max > PEAK_LPE_MAX )) && PEAK_LPE_MAX=$lpe_max
  if (( npu_avg >= 0 )); then
    (( npu_avg > PEAK_NPU_AVG )) && PEAK_NPU_AVG=$npu_avg
    (( npu_max > PEAK_NPU_MAX )) && PEAK_NPU_MAX=$npu_max
  fi

  printf "%-10s  " "$(date +%H:%M:%S)"
  printf "${GREEN}%3d%%${NC}/${YELLOW}%3d%%${NC} %s  " "$all_avg" "$all_max" "$(bar $all_avg 14)"
  if (( VERBOSE )); then
    printf "${GREEN}%3d%%${NC}/${YELLOW}%3d%%${NC} %s  " "$p_avg"   "$p_max"   "$(bar $p_avg 14)"
    printf "${GREEN}%3d%%${NC}/${YELLOW}%3d%%${NC} %s  " "$e_avg"   "$e_max"   "$(bar $e_avg 14)"
    printf "${GREEN}%3d%%${NC}/${YELLOW}%3d%%${NC} %s  " "$lpe_avg" "$lpe_max" "$(bar $lpe_avg 14)"
  fi
  if (( npu_avg >= 0 )); then
    printf "${GREEN}%3d%%${NC}/${YELLOW}%3d%%${NC} %s\n" "$npu_avg" "$npu_max" "$(bar $npu_avg 6)"
  else
    printf "     -       \n"
  fi

  # copy current -> previous for next iteration
  for id in "${!CUR_TOTAL[@]}"; do
    PREV_TOTAL[$id]=${CUR_TOTAL[$id]}
    PREV_IDLE[$id]=${CUR_IDLE[$id]}
  done
  SAMPLE_COUNT=$((SAMPLE_COUNT + 1))

  if (( DURATION > 0 )) && (( $(date +%s) - START_TS >= DURATION )); then
    on_exit
  fi
done
