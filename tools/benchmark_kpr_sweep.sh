#!/bin/bash
# Sweep the KPR OpenVINO benchmark over batch-size x device x precision and collect
# every run into a SINGLE shared summary CSV. Each run keeps its own report.log.
#
# Grid (override via env):
#   BATCH_SIZES : "1"
#   DEVICES     : "CPU NPU GPU"
#   PRECISIONS  : "fp16 fp16-int8"
#
# Other env:
#   DURATION    : per-run timed seconds (default 60)
#   PYTHON      : python interpreter (default ~/python3-venv/kpr-reid/bin/python, else `python`)
#   OUTPUT_CSV  : shared summary CSV path (default results/kpr_ov_sweep_<ts>.csv)
#   EXTRA_ARGS  : extra flags forwarded to benchmark_kpr_ov.py (e.g. "--limit-samples 4")
#
# Usage:
#   bash tools/benchmark_kpr_sweep.sh
#   DURATION=20 DEVICES="CPU NPU" bash tools/benchmark_kpr_sweep.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

BATCH_SIZES="${BATCH_SIZES:-1}"
DEVICES="${DEVICES:-CPU NPU GPU}"
PRECISIONS="${PRECISIONS:-fp16 fp16-int8}"
DURATION="${DURATION:-60}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$HOME/python3-venv/kpr-reid/bin/python" ]]; then
    PYTHON="$HOME/python3-venv/kpr-reid/bin/python"
  else
    PYTHON="python"
  fi
fi

TS="$(date +%Y%m%d-%H%M%S)"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT_DIR/results/kpr_ov_sweep_${TS}.csv}"
LOG_DIR="${OUTPUT_CSV%.csv}_logs"
CACHE_DIR="$ROOT_DIR/openvino_models/.ov_cache"
mkdir -p "$(dirname "$OUTPUT_CSV")" "$LOG_DIR"

# Start each sweep from a fresh combined CSV so stale rows never mix in.
rm -f "$OUTPUT_CSV"

echo "[sweep] python      : $PYTHON"
echo "[sweep] batch sizes : $BATCH_SIZES"
echo "[sweep] devices     : $DEVICES"
echo "[sweep] precisions  : $PRECISIONS"
echo "[sweep] duration    : ${DURATION}s per run"
echo "[sweep] summary CSV : $OUTPUT_CSV"
echo "[sweep] logs        : $LOG_DIR/"
echo

total=0
failed=0
for dev in $DEVICES; do
  for prec in $PRECISIONS; do
    for bs in $BATCH_SIZES; do
      total=$((total + 1))
      tag="${dev}_${prec}_bs${bs}"
      log_file="$LOG_DIR/${tag}.report.log"
      echo "[sweep] running ${tag} ..."
      "$PYTHON" tools/benchmark_kpr_ov.py \
        --device "$dev" \
        --precision "$prec" \
        --batch-size "$bs" \
        --duration "$DURATION" \
        --output-csv "$OUTPUT_CSV" \
        --append \
        --log-file "$log_file" \
        --cache-dir "$CACHE_DIR" \
        $EXTRA_ARGS
      if [[ $? -ne 0 ]]; then
        failed=$((failed + 1))
        echo "[sweep] FAILED ${tag} (see $log_file)"
      fi
    done
  done
done

echo
echo "[sweep] done: $((total - failed))/$total runs ok, $failed failed"
echo "[sweep] summary CSV: $OUTPUT_CSV"
if [[ -f "$OUTPUT_CSV" ]]; then
  echo
  column -t -s, "$OUTPUT_CSV" 2>/dev/null || cat "$OUTPUT_CSV"
fi
