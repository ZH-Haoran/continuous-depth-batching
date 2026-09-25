#!/usr/bin/env bash
# Profiling CB and CDB via Nsight Systems trace of the offline throughput.
# The analysis reports GPU busy/idle and per-NVTX-stage time.
#
# Submit with:
#   MODEL=ouro   ./shells/_submit.sh shells/paper/ablations/profile_nsys.sh
#   MODEL=huginn ./shells/_submit.sh shells/paper/ablations/profile_nsys.sh
#
# DATASET picks the replayed workload (sharegpt by default):
#   MODEL=ouro DATASET=alpaca ./shells/_submit.sh shells/paper/ablations/profile_nsys.sh

set -euo pipefail
source "${REPO_ROOT:?submit with ./shells/_submit.sh}/shells/paper/_common.sh"

NSYS_BIN="${NSYS_BIN:-nsys}"
command -v "$NSYS_BIN" >/dev/null || { echo "ERROR: nsys not found on PATH (source the CUDA env or set NSYS_BIN)"; exit 1; }

MODEL="${MODEL:?set MODEL to ouro or huginn}"
DATASET="${DATASET:-sharegpt}"

case "$MODEL" in
    ouro)
        MODEL_ID="$OURO_1_4B_MODEL_ID"
        RECUR_STEPS=4
        THRESHOLD=0.2 # earliest threshold from exit sweep
        MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-1}" # immediate coda (the default), as in the throughput sweeps
        LAYER_SPLITS="-" # Ouro has no prelude-core-coda split to ablate
        ;;
    huginn)
        MODEL_ID="$HUGINN_MODEL_ID"
        RECUR_STEPS=16
        THRESHOLD=0.28 # earliest threshold from exit sweep
        MIN_CODA_BATCHES="${MIN_CODA_BATCHES:-quarter}" # 'quarter' resolves against each maximum decode batch size
        LAYER_SPLITS="${LAYER_SPLITS:-2-4-2}" # released split; override to trace 0-4-0 / 1-4-1
        ;;
    *) echo "unknown MODEL '$MODEL' (expected: ouro, huginn)" >&2; exit 1 ;;
esac

WORKLOAD="$(workload_for "$MODEL" "$DATASET")"

BACKENDS="${BACKENDS:-cb cdb-norefill cdb-refill}"
WIDTHS="${WIDTHS:-64 512}" # Ablation width plus a near-uncapped wide regime
LIMIT="${LIMIT:-$(num_requests_for "$DATASET")}"
ANALYSIS_WINDOW="${ANALYSIS_WINDOW:-full}"

case "$ANALYSIS_WINDOW" in
    steady) MEASURED_LABEL=benchmark.steady ;;
    full) MEASURED_LABEL=benchmark.generate ;;
    *) echo "unknown ANALYSIS_WINDOW '$ANALYSIS_WINDOW' (expected: steady or full)" >&2; exit 1 ;;
esac

MIN_FREE_SLOTS="${MIN_FREE_SLOTS:--}"

WORKLOAD_STEM="$(basename "${WORKLOAD%.json}")"
THRESH_TAG="q$(printf '%s' "$THRESHOLD" | tr '.' 'p')"

echo "Experiment: nsys trace (replay benchmark), $MODEL_ID, $DATASET, q=$THRESHOLD"
echo "Backends:   $BACKENDS (min coda batches: $MIN_CODA_BATCHES, layer splits: $LAYER_SPLITS)"
echo "Max decode batch sizes: $WIDTHS (min free slots: $MIN_FREE_SLOTS)"
echo "Analysis:   $ANALYSIS_WINDOW window ($MEASURED_LABEL)"
echo "nsys:       $("$NSYS_BIN" --version | head -1)"
echo ""

for width in $WIDTHS; do
  for split in $LAYER_SPLITS; do
    split_tag=""
    [[ "$split" != "-" ]] && split_tag="_ls${split}"
    OUTPUT_DIR="outputs/nsys/${WORKLOAD_STEM}_${THRESH_TAG}${split_tag}_w${width}"
    mkdir -p "$OUTPUT_DIR"
    limit="$LIMIT"

    for backend in $BACKENDS; do
      # Only cdb-refill batches codas; the other backends run once per (width, split).
      min_codas="-"
      [[ "$backend" == cdb-refill ]] && min_codas="$MIN_CODA_BATCHES"
      for min_coda in $min_codas; do
        if [[ "$min_coda" == quarter ]]; then
          min_coda=$((width / 4))
        fi
        if [[ "$min_coda" != "-" ]] && ((min_coda < 1)); then
          min_coda=1
        fi
        for min_free in $MIN_FREE_SLOTS; do
          run_tag="$backend"
          [[ "$min_coda" != "-" ]] && run_tag="${backend}_k${min_coda}"
          [[ "$min_free" != "-" ]] && run_tag="${run_tag}_s${min_free}"
          echo "=== width: $width  split: $split  backend: $backend  min coda batch: $min_coda  min free slots: $min_free  limit: $limit ==="
          analysis_stem="${run_tag}_${ANALYSIS_WINDOW}"
          rep="$OUTPUT_DIR/${analysis_stem}.nsys-rep"
          sqlite="$OUTPUT_DIR/${analysis_stem}.sqlite"
          analysis_json="$OUTPUT_DIR/${analysis_stem}_analysis.json"
          analysis_csv="$OUTPUT_DIR/${analysis_stem}_analysis.csv"
          summary="$OUTPUT_DIR/${analysis_stem}_summary.jsonl"

          mapfile -t args < <(engine_flags "$backend")
          # engine_flags exits inside the process substitution, so an unknown backend
          # surfaces here as an empty array rather than a failed command.
          (( ${#args[@]} )) || exit 1
          args+=(
              --model "$MODEL_ID"
              --workload "$WORKLOAD"
              --exit-threshold "$THRESHOLD"
              --max-recurrent-depth "$RECUR_STEPS"
              --max-model-len "$MAX_MODEL_LEN"
              --max-num-seqs "$width"
              --attn-implementation "$ATTN_IMPL"
              --num-blocks "$NUM_BLOCKS"
              --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
              --block-size "$BLOCK_SIZE"
              --limit "$limit"
              # Benchmark rows measured under profiling overhead are quarantined here
              # instead of outputs/bench/.
              --summary-output "$summary"
              --nvtx
          )
          [[ "$split" != "-" ]] && args+=(--layer-split "$split")
          [[ "$min_coda" != "-" ]] && args+=(--min-coda-batch-size "$min_coda")
          [[ "$min_free" != "-" ]] && args+=(--min-free-slots "$min_free")

          "$NSYS_BIN" profile \
              --trace=cuda,nvtx \
              --sample=none \
              --cpuctxsw=none \
              --cuda-graph-trace=graph \
              --cuda-event-trace=false \
              --output="$rep" \
              --force-overwrite=true \
              python scripts/benchmark_throughput.py "${args[@]}"

          echo "--- exporting $sqlite"
          "$NSYS_BIN" export --type=sqlite --output="$sqlite" --force-overwrite=true "$rep"

          echo "--- analyzing $ANALYSIS_WINDOW window"
          python scripts/analyze_paper_nsys.py "$sqlite" \
              --label "${analysis_stem}_${WORKLOAD_STEM}_${THRESH_TAG}${split_tag}_w${width}" \
              --measured-label "$MEASURED_LABEL" \
              --output-json "$analysis_json" \
              --output-csv "$analysis_csv"
          echo ""
        done
      done
    done
  done
done

echo "Done. Traces and analyses in outputs/nsys/"
