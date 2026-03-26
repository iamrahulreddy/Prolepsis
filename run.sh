#!/bin/bash
set -euo pipefail

RESULTS_DIR="prolepsis_results"
RESULTS_ARCHIVE="${RESULTS_DIR}_download.zip"

echo ""
echo "  =========================================="
echo "  Prolepsis setup and execution"
echo "  =========================================="
echo ""

# 1. Install
echo "  [1/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -e ".[benchmark,vis]" -q
pip install pytest -q

# 2. Tests (run first — fail fast before burning GPU on benchmark)
echo "  [2/5] Running test suite..."
python -m pytest tests/ -v -x --tb=short --color=no 2>&1 | tee test_log.txt

# Keep the test transcript next to the saved benchmark artifacts.
mkdir -p "$RESULTS_DIR"
cp test_log.txt "$RESULTS_DIR/" 2>/dev/null || true

# 3. Prompt generation
echo "  [3/5] Generating testing prompt suite..."
python scripts/generate_prompts.py

# 4. Benchmark
echo "  [4/5] Running benchmark..."
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
python -m benchmark.run_benchmark \
    --draft-model Qwen/Qwen3-1.7B \
    --target-model Qwen/Qwen3-8B \
    --num-prompts 60 \
    --prompt-file robust_prompts.json \
    --output-len 1024 \
    --save-dir "$RESULTS_DIR" \
    --gamma 5 \
    --temperature 0.6 \
    --warmup 3 \
    --runs 1 \
    --dtype bfloat16 \
    --synchronize-timing \
    --visualize \
    --detailed-plots

# 5. Package
echo "  [5/5] Packaging artifacts..."
python scripts/package_results_archive.py \
    --results-dir "$RESULTS_DIR" \
    --output "$RESULTS_ARCHIVE"

echo ""
echo "  =========================================="
echo "  Done. Results in $RESULTS_DIR/"
echo "  Download archive: $RESULTS_ARCHIVE"
for artifact in \
    "benchmark_dashboard.png" \
    "benchmark_throughput.png" \
    "benchmark_latency.png" \
    "benchmark_memory.png" \
    "benchmark_summary.txt" \
    "benchmark_results.json" \
    "prompt_and_responses.txt" \
    "system_info.txt" \
    "test_log.txt"
do
    if [ -f "$RESULTS_DIR/$artifact" ]; then
        echo "  Saved: $RESULTS_DIR/$artifact"
    fi
done
if [ -d "$RESULTS_DIR/telemetry" ]; then
    echo "  Saved: $RESULTS_DIR/telemetry/"
fi
echo "  =========================================="
echo ""
