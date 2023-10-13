#!/bin/bash

ROOT_DIR=/tmp/benchmark_artifacts_tflite
mkdir -p "${ROOT_DIR}"

# Download models.
BASE_GCS_DIR="gs://iree-model-artifacts/jax/jax_models_0.4.18_1697544748"
declare -a MODELS=(
  "BERT_BASE_FP32_JAX_I32_SEQLEN8"
  "BERT_BASE_FP32_JAX_I32_SEQLEN32"
  "BERT_BASE_FP32_JAX_I32_SEQLEN64"
  "BERT_BASE_FP32_JAX_I32_SEQLEN128"
  "BERT_BASE_FP32_JAX_I32_SEQLEN256"
  "BERT_BASE_FP32_JAX_I32_SEQLEN512"

  "BERT_BASE_FP16_JAX_I32_SEQLEN8"
  "BERT_BASE_FP16_JAX_I32_SEQLEN32"
  "BERT_BASE_FP16_JAX_I32_SEQLEN64"
  "BERT_BASE_FP16_JAX_I32_SEQLEN128"
  "BERT_BASE_FP16_JAX_I32_SEQLEN256"
  "BERT_BASE_FP16_JAX_I32_SEQLEN512"

  "BERT_BASE_BF16_JAX_I32_SEQLEN8"
  "BERT_BASE_BF16_JAX_I32_SEQLEN32"
  "BERT_BASE_BF16_JAX_I32_SEQLEN64"
  "BERT_BASE_BF16_JAX_I32_SEQLEN128"
  "BERT_BASE_BF16_JAX_I32_SEQLEN256"
  "BERT_BASE_BF16_JAX_I32_SEQLEN512"

  "T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN16"
  "T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN32"
  "T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN64"
  "T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN128"
  "T5_4CG_SMALL_FP32_JAX_1X128XI32_GEN256"

  "VIT_CLASSIFICATION_JAX_3X224X224XF32"
)

MODEL_DIR="${ROOT_DIR}/models"

if (( DOWNLOAD_MODELS == 1 )); then
  rm -rf "${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"

  for model_name in "${MODELS[@]}"; do
    mkdir "${MODEL_DIR}/${model_name}"

    gsutil cp "${BASE_GCS_DIR}/${model_name}/model_fp32_stablehlo.tflite" "${MODEL_DIR}/${model_name}/"
    gsutil cp "${BASE_GCS_DIR}/${model_name}/model_fp32.tflite" "${MODEL_DIR}/${model_name}/"
  done
fi

TFLITE_BENCHMARK_MODEL="${ROOT_DIR}/benchmark_model"
TFLITE_BENCHMARK_MODEL_FLEX="${ROOT_DIR}/benchmark_model_plus_flex"

TD="$(cd $(dirname $0) && pwd)"
OUTPUT_CSV="${ROOT_DIR}/output.csv"

for model_name in "${MODELS[@]}"; do
  ARTIFACT_DIR="${MODEL_DIR}/${model_name}"

  echo "Benchmarking ${model_name}..."
  python3 "${TD}/run_benchmarks_tflite.py" -o "${OUTPUT_CSV}" --artifact_dir "${ARTIFACT_DIR}" --benchmark_model_path "${TFLITE_BENCHMARK_MODEL}" --benchmark_model_flex_path "${TFLITE_BENCHMARK_MODEL_FLEX}" --threads "1,8,16,32,64,128"
done
