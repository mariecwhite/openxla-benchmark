#!/bin/bash

ROOT_DIR=/tmp/benchmark_artifacts_tflite
mkdir -p "${ROOT_DIR}"

# Download models.
BASE_GCS_DIR="gs://iree-model-artifacts/tflite/tflite_models_1698315913"
declare -a MODELS=(
  "BERT_BASE_FP32_TFLITE_I32_SEQLEN8"
  "BERT_BASE_FP32_TFLITE_I32_SEQLEN32"
  "BERT_BASE_FP32_TFLITE_I32_SEQLEN64"
  "BERT_BASE_FP32_TFLITE_I32_SEQLEN128"
  "BERT_BASE_FP32_TFLITE_I32_SEQLEN256"
  "BERT_BASE_FP32_TFLITE_I32_SEQLEN512"

  "BERT_BASE_FP16_TFLITE_I32_SEQLEN8"
  "BERT_BASE_FP16_TFLITE_I32_SEQLEN32"
  "BERT_BASE_FP16_TFLITE_I32_SEQLEN64"
  "BERT_BASE_FP16_TFLITE_I32_SEQLEN128"
  "BERT_BASE_FP16_TFLITE_I32_SEQLEN256"
  "BERT_BASE_FP16_TFLITE_I32_SEQLEN512"

  "BERT_BASE_DYN_QUANT_TFLITE_I32_SEQLEN8"
  "BERT_BASE_DYN_QUANT_TFLITE_I32_SEQLEN32"
  "BERT_BASE_DYN_QUANT_TFLITE_I32_SEQLEN64"
  "BERT_BASE_DYN_QUANT_TFLITE_I32_SEQLEN128"
  "BERT_BASE_DYN_QUANT_TFLITE_I32_SEQLEN256"
  "BERT_BASE_DYN_QUANT_TFLITE_I32_SEQLEN512"

  "BERT_BASE_INT8_TFLITE_I32_SEQLEN8"
  "BERT_BASE_INT8_TFLITE_I32_SEQLEN32"
  "BERT_BASE_INT8_TFLITE_I32_SEQLEN64"
  "BERT_BASE_INT8_TFLITE_I32_SEQLEN128"
  "BERT_BASE_INT8_TFLITE_I32_SEQLEN256"
  "BERT_BASE_INT8_TFLITE_I32_SEQLEN512"

  "VIT_CLASSIFICATION_FP32_TFLITE_3X224X224XF32"
  "VIT_CLASSIFICATION_FP16_TFLITE_3X224X224XF32"
  "VIT_CLASSIFICATION_DYN_QUANT_TFLITE_3X224X224XF32"
  "VIT_CLASSIFICATION_INT8_TFLITE_3X224X224XINT8"
)

MODEL_DIR="${ROOT_DIR}/models"

if (( DOWNLOAD_MODELS == 1 )); then
  rm -rf "${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"

  for model_name in "${MODELS[@]}"; do
    mkdir "${MODEL_DIR}/${model_name}"

    gsutil cp "${BASE_GCS_DIR}/${model_name}/model_fp32.tflite" "${MODEL_DIR}/${model_name}/"
    gsutil cp "${BASE_GCS_DIR}/${model_name}/model_fp16.tflite" "${MODEL_DIR}/${model_name}/"
    gsutil cp "${BASE_GCS_DIR}/${model_name}/model_dynamic_range_quant.tflite" "${MODEL_DIR}/${model_name}/"
    gsutil cp "${BASE_GCS_DIR}/${model_name}/model_int8.tflite" "${MODEL_DIR}/${model_name}/"
  done
fi

TFLITE_BENCHMARK_MODEL="${ROOT_DIR}/benchmark_model"
TFLITE_BENCHMARK_MODEL_FLEX="${ROOT_DIR}/benchmark_model_plus_flex"

TD="$(cd $(dirname $0) && pwd)"
OUTPUT_CSV="${ROOT_DIR}/output.csv"
rm "${OUTPUT_CSV}"

for model_name in "${MODELS[@]}"; do
  ARTIFACT_DIR="${MODEL_DIR}/${model_name}"

  echo "Benchmarking ${model_name}..."
  python3 "${TD}/run_benchmarks_tflite.py" -o "${OUTPUT_CSV}" --artifact_dir "${ARTIFACT_DIR}" --benchmark_model_path "${TFLITE_BENCHMARK_MODEL}" --benchmark_model_flex_path "${TFLITE_BENCHMARK_MODEL_FLEX}" --threads "1,4,5" --tasksets "100,F0,1F0"
done
