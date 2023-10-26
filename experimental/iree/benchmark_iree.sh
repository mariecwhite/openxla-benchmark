#!/bin/bash

# Environment variables:
#   BUILD_IREE=1
#   DOWNLOAD_MODELS=1
#   COMPILE_MODELS=1
#   COMPILE_ANDROID=1

IREE_SOURCE_DIR=/tmp/iree
ANDROID_PLATFORM_VERSION="34"

if (( BUILD_IREE == 1 )); then
  #rm -rf "${IREE_SOURCE_DIR}"
  #mkdir -p "${IREE_SOURCE_DIR}"

  pushd "${IREE_SOURCE_DIR}"

  #git clone https://github.com/openxla/iree.git
  cd iree
  #git submodule update --init

  # Build local.
  cmake -G Ninja -B ../iree-build/ -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_BUILD_TESTS=OFF \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_ASSERTIONS=OFF \
    -DIREE_BUILD_BINDINGS_TFLITE=OFF \
    -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF \
    -DIREE_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=../iree-build/install
  cmake --build ../iree-build/
  cmake --build ../iree-build/ --target install

  if (( COMPILE_ANDROID == 1 )); then
    # Build Android.
    cmake -GNinja -B ../iree-build-android/ -S . \
      -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
      -DIREE_HOST_BIN_DIR="$PWD/../iree-build/install/bin" \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_PLATFORM="android-${ANDROID_PLATFORM_VERSION}" \
      -DIREE_BUILD_COMPILER=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DIREE_BUILD_PYTHON_BINDINGS=OFF \
      -DIREE_BUILD_SAMPLES=OFF \
      -DIREE_BUILD_TESTS=OFF \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DIREE_ENABLE_ASSERTIONS=OFF \
      -DIREE_BUILD_BINDINGS_TFLITE=OFF \
      -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF \
      -DIREE_ENABLE_LLD=ON
    cmake --build ../iree-build-android/
  fi

  popd # IREE_SOURCE_DIR.
fi

ROOT_DIR=/tmp/benchmark_artifacts
mkdir -p "${ROOT_DIR}"

# Download models.
JAX_GCS_DIR="gs://iree-model-artifacts/jax/jax_models_0.4.19_1698302455"
declare -a JAX_MODELS=(
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

  "SD_PIPELINE_FP32_JAX_64XI32_BATCH1"
  "SD_PIPELINE_FP32_JAX_64XI32_BATCH8"

  "SD_PIPELINE_FP16_JAX_64XI32_BATCH1"
  "SD_PIPELINE_FP16_JAX_64XI32_BATCH8"
)

TFLITE_GCS_DIR="gs://iree-model-artifacts/tflite/tflite_models_1698315913"
declare -a TFLITE_MODELS=(
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
  #rm -rf "${MODEL_DIR}"
  #mkdir -p "${MODEL_DIR}"

  # Download JAX models.
  for model_name in "${JAX_MODELS[@]}"; do
    gsutil -m cp -r "${JAX_GCS_DIR}/${model_name}" "${MODEL_DIR}/"

    tar zxf "${MODEL_DIR}/${model_name}/inputs_npy.tgz" --one-top-level --directory "${MODEL_DIR}/${model_name}"
    tar zxf "${MODEL_DIR}/${model_name}/outputs_npy.tgz" --one-top-level --directory "${MODEL_DIR}/${model_name}"

    rm "${MODEL_DIR}/${model_name}/inputs_npy.tgz"
    rm "${MODEL_DIR}/${model_name}/outputs_npy.tgz"
  done

  # Download TFLite models.
  for model_name in "${TFLITE_MODELS[@]}"; do
    gsutil -m cp -r "${TFLITE_GCS_DIR}/${model_name}" "${MODEL_DIR}/"

    tar zxf "${MODEL_DIR}/${model_name}/inputs_npy.tgz" --one-top-level --directory "${MODEL_DIR}/${model_name}"
    tar zxf "${MODEL_DIR}/${model_name}/outputs_npy.tgz" --one-top-level --directory "${MODEL_DIR}/${model_name}"

    rm "${MODEL_DIR}/${model_name}/inputs_npy.tgz"
    rm "${MODEL_DIR}/${model_name}/outputs_npy.tgz"
  done
fi

IREE_COMPILE_PATH="${IREE_SOURCE_DIR}/iree-build/tools/iree-compile"
IREE_RUN_MODULE_PATH="${IREE_SOURCE_DIR}/iree-build/tools/iree-run-module"
IREE_BENCHMARK_MODULE_PATH="${IREE_SOURCE_DIR}/iree-build/tools/iree-benchmark-module"

COMPILED_ARTIFACTS_PATH="${ROOT_DIR}/compiled"

# Change this depending on the target CPU.
#X86_CPU_FEATURES="host"
#X86_TARGET_CPU="znver2"
#X86_TARGET_CPU="cascadelake"
#ARM_CPU_FEATURES="x3"
# For Pixel-8 Pro Big Cores.
ARM_TARGET_CPU="cortex-a715"
#ARM_CPU_FEATURES="cortex-a510"
#ARM_CPU_FEATURES="+fp-armv8,+lse,+fp16,+fp16fml,+dotprod,+i8mm,+sve,+sve2,+sve2-bitperm"
#ARM_CPU_FEATURES="+fp-armv8,+lse,+fp16,+fp16fml,+dotprod,+i8mm"
#ARM_CPU_FEATURES="+fp-armv8,+lse,+fp16fml,+dotprod,+i8mm"
#ARM_CPU_FEATURES="+v9a,+fullfp16,fp-armv8,+neon,+aes,+sha2,+crc,+lse,+rdm,+complxnum,+rcpc,+sha3,+sm4,+dotprod,+sve,+fp16fml,+dit,+flagm,+ssbs,+sb,+sve2,+sve2-aes,+sve2-bitperm,+sve2-sha3,+sve2-sm4,+altnzcv,+fptoint,+bf16,+i8mm,+bti"
ARM_CPU_FEATURES="+v9a,+fullfp16,fp-armv8,+neon,+aes,+sha2,+crc,+lse,+rdm,+complxnum,+rcpc,+sha3,+sm4,+dotprod,+fp16fml,+dit,+flagm,+ssbs,+sb,+altnzcv,+fptoint,+bf16,+i8mm,+bti"


compile() {
  model_name="$1"
  dialect="$2"
  quantized=$3

  # Compile for x86.
  OUTPUT_PATH="${COMPILED_ARTIFACTS_PATH}/x86_64/${model_name}"
  mkdir -p "${OUTPUT_PATH}"

  declare -a common_args=(
    "${MODEL_DIR}/${model_name}/${dialect}.mlirbc"
    --iree-hal-target-backends=llvm-cpu
    --iree-input-type="${dialect}"
    --iree-llvmcpu-link-embedded=false
    --iree-input-demote-f64-to-f32=false
    --iree-input-demote-i64-to-i32=false
    --iree-llvmcpu-debug-symbols=false
    --iree-vm-bytecode-module-strip-source-map=true
    --iree-vm-emit-polyglot-zip=false
  )

  declare -a experimental_args=(
    --iree-opt-data-tiling
    --iree-llvmcpu-enable-microkernels
  )

  declare -a x86_common_args=(
    #--iree-llvmcpu-target-cpu="${X86_TARGET_CPU}"
    --iree-llvmcpu-target-cpu-features="host"
    --iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu
  )

  echo "Compiling ${model_name}/${dialect}.mlirbc with default flags for x86_64."
  declare -a x86_default_args=(
    "${common_args[@]}"
    "${x86_common_args[@]}"
    -o "${OUTPUT_PATH}/module_default.vmfb"
  )
  #"${IREE_COMPILE_PATH}" "${x86_default_args[@]}"

  echo "Compiling ${model_name}/${dialect}.mlirbc with experimental flags for x86_64."
  declare -a x86_experimental_args=(
    "${common_args[@]}"
    "${x86_common_args[@]}"
    "${experimental_args[@]}"
    -o "${OUTPUT_PATH}/module_experimental.vmfb"
  )

  #"${IREE_COMPILE_PATH}" "${x86_experimental_args[@]}"

  cp -r "${MODEL_DIR}/${model_name}/inputs_npy" "${OUTPUT_PATH}/"
  cp -r "${MODEL_DIR}/${model_name}/outputs_npy" "${OUTPUT_PATH}/"

  if (( COMPILE_ANDROID == 1 )); then
    # Compile for Android.
    OUTPUT_PATH="${COMPILED_ARTIFACTS_PATH}/arm64/${model_name}"
    mkdir -p "${OUTPUT_PATH}"

    declare -a arm64_common_args=(
      --iree-llvmcpu-target-cpu-features="${ARM_CPU_FEATURES}"
      --iree-llvmcpu-target-triple=aarch64-none-linux-android${ANDROID_PLATFORM_VERSION}
    )

    echo "Compiling ${model_name}/${dialect}.mlirbc with default flags for android."
    declare -a arm64_default_args=(
      "${common_args[@]}"
      "${arm64_common_args[@]}"
      -o "${OUTPUT_PATH}/module_default.vmfb"
    )

    "${IREE_COMPILE_PATH}" "${arm64_default_args[@]}"

    echo "Compiling ${model_name}/${dialect}.mlirbc with experimental flags for android."
    declare -a arm64_experimental_args=(
      "${common_args[@]}"
      "${arm64_common_args[@]}"
      "${experimental_args[@]}"
      -o "${OUTPUT_PATH}/module_experimental.vmfb"
    )

    "${IREE_COMPILE_PATH}" "${arm64_experimental_args[@]}"

    cp -r "${MODEL_DIR}/${model_name}/inputs_npy" "${OUTPUT_PATH}/"
    cp -r "${MODEL_DIR}/${model_name}/outputs_npy" "${OUTPUT_PATH}/"
  fi
}

if (( COMPILE_MODELS == 1 )); then
  rm -rf "${COMPILED_ARTIFACTS_PATH}"
  mkdir -p "${COMPILED_ARTIFACTS_PATH}"

  for model_name in "${JAX_MODELS[@]}"; do
    compile ${model_name} "stablehlo" false
  done

  for model_name in "${TFLITE_MODELS[@]}"; do
    compile ${model_name} "tosa" false
  done
fi

TD="$(cd $(dirname $0) && pwd)"
OUTPUT_CSV="${ROOT_DIR}/output.csv"
rm "${OUTPUT_CSV}"

#ATOL=1e-5
#RTOL=1e-8
ATOL=0.5
RTOL=0.5

python3 -m venv benchmarks.venv
source benchmarks.venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade numpy

for model_name in "${JAX_MODELS[@]}"; do
  ARTIFACT_DIR="${COMPILED_ARTIFACTS_PATH}/x86_64/${model_name}"

  echo "Benchmarking ${model_name}..."
  python3 "${TD}/run_benchmarks.py" -o "${OUTPUT_CSV}" --artifact_dir "${ARTIFACT_DIR}" --iree_run_module_path "${IREE_RUN_MODULE_PATH}" --iree_benchmark_module_path "${IREE_BENCHMARK_MODULE_PATH}" --threads "1,4,8,16,32,64,128,256" --atol "${ATOL}" --rtol "${RTOL}"
done

for model_name in "${TFLITE_MODELS[@]}"; do
  ARTIFACT_DIR="${COMPILED_ARTIFACTS_PATH}/x86_64/${model_name}"

  echo "Benchmarking ${model_name}..."
  python3 "${TD}/run_benchmarks.py" -o "${OUTPUT_CSV}" --artifact_dir "${ARTIFACT_DIR}" --iree_run_module_path "${IREE_RUN_MODULE_PATH}" --iree_benchmark_module_path "${IREE_BENCHMARK_MODULE_PATH}" --threads "1,4,8,16,32,64,128,256" --atol "${ATOL}" --rtol "${RTOL}"
done
