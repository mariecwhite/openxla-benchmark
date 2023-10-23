#!/bin/bash

IREE_SOURCE_DIR=/tmp/iree
ANDROID_PLATFORM_VERSION="33"

if (( BUILD_IREE == 1)); then
  rm -rf "${IREE_SOURCE_DIR}"
  mkdir -p "${IREE_SOURCE_DIR}"

  pushd "${IREE_SOURCE_DIR}"

  git clone https://github.com/openxla/iree.git
  cd iree
  git submodule update --init

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

  popd # IREE_SOURCE_DIR.
fi

ROOT_DIR=/tmp/benchmark_artifacts
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
  "BERT_BASE_FP32_JAX_I32_SEQLEN1024"
  "BERT_BASE_FP32_JAX_I32_SEQLEN2048"
  "BERT_BASE_FP32_JAX_I32_SEQLEN3072"
  "BERT_BASE_FP32_JAX_I32_SEQLEN4096"

  "BERT_BASE_FP16_JAX_I32_SEQLEN8"
  "BERT_BASE_FP16_JAX_I32_SEQLEN32"
  "BERT_BASE_FP16_JAX_I32_SEQLEN64"
  "BERT_BASE_FP16_JAX_I32_SEQLEN128"
  "BERT_BASE_FP16_JAX_I32_SEQLEN256"
  "BERT_BASE_FP16_JAX_I32_SEQLEN512"
  "BERT_BASE_FP16_JAX_I32_SEQLEN1024"
  "BERT_BASE_FP16_JAX_I32_SEQLEN2048"
  "BERT_BASE_FP16_JAX_I32_SEQLEN3072"
  "BERT_BASE_FP16_JAX_I32_SEQLEN4096"

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
    gsutil -m cp -r "${BASE_GCS_DIR}/${model_name}" "${MODEL_DIR}/"

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
X86_TARGET_CPU="cascadelake"
#ARM_CPU_FEATURES="x3"
# For Pixel-8 Pro Big Cores.
ARM_TARGET_CPU="cortex-a715"
#ARM_CPU_FEATURES="cortex-a510"

if (( COMPILE_MODELS == 1 )); then
  rm -rf "${COMPILED_ARTIFACTS_PATH}"
  mkdir -p "${COMPILED_ARTIFACTS_PATH}"

  for model_name in "${MODELS[@]}"; do
    # Compile for x86 first.
    OUTPUT_PATH="${COMPILED_ARTIFACTS_PATH}/x86_64/${model_name}"
    mkdir -p "${OUTPUT_PATH}"

    echo "Compiling ${model_name}/stablehlo.mlirbc with default flags for x86_64."
    "${IREE_COMPILE_PATH}" "${MODEL_DIR}/${model_name}/stablehlo.mlirbc" \
      --iree-llvmcpu-target-cpu="${X86_TARGET_CPU}" \
      --iree-llvmcpu-target-cpu-features="host" \
      --iree-hal-target-backends=llvm-cpu \
      --iree-input-type=stablehlo \
      --iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu \
      --iree-llvmcpu-debug-symbols=false \
      --iree-vm-bytecode-module-strip-source-map=true \
      --iree-vm-emit-polyglot-zip=false \
      -o "${OUTPUT_PATH}/module_default.vmfb"

    echo "Compiling ${model_name}/stablehlo.mlirbc with experimental flags for x86_64."
    "${IREE_COMPILE_PATH}" "${MODEL_DIR}/${model_name}/stablehlo.mlirbc" \
      --iree-llvmcpu-target-cpu="${X86_TARGET_CPU}" \
      --iree-llvmcpu-target-cpu-features="host" \
      --iree-hal-target-backends=llvm-cpu \
      --iree-input-type=stablehlo \
      --iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu \
      --iree-llvmcpu-debug-symbols=false \
      --iree-vm-bytecode-module-strip-source-map=true \
      --iree-vm-emit-polyglot-zip=false \
      --iree-opt-data-tiling \
      --iree-llvmcpu-enable-microkernels \
      -o "${OUTPUT_PATH}/module_experimental.vmfb"

    cp -r "${MODEL_DIR}/${model_name}/inputs_npy" "${OUTPUT_PATH}/"
    cp -r "${MODEL_DIR}/${model_name}/outputs_npy" "${OUTPUT_PATH}/"

    # Compile for Android first.
    OUTPUT_PATH="${COMPILED_ARTIFACTS_PATH}/arm64/${model_name}"
    mkdir -p "${OUTPUT_PATH}"

    echo "Compiling ${model_name}/stablehlo.mlirbc with default flags for android."
    "${IREE_COMPILE_PATH}" "${MODEL_DIR}/${model_name}/stablehlo.mlirbc" \
      --iree-llvmcpu-target-cpu="${ARM_TARGET_CPU}" \
      --iree-hal-target-backends=llvm-cpu \
      --iree-input-type=stablehlo \
      --iree-llvmcpu-target-triple=aarch64-none-linux-android${ANDROID_PLATFORM_VERSION} \
      --iree-llvmcpu-debug-symbols=false \
      --iree-vm-bytecode-module-strip-source-map=true \
      --iree-vm-emit-polyglot-zip=false \
      -o "${OUTPUT_PATH}/module_default.vmfb"

    echo "Compiling ${model_name}/stablehlo.mlirbc with experimental flags for android."
    "${IREE_COMPILE_PATH}" "${MODEL_DIR}/${model_name}/stablehlo.mlirbc" \
      --iree-llvmcpu-target-cpu="${ARM_TARGET_CPU}" \
      --iree-hal-target-backends=llvm-cpu \
      --iree-input-type=stablehlo \
      --iree-llvmcpu-target-triple=aarch64-none-linux-android${ANDROID_PLATFORM_VERSION} \
      --iree-llvmcpu-debug-symbols=false \
      --iree-vm-bytecode-module-strip-source-map=true \
      --iree-vm-emit-polyglot-zip=false \
      --iree-opt-data-tiling \
      --iree-llvmcpu-enable-microkernels \
      -o "${OUTPUT_PATH}/module_experimental.vmfb"

    cp -r "${MODEL_DIR}/${model_name}/inputs_npy" "${OUTPUT_PATH}/"
    cp -r "${MODEL_DIR}/${model_name}/outputs_npy" "${OUTPUT_PATH}/"
  done
fi

TD="$(cd $(dirname $0) && pwd)"
OUTPUT_CSV="${ROOT_DIR}/output.csv"

for model_name in "${MODELS[@]}"; do
  ARTIFACT_DIR="${COMPILED_ARTIFACTS_PATH}/x86_64/${model_name}"

  echo "Benchmarking ${model_name}..."
  python3 "${TD}/run_benchmarks.py" -o "${OUTPUT_CSV}" --artifact_dir "${ARTIFACT_DIR}" --iree_run_module_path "${IREE_RUN_MODULE_PATH}" --iree_benchmark_module_path "${IREE_BENCHMARK_MODULE_PATH}" --threads "1,8,16,32,64,128"
done
