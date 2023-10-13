# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numbers
import numpy as np
import flax
import tensorflow as tf
from jax.experimental import jax2tf
from collections.abc import Callable, Sequence

import argparse
import concurrent.futures
import functools
import jax
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any, List, Optional

# Add openxla dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parents[5]))
from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite.jax import model_definitions
from openxla.benchmark.models import utils
from openxla.benchmark.models.jax.bert import bert_model

HLO_FILENAME_REGEX = r".*jit_forward.before_optimizations.txt"
GCS_UPLOAD_DIR = os.getenv("GCS_UPLOAD_DIR", "gs://iree-model-artifacts/jax")


def _generate_mlir(jit_function: Any, jit_inputs: Any, model_dir: pathlib.Path,
                   iree_ir_tool: Optional[pathlib.Path]):
  mlir = jit_function.lower(*jit_inputs).compiler_ir(dialect="stablehlo")
  mlir_path = model_dir / "stablehlo.mlir"
  print(f"Saving mlir to {mlir_path}")
  with open(mlir_path, "w") as f:
    f.write(str(mlir))

  if iree_ir_tool:
    binary_mlir_path = model_dir / "stablehlo.mlirbc"
    subprocess.run(
        [
            iree_ir_tool, "cp", "--emit-bytecode", mlir_path, "-o",
            binary_mlir_path
        ],
        check=True,
    )
    mlir_path.unlink()


def _generate_tflite(model_obj: Any, inputs: Any, model_dir: pathlib.Path,
                     export_types: List[def_types.ModelArtifactType]):
  try:

    def predict(*args):
      return model_obj.apply(*args)

    input_signature = []
    for input in inputs:
      input_signature.append(
          tf.TensorSpec(shape=input.shape, dtype=tf.as_dtype(input.dtype)))

    tf_predict = tf.function(jax2tf.convert(predict,
                                            enable_xla=False,
                                            with_gradient=False),
                             input_signature=input_signature,
                             autograph=False)
    tf_predict(*inputs)

  except Exception as e:
    print(f"Failed to convert Flax model to TF {e}")
    return

  def create_converter(tf_predict_fn):
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict_fn.get_concrete_function()], tf_predict_fn)
    converter._experimental_disable_per_channel = True
    converter._experimental_use_buffer_offset = True
    converter.exclude_conversion_metadata = True
    return converter

  if def_types.ModelArtifactType.TFLITE_FP32_STABLEHLO in export_types:
    # Generate FP32 TFLite model with StableHLO ops.
    try:
      converter = create_converter(tf_predict)
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.EXPERIMENTAL_STABLEHLO_OPS,
      ]
      tflite_model = converter.convert()
      tflite_model_path = model_dir / "model_fp32_stablehlo.tflite"
      tflite_model_path.write_bytes(tflite_model)
      print(f"Successfully generated {tflite_model_path.name}")
    except Exception as e:
      print(f"Failed to generate fp32 StableHLO TFLite model. Exception: {e}")

  if def_types.ModelArtifactType.TFLITE_FP32 in export_types:
    # Generate FP32 TFLite model with TFLite and TF ops.
    try:
      converter = create_converter(tf_predict)
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS,
          tf.lite.OpsSet.SELECT_TF_OPS,
      ]
      tflite_model = converter.convert()
      tflite_model_path = model_dir / "model_fp32.tflite"
      tflite_model_path.write_bytes(tflite_model)
      print(f"Successfully generated {tflite_model_path.name}")
    except Exception as e:
      print(f"Failed to generate fp32 TFLite model. Exception: {e}")

  # Below we run post-training quantization using the guide: https://www.tensorflow.org/lite/performance/model_optimization

  if def_types.ModelArtifactType.TFLITE_FP16 in export_types:
    # Generate FP16 TFLite model.
    try:
      converter = create_converter(tf_predict)
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS,
          tf.lite.OpsSet.SELECT_TF_OPS,
      ]
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.target_spec.supported_types = [tf.float16]
      tflite_model = converter.convert()
      tflite_model_path = model_dir / "model_fp16.tflite"
      tflite_model_path.write_bytes(tflite_model)
      print(f"Successfully generated {tflite_model_path.name}")
    except Exception as e:
      print(f"Failed to generate FP16 TFLite model. Exception: {e}")

  # Generate dynamic range quantized TFLite model.
  if def_types.ModelArtifactType.TFLITE_DYNAMIC_RANGE_QUANT in export_types:
    try:
      converter = create_converter(tf_predict)
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS,
          tf.lite.OpsSet.SELECT_TF_OPS,
      ]
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      tflite_model = converter.convert()
      tflite_model_path = model_dir / "model_dynamic_range_quant.tflite"
      tflite_model_path.write_bytes(tflite_model)
      print(f"Successfully generated {tflite_model_path.name}")
    except Exception as e:
      print(
          f"Failed to generate dynamic range quantized TFLite model. Exception: {e}"
      )

  # Generate full integer quantization.
  if def_types.ModelArtifactType.TFLITE_INT8 in export_types:
    try:

      def representative_examples():
        for _ in range(100):
          random_inputs = []
          for input in inputs:
            if issubclass(input.dtype.type, numbers.Integral):
              # If input is integer, use full extents.
              min = np.iinfo(input.dtype.type).min
              max = np.iinfo(input.dtype.type).max
            elif issubclass(input.dtype.type, numbers.Real):
              # If input is float, use values between 0 and 1.
              min = 0
              max = 1
            else:
              raise TypeError(f"Input dtype not supported: {input.dtype}")

            random_inputs.append(
                np.random.uniform(low=min, high=max,
                                  size=input.shape).astype(input.dtype.type))
          yield random_inputs

      converter = create_converter(tf_predict)
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS_INT8
      ]
      converter.target_spec.supported_types = [tf.int8]
      converter.representative_dataset = representative_examples
      converter.inference_type = tf.int8
      converter.inference_input_type = tf.int8
      converter.inference_output_type = tf.int8
      tflite_model_int8 = converter.convert()
      tflite_model_int8_path = model_dir / "model_int8.tflite"
      tflite_model_int8_path.write_bytes(tflite_model_int8)
      print(f"Successfully generated {tflite_model_int8_path.name}")
    except Exception as e:
      print(f"Failed to generate int8 TFLite model. Exception: {e}")


def _generate_artifacts(model: def_types.Model, save_dir: pathlib.Path,
                        iree_ir_tool: Optional[pathlib.Path],
                        auto_upload: bool):
  model_dir = save_dir / model.name
  model_dir.mkdir(exist_ok=True)

  try:
    # Configure to dump hlo.
    if def_types.ModelArtifactType.XLA_HLO_DUMP in model.exported_model_types:
      hlo_dir = model_dir / "hlo"
      hlo_dir.mkdir(exist_ok=True)
      # Only dump hlo for the inference function `jit_model_jitted`.
      os.environ[
          "XLA_FLAGS"] = f"--xla_dump_to={hlo_dir} --xla_dump_hlo_module_re=.*jit_forward.*"

    model_obj = utils.create_model_obj(model)

    inputs = utils.generate_and_save_inputs(model_obj, model_dir)

    jit_inputs = jax.device_put(inputs)
    jit_function = jax.jit(model_obj.forward)
    jit_output_obj = jit_function(*jit_inputs)
    jax.block_until_ready(jit_output_obj)
    output_obj = jax.device_get(jit_output_obj)

    outputs = utils.canonicalize_to_tuple(output_obj)
    utils.save_outputs(outputs, model_dir)

    if def_types.ModelArtifactType.XLA_HLO_DUMP in model.exported_model_types:
      utils.cleanup_hlo(hlo_dir, model_dir, HLO_FILENAME_REGEX)
      os.unsetenv("XLA_FLAGS")

    if def_types.ModelArtifactType.STABLEHLO_MLIR in model.exported_model_types:
      _generate_mlir(jit_function=jit_function,
                     jit_inputs=jit_inputs,
                     model_dir=model_dir,
                     iree_ir_tool=iree_ir_tool)

    if def_types.ModelArtifactType.TFLITE_FP32 in model.exported_model_types:
      _generate_tflite(model_obj=model_obj,
                       inputs=inputs,
                       model_dir=model_dir,
                       export_types=model.exported_model_types)

    print(f"Completed generating artifacts {model.name}\n")

    if auto_upload:
      utils.gcs_upload(str(model_dir),
                       f"{GCS_UPLOAD_DIR}/{save_dir.name}/{model_dir.name}")
      shutil.rmtree(model_dir)

  except Exception as e:
    print(f"Failed to import model {model.name}. Exception: {e}")
    # Remove all generated files.
    shutil.rmtree(model_dir)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generates JAX model artifacts for benchmarking.")
  parser.add_argument("-o",
                      "--output_dir",
                      type=pathlib.Path,
                      required=True,
                      help="Directory to save model artifacts.")
  parser.add_argument("-f",
                      "--filter",
                      dest="filters",
                      nargs="+",
                      default=[".*"],
                      help="The regex patterns to filter model names.")
  parser.add_argument("--iree-ir-tool",
                      "--iree_ir_tool",
                      type=pathlib.Path,
                      default=None,
                      help="Path to `iree-ir-tool`. Used to binarize mlir.")
  parser.add_argument(
      "--auto-upload",
      "--auto_upload",
      action="store_true",
      help=
      f"If set, uploads artifacts automatically to {GCS_UPLOAD_DIR} and removes them locally once uploaded."
  )
  parser.add_argument(
      "-j",
      "--jobs",
      type=int,
      default=1,
      help="Max number of concurrent jobs to generate artifacts. Be cautious"
      " when generating with GPU.")
  return parser.parse_args()


def main(output_dir: pathlib.Path, filters: List[str],
         iree_ir_tool: pathlib.Path, auto_upload: bool, jobs: int):
  combined_filters = "|".join(f"({name_filter})" for name_filter in filters)
  name_pattern = re.compile(f"^{combined_filters}$")
  models = [
      model for model in model_definitions.ALL_MODELS
      if name_pattern.match(model.name)
  ]

  if not models:
    all_models_list = "\n".join(
        model.name for model in model_definitions.ALL_MODELS)
    raise ValueError(f'No model matches "{filters}".'
                     f' Available models:\n{all_models_list}')

  output_dir.mkdir(parents=True, exist_ok=True)

  with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
    future_list = []
    for model in models:
      # We need to generate artifacts in a separate process each time in order for
      # XLA to update the HLO dump directory.
      print(f"Submitting job for model {model}")
      future_list.append(
          executor.submit(_generate_artifacts,
                          model=model,
                          save_dir=output_dir,
                          iree_ir_tool=iree_ir_tool,
                          auto_upload=auto_upload))
    concurrent.futures.wait(future_list)

  if auto_upload:
    utils.gcs_upload(f"{output_dir}/**", f"{GCS_UPLOAD_DIR}/{output_dir.name}/")


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
