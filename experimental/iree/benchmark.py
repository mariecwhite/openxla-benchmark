#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import dataclasses
import json
import pathlib
import re
import statistics
import subprocess
import sys

# Add comparative_benchmark dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "comparative_benchmark"))
import utils

# Add common_benchmark_suite dir to the search path.
sys.path.insert(
    0, str(pathlib.Path(__file__).parents[2] / "common_benchmark_suite"))
<<<<<<< Updated upstream
from openxla.benchmark import def_types, devices
=======

from openxla.benchmark import def_types, devices
import openxla.benchmark.comparative_suite.tf.benchmark_definitions as tf_benchmark_definitions


>>>>>>> Stashed changes

ALL_DEVICE_NAMES = [device.name for device in devices.ALL_DEVICES]


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run GGML benchmarks.")
  parser.add_argument("-name",
                      "--benchmark_name",
                      type=str,
                      required=True,
                      help="The regex pattern to match benchmark names.")
  parser.add_argument("-c",
                      "--compile_binary",
                      type=pathlib.Path,
                      default="iree-compile",
<<<<<<< Updated upstream
                      required=True,
=======
>>>>>>> Stashed changes
                      help="Path to benchmark `iree-compile`")
  parser.add_argument("-b",
                      "--benchmark_binary",
                      type=pathlib.Path,
                      default="iree-benchmark-module",
<<<<<<< Updated upstream
                      required=True,
=======
>>>>>>> Stashed changes
                      help="Path to benchmark `iree-benchmark-module`")
  parser.add_argument("-d",
                      "--data_type",
                      type=str,
                      default="fp32",
                      help="The model data type.")
  parser.add_argument("-t",
                      "--threads",
                      type=int,
                      default=8,
                      help="The number of threads to use.")
  parser.add_argument("-o",
                      "--output",
                      type=pathlib.Path,
                      required=True,
                      help="JSON file path to merge the results.")
  parser.add_argument("-device",
                      "--target_device",
                      dest="target_device_name",
                      type=str,
                      required=True,
                      choices=ALL_DEVICE_NAMES,
                      help="The target device to benchmark.")
<<<<<<< Updated upstream
  parser.add_argument("-w",
                      "--warmup_iterations",
                      type=int,
                      default=5,
                      help="The number of warmup steps.")
  parser.add_argument("-iter",
                      "--iterations",
                      type=int,
                      default=100,
                      help="The number of iterations to benchmark.")
=======
  parser.add_argument("--root-dir",
                      "--root_dir",
                      type=pathlib.Path,
                      default=pathlib.Path("/tmp/openxla-benchmark/iree"),
                      help="Root directory stores benchmark artifacts.")
>>>>>>> Stashed changes
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")
  return parser.parse_args()


def main(benchmark_name: str, compile_binary: pathlib.Path,
<<<<<<< Updated upstream
    benchmark_binary: pathlib.Path,
    warmup_iterations: int, iterations: int, data_type: str,
    threads: int, output: pathlib.Path,
    target_device_name: str, verbose: bool):

=======
    benchmark_binary: pathlib.Path, data_type: str,
    threads: int, output: pathlib.Path,
    root_dir: pathlib.Path,
    target_device_name: str, verbose: bool):
>>>>>>> Stashed changes
  try:
    target_device = next(device for device in devices.ALL_DEVICES
                         if device.name == target_device_name)
  except StopIteration:
    raise ValueError(f'Target device "{target_device_name}" is not defined.'
                     f' Available device options:\n{ALL_DEVICE_NAMES}')

<<<<<<< Updated upstream
=======
  # The only benchmark supported is `models/GPT2LMHEAD_FP32_TF/inputs/INPUT_DATA_MODEL_DEFAULT`.
  assert benchmark_name == "models/GPT2LMHEAD_FP32_TF/inputs/INPUT_DATA_MODEL_DEFAULT"

>>>>>>> Stashed changes
  benchmark_definition = {
      "benchmark_name": benchmark_name,
      "framework": str(def_types.ModelFrameworkType.TF_V2),
      "data_type": data_type,
      "batch_size": 1,
      "compiler": "iree",
      "device": target_device.name,
      "num_threads": threads,
<<<<<<< Updated upstream
      "warmup_iterations": warmup_iterations,
      "num_iterations": iterations,
      "tags": ["gpt2", "ggml"],
  }

  cmd = [
      benchmark_binary,
      "--model",
      f"{model}",
      "--prompt",
      f"{prompt}",
      "--seed",
      f"{seed}",
      "--threads",
      f"{threads}",
  ]

  # Run warmup iterations.
  for i in range(warmup_iterations):
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  load_times = []
  first_prediction_times = []
  loop_prediction_times = []
  total_prediction_times = []
  sample_times = []
  e2e_prediction_times = []

  # Run iterations.
  for i in range(iterations):
    raw_result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    raw_result = raw_result.stdout.decode("utf-8")
    metrics = _parse_output(raw_result)

    load_times.append(metrics["load_time_ms"])
    first_prediction_times.append(metrics["first_prediction_ms"])
    loop_prediction_times.append(metrics["loop_prediction_ms"])
    total_prediction_times.append(metrics["total_prediction_ms"])
    sample_times.append(metrics["sample_time_ms"])
    e2e_prediction_times.append(metrics["e2e_prediction_ms"])

  benchmark_metrics = {
      "median_load_time_ms":
        statistics.median(load_times) if load_times else None,
      "median_first_prediction_ms":
        statistics.median(first_prediction_times)
        if first_prediction_times else None,
      "median_loop_prediction_ms":
        statistics.median(loop_prediction_times)
        if loop_prediction_times else None,
      "median_total_prediction_ms":
        statistics.median(total_prediction_times)
        if total_prediction_times else None,
      "median_sample_time_ms":
        statistics.median(sample_times) if sample_times else None,
      "median_e2e_prediction_times":
        statistics.median(e2e_prediction_times)
        if e2e_prediction_times else None,
=======
      "tags": ["gpt2"],
  }

  # Download artifacts.
  print("Downloading")
  utils.download_file("https://storage.googleapis.com/iree-shared-files/tf_gpt2.tgz", root_dir, verbose=verbose)

  root_dir = root_dir.as_posix()

  # Compile with default flags.
  print("Compiling seqlen5 with default flags")
  cmd = [
      compile_binary.as_posix(),
      "--iree-hal-target-backends=llvm-cpu",
      "--iree-llvmcpu-target-cpu=cascadelake",
      "--iree-input-type=stablehlo",
      "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu",
      f"{root_dir}/tf_gpt2/static_input_seqlen5/stablehlo.mlir",
      "-o",
      f"{root_dir}/tf_gpt2/static_input_seqlen5/module_default.vmfb",
  ]
  print(cmd)
  subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  print("Compiling seqlen1 with default flags")
  cmd = [
      compile_binary.as_posix(),
      "--iree-hal-target-backends=llvm-cpu",
      "--iree-llvmcpu-target-cpu=cascadelake",
      "--iree-input-type=stablehlo",
      "--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu",
      f"{root_dir}/tf_gpt2/static_input_seqlen1/stablehlo.mlir",
      "-o",
      f"{root_dir}/tf_gpt2/static_input_seqlen1/module_default.vmfb",
  ]
  print(cmd)
  subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  cmd = [
      benchmark_binary.as_posix(),
      "--function=forward",
      f"--input=@{root_dir}/tf_gpt2/static_input_seqlen5/inputs_npy/input_0.npy",
      f"--input=@{root_dir}/tf_gpt2/static_input_seqlen5/inputs_npy/input_1.npy",
      f"--module={root_dir}/tf_gpt2/static_input_seqlen5/module_default.vmfb",
      f"--benchmark_min_warmup_time=2",
      f"--benchmark_min_time=5",
      f"--task_topology_group_count=${threads}",
  ]

  print("Benchmarking")
  print(cmd)

  first_prediction_times = []
  raw_result = subprocess.run(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
  raw_result = raw_result.stdout.decode("utf-8")
  print(raw_result)

  benchmark_metrics = {
      "median_first_prediction_ms":
        statistics.median(first_prediction_times)
        if first_prediction_times else None,
>>>>>>> Stashed changes
  }

  benchmark_result = utils.BenchmarkResult(
      definition=benchmark_definition,
      metrics={
          "compiler_level": benchmark_metrics,
      },
  )

  if verbose:
    print(json.dumps(dataclasses.asdict(benchmark_result), indent=2))
  utils.append_benchmark_result(output, benchmark_result)


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
