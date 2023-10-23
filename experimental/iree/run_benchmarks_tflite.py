# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import pathlib
import re
import subprocess
import time

# Regexes for retrieving memory information.
_VMHWM_REGEX = re.compile(r".*?VmHWM:.*?(\d+) kB.*")
_VMRSS_REGEX = re.compile(r".*?VmRSS:.*?(\d+) kB.*")
_RSSFILE_REGEX = re.compile(r".*?RssFile:.*?(\d+) kB.*")


def run_command(benchmark_command: list[str]) -> tuple[str]:
  """Runs `benchmark_command` and polls for memory consumption statistics.
  Args:
    benchmark_command: A bash command string that runs the benchmark.
  Returns:
    An array containing values for [`latency`, `vmhwm`, `vmrss`, `rssfile`]
  """
  benchmark_process = subprocess.Popen(benchmark_command,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)

  # Keep a record of the highest VmHWM corresponding VmRSS and RssFile values.
  vmhwm = 0
  vmrss = 0
  rssfile = 0
  while benchmark_process.poll() is None:
    pid_status = subprocess.run(
        ["cat", "/proc/" + str(benchmark_process.pid) + "/status"],
        capture_output=True,
    )
    output = pid_status.stdout.decode()
    vmhwm_matches = _VMHWM_REGEX.search(output)
    vmrss_matches = _VMRSS_REGEX.search(output)
    rssfile_matches = _RSSFILE_REGEX.search(output)

    if vmhwm_matches and vmrss_matches and rssfile_matches:
      curr_vmhwm = float(vmhwm_matches.group(1))
      if curr_vmhwm > vmhwm:
        vmhwm = curr_vmhwm
        vmrss = float(vmrss_matches.group(1))
        rssfile = float(rssfile_matches.group(1))

    time.sleep(0.5)

  stdout_data, _ = benchmark_process.communicate()

  if benchmark_process.returncode != 0:
    print(f"Warning! Benchmark command failed with return code:"
          f" {benchmark_process.returncode}")
    return [0, 0, 0, 0]
  else:
    output = stdout_data.decode()
    print(output)

  LATENCY_REGEX = re.compile(
      "INFO: Inference timings in us: .* Inference \(avg\): (.*)")
  match = LATENCY_REGEX.search(output)
  latency_ms = 0 if not match else float(match.group(1)) * 1e-3

  return (latency_ms, vmhwm * 1e-3)


def benchmark(artifact_dir: pathlib.Path, tflite_filename: str,
              benchmark_model_path: pathlib.Path, num_threads: str):
  model_path = artifact_dir / f"{tflite_filename}.tflite"
  command = [
      str(benchmark_model_path),
      f"--graph={model_path}",
      f"--num_threads={num_threads}",
  ]
  command_str = " ".join(command)
  print(f"Running command: {command_str}")

  latency_ms, system_mem_peak = run_command(command)

  command.append("--report_peak_memory_footprint=true")
  result = subprocess.run(command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
  output = result.stdout.decode("utf-8")
  print(output)

  IREE_MEM_PEAK_REGEX = re.compile(
      "INFO: Overall peak memory footprint \(MB\) via periodic monitoring: (.*)"
  )
  match = IREE_MEM_PEAK_REGEX.search(output)
  tflite_mem_peak = 0 if not match else float(match.group(1).strip())

  print(f"{latency_ms},{tflite_mem_peak},{system_mem_peak}")

  return (latency_ms, tflite_mem_peak, system_mem_peak)


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Runs IREE benchmarks.")
  parser.add_argument("-o",
                      "--output_csv",
                      type=pathlib.Path,
                      required=True,
                      help="Path to save results to in csv format.")
  parser.add_argument(
      "--artifact_dir",
      type=pathlib.Path,
      required=True,
      help="The directory containing all required benchmark artifacts.")
  parser.add_argument("--benchmark_model_path",
                      type=pathlib.Path,
                      required=True,
                      help="Path to the TFLite benchmark_model binary.")
  parser.add_argument(
      "--benchmark_model_flex_path",
      type=pathlib.Path,
      required=True,
      help="Path to the TFLite benchmark_model with flex ops binary.")
  parser.add_argument("--threads",
                      type=str,
                      help="A comma-separated list of threads.")
  parser.add_argument(
      "--tasksets",
      type=str,
      help=
      "(Optional) A comma-separated list of tasksets to run under each thread configuration."
  )
  return parser.parse_args()


def main(output_csv: pathlib.Path,
         artifact_dir: pathlib.Path,
         benchmark_model_path: pathlib.Path,
         benchmark_model_flex_path: pathlib.Path,
         threads: str,
         tasksets: str = None):

  if not output_csv.exists():
    output_csv.write_text(
        "name,opset,benchmark_binary,threads,latency,peak_mem_tflite,peak_mem_system\n"
    )

  threads = threads.split(",")
  for thread in threads:
    try:
      latency_ms, tflite_mem_peak, system_mem_peak = benchmark(
          artifact_dir, "model_fp32", benchmark_model_path, thread)

      with open(output_csv, 'a') as file:
        file.write(
            f"{artifact_dir.name},tfl,no_flex,{thread},{latency_ms},{tflite_mem_peak},{system_mem_peak}\n"
        )
    except Exception as e:
      print(f"Failed to benchmark model {artifact_dir.name}. Exception: {e}")
      with open(output_csv, 'a') as file:
        file.write(
            f"{artifact_dir.name},tfl,no_flex,{thread},exception: {e},,,\n")

    try:
      latency_ms, tflite_mem_peak, system_mem_peak = benchmark(
          artifact_dir, "model_fp32", benchmark_model_flex_path, thread)

      with open(output_csv, 'a') as file:
        file.write(
            f"{artifact_dir.name},tfl,flex,{thread},{latency_ms},{tflite_mem_peak},{system_mem_peak}\n"
        )
    except Exception as e:
      print(f"Failed to benchmark model {artifact_dir.name}. Exception: {e}")
      with open(output_csv, 'a') as file:
        file.write(f"{artifact_dir.name},tfl,flex,{thread},exception: {e},,,\n")


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
