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

import numpy as np

# Regexes for retrieving memory information.
_VMHWM_REGEX = re.compile(r".*?VmHWM:.*?(\d+) kB.*")
_VMRSS_REGEX = re.compile(r".*?VmRSS:.*?(\d+) kB.*")
_RSSFILE_REGEX = re.compile(r".*?RssFile:.*?(\d+) kB.*")


def compare_npy(a, b, atol, rtol):
  if (a.shape != b.shape):
    print(f"Array dimensions are different. a: {a.shape}, b: {b.shape}")
    return

  return np.allclose(a, b, atol=atol, rtol=rtol)


def check_accuracy(artifact_dir: pathlib.Path, vmfb_name: str,
                   iree_run_module_path: pathlib.Path, num_threads: int,
                   atol: float, rtol: float):
  inputs_dir = artifact_dir / "inputs_npy"
  num_inputs = len(list(inputs_dir.glob("*.npy")))

  module_path = artifact_dir / f"{vmfb_name}.vmfb"
  output_npy = artifact_dir / f"{vmfb_name}_output.npy"
  command = [
      str(iree_run_module_path),
      f"--module={module_path}",
      f"--task_topology_group_count={num_threads}",
      "--device=local-task",
      "--function=main",
      f"--output=@{output_npy}",
  ]

  for i in range(num_inputs):
    command.append(f"--input=@{inputs_dir}/input_{i}.npy")

  subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  a = np.load(output_npy)
  b = np.load(artifact_dir / "outputs_npy" / "output_0.npy")
  return compare_npy(a, b, atol, rtol)


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

  LATENCY_REGEX = re.compile(r".*?BM_main/process_time/real_time\s+(.*?) ms.*")
  match = LATENCY_REGEX.search(output)
  latency_ms = 0 if not match else float(match.group(1))

  IREE_MEM_PEAK_REGEX = re.compile(r".*?DEVICE_LOCAL: (.*?)B peak .*")
  match = IREE_MEM_PEAK_REGEX.search(output)
  iree_mem_peak = 0 if not match else float(match.group(1).strip()) * 1e-6
  print(f"latency_ms: {latency_ms}")
  print(f"iree_mem_peak: {iree_mem_peak}")
  return (latency_ms, iree_mem_peak, vmhwm * 1e-3)


def benchmark(artifact_dir: pathlib.Path, vmfb_name: str,
              iree_benchmark_module_path: pathlib.Path, num_threads: str):
  inputs_dir = artifact_dir / "inputs_npy"
  num_inputs = len(list(inputs_dir.glob("*.npy")))

  module_path = artifact_dir / f"{vmfb_name}.vmfb"
  command = [
      str(iree_benchmark_module_path),
      f"--module={module_path}",
      f"--task_topology_group_count={num_threads}",
      "--device=local-task",
      "--function=main",
      "--print_statistics",
  ]

  for i in range(num_inputs):
    command.append(f"--input=@{inputs_dir}/input_{i}.npy")

  return run_command(command)


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
  parser.add_argument("--iree_run_module_path",
                      type=pathlib.Path,
                      required=True,
                      help="Path to the iree-run-module binary.")
  parser.add_argument("--iree_benchmark_module_path",
                      type=pathlib.Path,
                      required=True,
                      help="Path to the iree-benchmark-module binary.")
  parser.add_argument("--threads",
                      type=str,
                      help="A comma-separated list of threads.")
  parser.add_argument(
      "--tasksets",
      type=str,
      help=
      "(Optional) A comma-separated list of tasksets to run under each thread configuration."
  )
  parser.add_argument("--atol",
                      type=float,
                      default=1.e-1,
                      help="The absolute tolerance to use for accuracy checks.")
  parser.add_argument("--rtol",
                      type=float,
                      default=1.e-8,
                      help="The relative tolerance to use for accuracy checks.")
  return parser.parse_args()


def main(output_csv: pathlib.Path,
         artifact_dir: pathlib.Path,
         iree_run_module_path: pathlib.Path,
         iree_benchmark_module_path: pathlib.Path,
         threads: str,
         atol: float,
         rtol: float,
         tasksets: str = None):

  if not output_csv.exists():
    output_csv.write_text(
        "name,compile_flags,threads,accuracy,latency,peak_mem_iree,peak_mem_system\n"
    )

  threads = threads.split(",")
  for thread in threads:
    try:
      is_accurate = check_accuracy(artifact_dir, "module_default",
                                   iree_run_module_path, thread, atol, rtol)
      if not is_accurate:
        with open(output_csv, 'a') as file:
          file.write(f"{artifact_dir.name},default,{thread},fail,,,\n")
      else:
        latency_ms, iree_mem_peak, system_mem_peak = benchmark(
            artifact_dir, "module_default", iree_benchmark_module_path, thread)

        with open(output_csv, 'a') as file:
          file.write(
              f"{artifact_dir.name},default,{thread},pass,{latency_ms},{iree_mem_peak},{system_mem_peak}\n"
          )
    except Exception as e:
      print(f"Failed to benchmark model {artifact_dir.name}. Exception: {e}")
      with open(output_csv, 'a') as file:
        file.write(f"{artifact_dir.name},default,{thread},exception: {e},,,\n")

    try:
      is_accurate = check_accuracy(artifact_dir, "module_experimental",
                                   iree_run_module_path, thread, atol, rtol)
      if not is_accurate:
        with open(output_csv, 'a') as file:
          file.write(f"{artifact_dir.name},experimental,{thread},fail,,,\n")
      else:
        latency_ms, iree_mem_peak, system_mem_peak = benchmark(
            artifact_dir, "module_experimental", iree_benchmark_module_path,
            thread)

        with open(output_csv, 'a') as file:
          file.write(
              f"{artifact_dir.name},experimental,{thread},pass,{latency_ms},{iree_mem_peak},{system_mem_peak}\n"
          )
    except Exception as e:
      print(f"Failed to benchmark model {artifact_dir.name}. Exception: {e}")
      with open(output_csv, 'a') as file:
        file.write(
            f"{artifact_dir.name},experimental,{thread},exception: {e},,,\n")


if __name__ == "__main__":
  main(**vars(_parse_arguments()))
