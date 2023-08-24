#!/bin/bash
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Environment variables:
#  REPO_DIR=mmperf
#
# Example usage:
#   ./run_mmperf.sh <build-dir> <output-dir>

set -xeuo pipefail

REPO_DIR="${REPO_DIR:-mmperf}"
BUILD_DIR=$1
OUTPUT_DIR=$2

source ${REPO_DIR}/mmperf.venv/bin/activate
python3 ${REPO_DIR}/mmperf.py ${BUILD_DIR}/matmul/ ${OUTPUT_DIR}
