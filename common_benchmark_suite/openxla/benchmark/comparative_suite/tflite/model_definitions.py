# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import string

from openxla.benchmark import def_types
from openxla.benchmark.comparative_suite import utils

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/tflite/tflite_models_1698017047/"

TFLITE_MODEL_IMPL = def_types.ModelImplementation(
    name="TFLITE_MODEL_IMPL",
    tags=["tflite"],
    framework_type=def_types.ModelFrameworkType.TFLITE,
    module_path=f"{utils.MODELS_MODULE_PATH}.tflite.tflite_model",
)

BERT_BASE_FP32_TFLITE_I32_SEQLEN8 = def_types.Model(
    name="BERT_BASE_FP32_TFLITE_I32_SEQLEN8",
    tags=["fp32", "batch-1"],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "uri":
            "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.19_1698129954/BERT_BASE_FP32_JAX_I32_SEQLEN8/model_fp32.tflite"
    },
    artifacts_dir_url=f"{PARENT_GCS_DIR}/BERT_BASE_FP32_TFLITE_I32_SEQLEN8",
)

BERT_BASE_F16_TFLITE_I32_SEQLEN8 = def_types.Model(
    name="BERT_BASE_F16_TFLITE_I32_SEQLEN8",
    tags=["fp32", "batch-1"],
    model_impl=TFLITE_MODEL_IMPL,
    model_parameters={
        "uri":
            "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.19_1698129954/BERT_BASE_FP32_JAX_I32_SEQLEN8/model_fp16.tflite"
    },
    artifacts_dir_url=f"{PARENT_GCS_DIR}/BERT_BASE_F16_TFLITE_I32_SEQLEN8",
)

ALL_MODELS = [
    BERT_BASE_FP32_TFLITE_I32_SEQLEN8, BERT_BASE_F16_TFLITE_I32_SEQLEN8
]
