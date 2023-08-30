# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import tensorflow as tf

from openxla.benchmark.models import model_interfaces, utils
from transformers import AutoTokenizer, GPT2Tokenizer, TFGPT2LMHeadModel
from typing import Any, List, Tuple


class GPT2LMHead(tf.Module, model_interfaces.InferenceModel):
  """See https://huggingface.co/docs/transformers/model_doc/gpt2 for more information.

  This version of GPT2 is configured to match the GGML implementation in https://github.com/ggerganov/ggml/blob/7b5fcf5f2a676e6d7c018c6a15dcb6338d9b2a38/examples/gpt-2/main.cpp.
  Specifically:
      1. Use the Tensorflow version since GGML uses a TF checkpoint to retrieve GPT2 co-efficients.
      2. Use sampling for post-processing with combined top_k and top_p: https://github.com/ggerganov/ggml/blob/7b5fcf5f2a676e6d7c018c6a15dcb6338d9b2a38/examples/common.cpp#L414
      3. Assume RNG seed of 0 for sampling.
      4. Generate a maximum of 200 new tokens: https://github.com/ggerganov/ggml/blob/7b5fcf5f2a676e6d7c018c6a15dcb6338d9b2a38/examples/common.h#L20
      5. Input sequence length of 1024: https://github.com/ggerganov/ggml/blob/7b5fcf5f2a676e6d7c018c6a15dcb6338d9b2a38/examples/gpt-2/main.cpp#L23
      6. Vocab size of 50257: https://github.com/ggerganov/ggml/blob/7b5fcf5f2a676e6d7c018c6a15dcb6338d9b2a38/examples/gpt-2/main.cpp#L22
  """

  batch_size: int
  seq_len: int
  model: TFGPT2LMHeadModel
  model_name: str
  tokenizer: GPT2Tokenizer
  tokenization_kwargs: dict[str, Any]

  def __init__(
      self,
      batch_size: int,
      seq_len: int,
      model_name: str,
  ):
    self.model = TFGPT2LMHeadModel.from_pretrained(model_name)
    self.model_name = model_name
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=self.seq_len, padding_side="left",)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    # GGML does not implement padding but we set padding here to make sure the
    # input size in the lowered artifacts support `seq_len`.
    self.tokenization_kwargs = {
        #"pad_to_multiple_of": self.seq_len,
        #"pad_to_multiple_of": 8,
        #"padding": True,
        "return_tensors": "tf",
    }

    # Make sure results are deterministic.
    #tf.random.set_seed(0)

  def generate_default_inputs(self) -> str:
    return "Once upon a time"

  def preprocess(self, input_text: str) -> Tuple[Any, Any]:
    batch_input_text = [input_text] * self.batch_size
    inputs = self.tokenizer(text=batch_input_text, **self.tokenization_kwargs)
    return (inputs["input_ids"], inputs["attention_mask"])
    #return (inputs["input_ids"],)

  def init(self, input_ids: Any, attention_mask: Any) -> Any:
    return 0

  @tf.function(jit_compile=True)
  #def init(self, input_ids: Any, attention_mask: Any) -> Any:
  def forward(self, input_ids: Any, attention_mask: Any) -> Any:
    results = self.model(input_ids, attention_mask=attention_mask)
    past_key_values = tf.stack(results.past_key_values, axis=0)
    return (results.logits, past_key_values)

  #@tf.function(jit_compile=True)
  #@tf.function(input_signature=(tf.TensorSpec(shape=[1, None], dtype=tf.int32), tf.TensorSpec(shape=[1, None], dtype=tf.int32)))
  #@tf.function
  #def forward(self, input_ids: Any, attention_mask: Any) -> Any:
  #def forward(self, input_ids: Any) -> Any:
    #return self.model.generate(input_ids, max_new_tokens=200, do_sample=True,
    #                           top_k=40, top_p=0.9, repetition_penalty=1.0,
    #                           num_beams=1)
    #return self.model.generate(input_ids, max_new_tokens=200)
    #return self.model(input_ids, attention_mask=attention_mask).logits

  # @tf.function(jit_compile=True)
  # def forward(self, input_ids: Any, past_key_values: Any) -> Any:
  #   past_key_values = tf.unstack(past_key_values, axis=0)
  #   results = self.model(input_ids, past_key_values=past_key_values)
  #   past_key_values = tf.stack(results.past_key_values, axis=0)
  #   return (results.logits, past_key_values)

  def postprocess(self, output: Any) -> List[str]:
    return self.tokenizer.batch_decode(output, skip_special_tokens=True)


def create_model(batch_size: int = 1,
                 seq_len: int = 1024,
                 model_name: str = "gpt2",
                 **_unused_params) -> GPT2LMHead:
  """Configure and create a TF GPT2LMHead model instance.

  Args:
    batch_size: input batch size.
    seq_len: input sequence length.
    model_name: The name of the GPT2 variant to use. Supported variants include:
      gpt2 (117M params), gpt2-medium (345M params), gpt2-large (774M params),
      gpt2-xl (1558M params).
  Returns:
    A TF GPT2LMHead model.
  """
  return GPT2LMHead(batch_size=batch_size,
                    seq_len=seq_len,
                    model_name=model_name)
