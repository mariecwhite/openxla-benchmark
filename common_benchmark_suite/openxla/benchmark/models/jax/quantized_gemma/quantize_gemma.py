import argparse
import os
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from paxml import base_experiment
from praxis import base_hyperparams
from praxis.layers.quantization import utils, quantization_hparams
from saxml.server import servable_model_registry
from saxml.server.jax import servable_model
from saxml.server.pax.lm.params import gemma, template


class QuantBase(base_experiment.BaseExperiment):

  # Shared parameters
  QUANTIZATION_MODE = quantization_hparams.QuantizationMode.INFERENCE
  QUANTIZATION_TYPE = quantization_hparams.QuantizationType.AQT
  USE_SYMMETRIC = True
  MIN_CLIPPING = None
  NUM_OPTIMIZE_CLIPPING = None
  ADD_SCALE_EPS = False
  OPTIMIZE_CLIPPING_PER_CHANNEL = False
  QUANT_LOSS_WEIGHT = None
  CLIPPING_COEFF = 1.0
  SUB_CHANNELS = None
  USE_INT4_PACKED_WEIGHTS = False
  USE_INT4_TYPES = True
  KURT_LOSS_WEIGHT = None

  # Sublayer parameters
  PREC_WEIGHT_FFN = 4
  PREC_ACT_FFN = None

  PREC_WEIGHT_ATTN = None
  PREC_ACT_ATTN = None

  PREC_WEIGHT_EMBEDDING_SOFTMAX = None
  TRANSPOSED_EMBEDDING_SOFTMAX = True
  PREC_ACT_EMBEDDING_SOFTMAX = None

  def _get_weight_params(self, prec: int) -> WeightQuantizationParams:
    dtype = jnp.int4 if prec <= 4 and self.USE_INT4_TYPES else jnp.int8
    return WeightQuantizationParams(
        precision=prec,
        dtype=dtype,
        use_symmetric=self.USE_SYMMETRIC,
        min_clipping=self.MIN_CLIPPING,
        num_optimize_clipping=self.NUM_OPTIMIZE_CLIPPING,
        optimize_clipping_per_channel=self.OPTIMIZE_CLIPPING_PER_CHANNEL,
        add_scale_eps=self.ADD_SCALE_EPS,
        quant_loss_weight=self.QUANT_LOSS_WEIGHT,
        clipping_coeff=self.CLIPPING_COEFF,
        sub_channels=self.SUB_CHANNELS,
        use_int4_packed_weights=self.USE_INT4_PACKED_WEIGHTS,
        kurt_loss_weight=self.KURT_LOSS_WEIGHT,
    )

  def _get_act_params(self, prec: int) -> ActQuantizationParams | None:
    if prec is None:
      return None
    else:
      return ActQuantizationParams(precision=prec)

  def _quantize_transformer_ffn_layers(self, model):
    if self.PREC_WEIGHT_FFN is None:
      return

    weight_params = self._get_weight_params(self.PREC_WEIGHT_FFN)
    act_params = self._get_act_params(self.PREC_ACT_FFN)

    tr_tpls = utils.find_target_tpl(model, layers.transformers.Transformer)
    assert tr_tpls
    for tr_tpl in tr_tpls:
      quantize.quantize_transformer_feed_forward_layer_weights(
          tr_tpl.tr_fflayer_tpl,
          self.QUANTIZATION_TYPE,
          self.QUANTIZATION_MODE,
          weight_params,
          act_params,
      )

    return model

  def _quantize_attention_layers(self, model):
    if self.PREC_WEIGHT_ATTN is None:
      return

    weight_params = self._get_weight_params(self.PREC_WEIGHT_ATTN)
    act_params = self._get_act_params(self.PREC_ACT_ATTN)

    tr_tpls = utils.find_target_tpl(model, layers.transformers.Transformer)
    assert tr_tpls
    for tr_tpl in tr_tpls:
      quantize.quantize_attention_layer_weights(
          tr_tpl,
          self.QUANTIZATION_TYPE,
          self.QUANTIZATION_MODE,
          weight_params,
          act_params,
      )

  def _quantize_embedding_softmax_layer(self, model):
    if self.PREC_WEIGHT_EMBEDDING_SOFTMAX is None:
      return

    weight_params = self._get_weight_params(self.PREC_WEIGHT_EMBEDDING_SOFTMAX)
    act_params = self._get_act_params(self.PREC_ACT_EMBEDDING_SOFTMAX)

    lm_tpls = utils.find_target_tpl(model, layers.TransformerLm)
    assert lm_tpls
    for lm_tpl in lm_tpls:
      quantize._quantize_embedding_softmax_layer_weights(
          lm_tpl,
          quantization_type=self.QUANTIZATION_TYPE,
          mode=self.QUANTIZATION_MODE,
          weight_quantization_params=weight_params,
          act_quantization_params=act_params,
          transposed_embedding_softmax=self.TRANSPOSED_EMBEDDING_SOFTMAX,
      )

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    self._quantize_transformer_ffn_layers(task_p.model)
    self._quantize_attention_layers(task_p.model)
    self._quantize_embedding_softmax_layer(task_p.model)
    return task_p


class GemmaQuantBase(QuantBase):
  TRANSPOSED_EMBEDDING_SOFTMAX = True
  USE_SYMMETRIC = True


@servable_model_registry.register
class Gemma2B_i8w(GemmaQuantBase, gemma.Gemma2BFP16):
  PREC_WEIGHT_FFN = 8
  PREC_WEIGHT_ATTN = 8
  PREC_WEIGHT_EMBEDDING_SOFTMAX = 8


@servable_model_registry.register
class Gemma2B_i8wa(Gemma2B_i8w):
  PREC_ACT_FFN = 8
  PREC_ACT_ATTN = 8
  PREC_ACT_EMBEDDING_SOFTMAX = 8


@servable_model_registry.register
class Gemma2B_i488w(GemmaQuantBase, gemma.Gemma2BFP16):
  PREC_WEIGHT_FFN = 4
  PREC_WEIGHT_ATTN = 8
  PREC_WEIGHT_EMBEDDING_SOFTMAX = 8


@servable_model_registry.register
class Gemma2B_i488wa(Gemma2B_i488w):
  PREC_ACT_FFN = 8
  PREC_ACT_ATTN = 8
  PREC_ACT_EMBEDDING_SOFTMAX = 8


@servable_model_registry.register
class Gemma2B_i484w(GemmaQuantBase, gemma.Gemma2BFP16):
  PREC_WEIGHT_FFN = 4
  PREC_WEIGHT_ATTN = 8
  PREC_WEIGHT_EMBEDDING_SOFTMAX = 4


@servable_model_registry.register
class Gemma2B_i484wa(Gemma2B_i484w):
  PREC_ACT_FFN = 8
  PREC_ACT_ATTN = 8
  PREC_ACT_EMBEDDING_SOFTMAX = 8


@servable_model_registry.register
class Gemma7B_i8w(GemmaQuantBase, gemma.Gemma7BFP16):
  PREC_WEIGHT_FFN = 8
  PREC_WEIGHT_ATTN = 8
  PREC_WEIGHT_EMBEDDING_SOFTMAX = 8


@servable_model_registry.register
class Gemma7B_i8wa(Gemma7B_i8w):
  PREC_ACT_FFN = 8
  PREC_ACT_ATTN = 8
  PREC_ACT_EMBEDDING_SOFTMAX = 8


@servable_model_registry.register
class Gemma7B_i488w(GemmaQuantBase, gemma.Gemma7BFP16):
  PREC_WEIGHT_FFN = 4
  PREC_WEIGHT_ATTN = 8
  PREC_WEIGHT_EMBEDDING_SOFTMAX = 8


@servable_model_registry.register
class Gemma7B_i488wa(Gemma7B_i488w):
  PREC_ACT_FFN = 8
  PREC_ACT_ATTN = 8
  PREC_ACT_EMBEDDING_SOFTMAX = 8


@servable_model_registry.register
class Gemma7B_i484w(GemmaQuantBase, gemma.Gemma7BFP16):
  PREC_WEIGHT_FFN = 4
  PREC_WEIGHT_ATTN = 8
  PREC_WEIGHT_EMBEDDING_SOFTMAX = 4


@servable_model_registry.register
class Gemma7B_i484wa(Gemma7B_i484w):
  PREC_ACT_FFN = 8
  PREC_ACT_ATTN = 8
  PREC_ACT_EMBEDDING_SOFTMAX = 8


def _remove_padding(x, shape):
  original_dtype = None
  if x.dtype in utils.INT4_TYPES:
    original_dtype = x.dtype
    x = x.astype(jnp.int8)
  x = servable_model.remove_padding(x, shape)
  if original_dtype is not None:
    x = x.astype(original_dtype)
  return x


def _zeros_like(x):
  original_dtype = None
  if x.dtype in utils.INT4_TYPES:
    original_dtype = x.dtype
    x = x.astype(jnp.int8)
  x = jnp.zeros_like(x)
  if original_dtype is not None:
    x = x.astype(original_dtype)
  return x


def get_hparams_str(cfg):
  class_fields = []
  for k in dir(cfg):
    if k.upper() == k:
      class_fields.append(f"{k}: {getattr(cfg, k)}")
  class_fields = "\n".join(class_fields)
  hparams = base_hyperparams.nested_struct_to_text(cfg.task())
  return f"{class_fields}\n{hparams}"


def print_time(name, start, stop):
  total = stop - start
  print(f"{name}: {total:.1f}s ({total / 60:.2f}m)")


def create_sax_config(
    *,
    model_cls,
    method: str,
    checkpoint_path: str,
    batch_size: int | None,
    input_seq_len: int | None,
    max_decode_steps: int | None,
    repeated: bool,
):

  class Exportable(model_cls):
    REPEATED_LAYERS = repeated  # TPU training optimization.
    TOP_K_RECALL_TARGET = 1.0  # Use TopK instead of ApproxTopK.
    MODEL_DTYPE = jnp.float32
    FPROP_DTYPE = jnp.float32
    ICI_MESH_SHAPE = [1, 1, 1]
    GRADIENT_WRT_INPUT_TENSOR_NAMES = ["weights"]

    @property
    def test_mode(self) -> bool:
      return checkpoint_path is None

  if method == "lm.gradient":
    Exportable = template.make_servable(template.ServingWithGradientTemplate)(
        Exportable
    )

  cfg = Exportable()
  if batch_size:
    cfg.BATCH_SIZE = batch_size
  if input_seq_len:
    cfg.INPUT_SEQ_LEN = input_seq_len
  if max_decode_steps:
    cfg.MAX_DECODE_STEPS = max_decode_steps

  return cfg


def load_sax_method(*, cfg, checkpoint_path: str, method: str):
  _servable_model = cfg.create_model(primary_process_id=0)
  basemodel, model_state = _servable_model.load_state(
      checkpoint_path=checkpoint_path,
      prng_key=jax.random.PRNGKey(0),
      precompile=False,
  )
  method_params = _servable_model.model_config.methods()[method]
  method_obj = _servable_model.init_method(
      method,
      basemodel,
      model_state,
      method_params,
      jax.random.PRNGKey(0),
  )
  return method_obj


def generate_inputs(
    *,
    method_obj,
    method: str,
    batch_size: int,
    input_seq_len: int | None,
    input_text: str | None = None,
):
  # Load input text.
  if input_text is None:
    input_text = "A great idea for "
  if isinstance(input_text, str):
    input_text = [input_text]

  if method == "lm.gradient":
    input_text = [input_text * 2]  # [[text_file, text_file]]
  input_text *= batch_size

  # Convert to tokens.
  inputs = method_obj.pre_processing(input_text)
  input_keys = set(inputs.keys())
  inputs = method_obj.update_extra_inputs(inputs, batch_size, extra_inputs=None)
  extra_input_keys = set(inputs.keys()) - input_keys
  inputs = jax.tree.map(jnp.asarray, inputs)
  return inputs, extra_input_keys


def randomize(x: jnp.ndarray) -> jnp.ndarray:
  if x.dtype in utils.INT_TYPES:
    bits = utils.dtype_to_bits(x.dtype)
    low = -2 ** (bits - 1) + 1
    high = 2 ** (bits - 1) - 1
    # Numpy's randint's high is exclusive.
    random_x = np.random.randint(low, high + 1, size=x.shape)
    random_x = jnp.asarray(random_x).astype(x.dtype)
  else:
    random_x = np.random.normal(size=x.shape)
    random_x = jnp.asarray(random_x).astype(x.dtype)
  return random_x


def export_method(
    *,
    model_cls,
    checkpoint_path: str,
    output_dir: str,
    batch_size: int,
    input_seq_len: int,
    max_decode_steps: int,
    input_text: str,
    method: str = "lm.generate",
    repeated: bool = False,
    zero_weights: bool = False,
    save_expected_outputs: bool = True,
  ) -> None:

  np.random.seed(0)
  os.environ["IGNORE_PRESET_XLA_TPU_FLAGS"] = "1"

  cfg = create_sax_config(
      model_cls=model_cls,
      method=method,
      checkpoint_path=checkpoint_path,
      batch_size=batch_size,
      input_seq_len=input_seq_len,
      max_decode_steps=max_decode_steps,
      repeated=repeated,
  )

  # Create a unique name to save model artifacts to.
  artifact_dir = os.path.join(output_dir, model_cls.__name__)
  os.makedirs(artifact_dir, exist_ok=True)
  artifact_name = (
      f"{method}_batch_{batch_size}_input_{cfg.INPUT_SEQ_LEN}_"
      f"max_decode_{cfg.MAX_DECODE_STEPS}"
  )
  if repeated:
    artifact_name += "repeated"
  method_path = os.path.join(artifact_dir, artifact_name)
  print(f"METHOD_PATH={method_path}")

  def save_artifact(suffix: str, artifact):
    path = f"{method_path}{suffix}"
    print(f"Saving artifact to {path}")
    if isinstance(artifact, str):
      with open(path, "w") as f:
        f.write(artifact)
    else:
      with open(path, "wb") as f:
        np.save(f, artifact)

  save_artifact(".hparams", get_hparams_str(cfg))

  # Load the model.
  sax_load_start = time.perf_counter()
  method_obj = load_sax_method(
      cfg=cfg, checkpoint_path=checkpoint_path, method=method
  )
  sax_load_stop = time.perf_counter()
  print_time("sax load time", sax_load_start, sax_load_stop)

  # Tokenize the inputs and preprocess model variables.
  inputs_start = time.perf_counter()
  inputs, extra_input_keys = generate_inputs(
      method_obj=method_obj,
      method=method,
      batch_size=batch_size,
      input_seq_len=input_seq_len,
      input_text=input_text,
  )
  input_key = jax.random.PRNGKey(0)
  save_artifact("_input_ids.npy", inputs.ids)
  save_artifact("_input_prefix_lengths.npy", inputs.prefix_lengths)
  save_artifact("_input_key.npy", input_key)

  # Remove padding on the vars.
  mdl_vars = method_obj.model_state.mdl_vars
  for key in mdl_vars.keys():
    if key in method_obj.model_state.mdl_var_unpadded_shapes:
      mdl_vars[key] = jax.tree.map(
          _remove_padding,
          mdl_vars[key],
          method_obj.model_state.mdl_var_unpadded_shapes[key],
      )
  if zero_weights:
    mdl_vars = jax.tree.map(_zeros_like, mdl_vars)
  elif not checkpoint_path:
    # Manually randomize weights to account for quantized variables being zero
    # initialized.
    mdl_vars = jax.tree.map(randomize, mdl_vars)

  inputs_stop = time.perf_counter()
  print_time("inputs time", inputs_start, inputs_stop)

  # Export the model's method without wrapping it in pjit.
  export_start = time.perf_counter()

  def export_func(_inputs, _key):
    _inputs = _inputs.copy()
    # Each per-example tensor below could be turned into a single value.
    for key in extra_input_keys:
      # Remove the extra inputs from the input signature and then capture them
      # as constants.
      del _inputs[key]
      _inputs[key] = inputs[key]

    outputs = method_obj.jax_func(
        mdl_vars,
        _key,
        _inputs,
        non_batched_inputs=(),
    )

    @jax.vmap
    def dynamic_slice_generated_ids(ids, prefix_lengths):
      """Slices the generated ids from the output_ids based on prefix length."""
      return jax.lax.dynamic_slice_in_dim(
          ids,
          start_index=prefix_lengths,
          # slice_size must be a scalar, so we capture it here.
          slice_size=max_decode_steps,
          axis=-1,
      )

    # The max lengths of the prefix, which are the number of unpadded tokens.
    prefix_lengths = jnp.sum(1 - _inputs.paddings.astype(jnp.int32), axis=1)
    outputs.generated_ids = dynamic_slice_generated_ids(
        outputs.output_ids, prefix_lengths
    )

    return outputs

  lowered = jax.jit(export_func).lower(inputs, input_key)
  module = lowered.compiler_ir(dialect="stablehlo")
  asm = module.operation.get_asm(
      enable_debug_info=True, print_generic_op_form=False
  )
  save_artifact(".mlir", asm)

  export_stop = time.perf_counter()
  print_time("export time", export_start, export_stop)

  stop_time = time.perf_counter()
  if save_expected_outputs:
    output_start = time.perf_counter()
    outputs = export_func(inputs, input_key)
    for k in sorted(outputs.keys()):
      print(f"{k}:\n{outputs[k]}\n")
    print(method_obj.tf_post_processing(outputs)["topk_decoded"])

    save_artifact("_xla_decode_lengths.npy", outputs.decode_lengths)
    save_artifact("_xla_generated_ids.npy", outputs.generated_ids)
    save_artifact("_xla_output_ids.npy", outputs.output_ids)
    save_artifact("_xla_prefix_lengths.npy", outputs.prefix_lengths)
    save_artifact("_xla_scores.npy", outputs.scores)

    stop_time = time.perf_counter()
    print_time("output time", output_start, stop_time)


class QuantizationConfigs_i8w_trans(quantization_configs.QuantizationConfigs):

  factor = 1.0
  LINEAR_PREC = 8
  ATTEN_PREC = 8
  EMB_PREC = 8
  configs = {
      'ff_layer.ffn_layer1.linear.w':      ([0],    factor, 0, LINEAR_PREC),
      'ff_layer.ffn_layer1_gate.linear.w': ([0],    factor, 0, LINEAR_PREC),
      'ff_layer.ffn_layer2.linear.w':      ([0],    factor, 0, LINEAR_PREC),
      'self_attention.post.w':             ([1, 2], factor, 0, ATTEN_PREC),
      'self_attention.key.w':              ([0],    factor, 0, ATTEN_PREC),
      'self_attention.query.w':            ([0],    factor, 0, ATTEN_PREC),
      'self_attention.value.w':            ([0],    factor, 0, ATTEN_PREC),
      'self_attention.combined_qkv.w':     ([1],    factor, 1, ATTEN_PREC),
      'softmax.logits_ffn.linear.w':       ([1],    factor, 0, EMB_PREC),
      'softmax.w':                         ([1],    factor, 0, EMB_PREC),
  }


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Runs post-training quantization on Gemma and generates artifacts for benchmarking.")
  parser.add_argument("-o",
                      "--output_dir",
                      type=pathlib.Path,
                      required=True,
                      help="Directory to save model artifacts.")
  parser.add_argument("-c",
                      "--checkpoint",
                      type=pathlib.Path,
                      required=True,
                      help="Directory to the Gemma checkpoint.")
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
  return parser.parse_args()

def main(output_dir: pathlib.Path, checkpoint: pathlib.Path,
         iree_ir_tool: pathlib.Path, auto_upload: bool):
  offline_quantize(
    input_dir=checkpoint / "state",
    output_dir="/tmp/gemma_ptq",
    quantization_config=QuantizationConfigs_i8w_trans(),
    symmetric=True,
  )

  export_method(
    model_cls=Gemma2B_i8wa,
    checkpoint_path="/tmp/gemma_ptq/",
    output_dir="/tmp/gemma/",
    batch_size=1,
    input_seq_len=1024,
    max_decode_steps=256,
    input_text="",
)

if __name__ == "__main__":
  main(**vars(_parse_arguments()))

