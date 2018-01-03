# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 12/29/2017
#
# Description: Cindy, a mini CH-EN translation system
#
# Last Modified at: 12/30/2017, by: Synrey Yee

###############################################################
###                                                         ###
### The system is a mini version of Google's nmt (seq2seq). ###
###          (https://github.com/tensorflow/nmt)            ###
###           Hence, here is its license below.             ###
###                                                         ###
###############################################################

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

import argparse
import os
import random
import sys

import numpy as np
import tensorflow as tf

from .third_party import misc_utils as utils

utils.check_tensorflow_version()

FLAGS = None

def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # network
  parser.add_argument("--num_units", type=int, default=32, help="Network size.")
  parser.add_argument("--num_layers", type=int, default=2,
                      help="Network depth.")
  
  parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                      default=True,
                      help="Whether to use time-major mode for dynamic RNN.")

  # optimizer, use scheduling SGD
  parser.add_argument("--learning_rate", type=float, default=1.0,
                      help="Learning rate. Adam: 0.001 | 0.0001")
  parser.add_argument(
      "--decay_scheme", type=str, default="", help="""\
      How we decay learning rate. Options include:
        luong234: after 2/3 num train steps, we start halving the learning rate
          for 4 times before finishing.
        luong10: after 1/2 num train steps, we start halving the learning rate
          for 10 times before finishing.\
      """)

  parser.add_argument(
      "--num_train_steps", type=int, default=12000, help="Num steps to train.")

  # initializer
  parser.add_argument("--init_op", type=str, default="uniform",
                      help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument("--init_weight", type=float, default=0.1,
                      help=("for uniform init_op, initialize weights "
                            "between [-this, this]."))

  # data
  parser.add_argument("--src", type=str, default=None,
                      help="Source suffix, e.g., en.")
  parser.add_argument("--tgt", type=str, default=None,
                      help="Target suffix, e.g., de.")
  parser.add_argument("--train_prefix", type=str, default=None,
                      help="Train prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--dev_prefix", type=str, default=None,
                      help="Dev prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--test_prefix", type=str, default=None,
                      help="Test prefix, expect files with src/tgt suffixes.")
  parser.add_argument("--out_dir", type=str, default=None,
                      help="Store log/model files.")

  # Vocab
  parser.add_argument("--vocab_prefix", type=str, default=None, help="""\
      Vocab prefix, expect files with src/tgt suffixes.\
      """)
  parser.add_argument("--sos", type=str, default="<s>",
                      help="Start-of-sentence symbol.")
  parser.add_argument("--eos", type=str, default="</s>",
                      help="End-of-sentence symbol.")
  parser.add_argument("--check_special_token", type="bool", default=True,
                      help="""\
                      Whether check special sos, eos, unk tokens exist in the
                      vocab files.\
                      """)

  # Sequence lengths
  parser.add_argument("--src_max_len", type=int, default=80,
                      help="Max length of src sequences during training.")
  parser.add_argument("--tgt_max_len", type=int, default=50,
                      help="Max length of tgt sequences during training.")
  parser.add_argument("--src_max_len_infer", type=int, default=None,
                      help="Max length of src sequences during inference.")
  parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                      help="""\
      Max length of tgt sequences during inference.  Also use to restrict the
      maximum decoding length.\
      """)

  # Default settings works well (rarely need to change), unit_type is LSTM
  parser.add_argument("--forget_bias", type=float, default=1.0,
                      help="Forget bias for BasicLSTMCell.")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (not keep_prob)")
  parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
  parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

  parser.add_argument("--steps_per_stats", type=int, default=100,
                      help=("How many training steps to do per stats logging."
                            "Save checkpoint every 10x steps_per_stats"))
  parser.add_argument("--num_buckets", type=int, default=5,
                      help="Put data into similar-length buckets.")

  # Misc
  parser.add_argument("--log_device_placement", type="bool", nargs="?",
                      const=True, default=False, help="Debug GPU allocation.")
  parser.add_argument("--metrics", type=str, default="bleu",
                      help=("Comma-separated list of evaluations "
                            "metrics (bleu,rouge,accuracy)"))
  parser.add_argument("--steps_per_external_eval", type=int, default=None,
                      help="""\
      How many training steps to do per external evaluation.  Automatically set
      based on data if None.\
      """)

  parser.add_argument("--hparams_path", type=str, default=None,
                      help=("Path to standard hparams json file that overrides"
                            "hparams values from FLAGS."))
  parser.add_argument("--random_seed", type=int, default=None,
                      help="Random seed (>0, set a specific seed).")

  # Inference
  parser.add_argument("--ckpt", type=str, default="",
                      help="Checkpoint file to load a model for inference.")
  parser.add_argument("--inference_input_file", type=str, default=None,
                      help="Set to the text to decode.")
  parser.add_argument("--infer_batch_size", type=int, default=32,
                      help="Batch size for inference mode.")
  parser.add_argument("--inference_output_file", type=str, default=None,
                      help="Output file to store decoding results.")
  parser.add_argument("--inference_ref_file", type=str, default=None,
                      help=("""\
      Reference file to compute evaluation scores (if provided).\
      """))
  parser.add_argument("--beam_width", type=int, default=10,
                      help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
  parser.add_argument("--num_translations_per_input", type=int, default=1,
                      help=("""\
      Number of translations generated for each sentence. This is only used for
      inference.\
      """))

def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      src=flags.src,
      tgt=flags.tgt,
      train_prefix=flags.train_prefix,
      dev_prefix=flags.dev_prefix,
      test_prefix=flags.test_prefix,
      vocab_prefix=flags.vocab_prefix,
      out_dir=flags.out_dir,

      # Networks
      num_units=flags.num_units,
      num_layers=flags.num_layers,
      dropout=flags.dropout,
      time_major=flags.time_major,

      # Train
      num_train_steps=flags.num_train_steps,
      batch_size=flags.batch_size,
      init_op=flags.init_op,
      init_weight=flags.init_weight,
      max_gradient_norm=flags.max_gradient_norm,
      learning_rate=flags.learning_rate,
      decay_scheme=flags.decay_scheme,

      # Data constraints
      num_buckets=flags.num_buckets,
      src_max_len=flags.src_max_len,
      tgt_max_len=flags.tgt_max_len,

      # Inference
      src_max_len_infer=flags.src_max_len_infer,
      tgt_max_len_infer=flags.tgt_max_len_infer,
      infer_batch_size=flags.infer_batch_size,
      beam_width=flags.beam_width,
      num_translations_per_input=flags.num_translations_per_input,

      # Vocab
      sos=flags.sos if flags.sos else vocab_utils.SOS,
      eos=flags.eos if flags.eos else vocab_utils.EOS,
      check_special_token=flags.check_special_token,

      # Misc
      forget_bias=flags.forget_bias,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=flags.steps_per_stats,
      steps_per_external_eval=flags.steps_per_external_eval,
      metrics=flags.metrics.split(","),
      log_device_placement=flags.log_device_placement,
      random_seed=flags.random_seed,
      num_keep_ckpts=5,  # saves 5 checkpoints by default.
  )


def run_main(flags, hparams, train_fn, inference_fn):
  """Run main."""
  # Random
  random_seed = flags.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

  ## Train / Decode
  out_dir = flags.out_dir
  if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

  if flags.inference_input_file:
    # Inference
    trans_file = flags.inference_output_file
    ckpt = flags.ckpt
    if not ckpt:
      ckpt = tf.train.latest_checkpoint(out_dir)

    # inference_fn has no "job_id" parameter
    inference_fn(ckpt, flags.inference_input_file,
                 trans_file, hparams, num_workers)

    # Evaluation
    ref_file = flags.inference_ref_file
    if ref_file and tf.gfile.Exists(trans_file):
      for metric in hparams.metrics:
        score = evaluation_utils.evaluate(
            ref_file,
            trans_file,
            metric,
            hparams.subword_option)
        utils.print_out("  %s: %.1f" % (metric, score))
  else:
    # Train
    train_fn(hparams)

def train_test(hparams):
  print("Training is OK!")

def infer_test(ckpt, inference_input_file,
                 trans_file, hparams, num_workers):
  print("Inference is OK!")


def main(unused_argv):
  default_hparams = create_hparams(FLAGS)
  # train_fn = train.train
  # inference_fn = inference.inference
  train_fn = train_test
  inference_fn = infer_test
  run_main(FLAGS, default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)