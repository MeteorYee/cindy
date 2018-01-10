# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 12/30/2017
#
# Description: Training script
#
# Last Modified at: 01/05/2018, by: Synrey Yee

################################################################
###                                                          ###
### This file is based on the same file of Google's seq2seq. ###
###           (https://github.com/tensorflow/nmt)            ###
###            Hence, here is its license below.             ###
###                                                          ###
################################################################

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

import codecs
import time

import tensorflow as tf

from . import attmodel as attention_model
from . import model_helper
from .third_party import misc_utils as utils
from .third_party import nmt_utils


def load_data(inference_input_file, hparams=None):
  """Load inference data."""
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(inference_input_file, mode="rb")) as f:
    inference_data = f.read().splitlines()

  return inference_data


def inference(ckpt,
              inference_input_file,
              inference_output_file,
              hparams):
  """Perform translation."""
  model_creator = attention_model.AttModel

  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  single_worker_inference(
      infer_model,
      ckpt,
      inference_input_file,
      inference_output_file,
      hparams)


def single_worker_inference(infer_model,
                            ckpt,
                            inference_input_file,
                            inference_output_file,
                            hparams):
  """Inference with a single worker."""
  output_infer = inference_output_file

  # Read data
  infer_data = load_data(inference_input_file, hparams)

  with tf.Session(
      graph=infer_model.graph, config=utils.get_config_proto()) as sess:
    loaded_infer_model = model_helper.load_model(
        infer_model.model, ckpt, sess, "infer")
    sess.run(
        infer_model.iterator.initializer,
        feed_dict={
            infer_model.src_placeholder: infer_data,
            infer_model.batch_size_placeholder: hparams.infer_batch_size
        })
    # Decode
    utils.print_out("# Start decoding")

    nmt_utils.decode_and_evaluate(
        "infer",
        loaded_infer_model,
        sess,
        output_infer,
        ref_file=None,
        metrics=hparams.metrics,
        subword_option=None,
        beam_width=hparams.beam_width,
        tgt_eos=hparams.eos,
        num_translations_per_input=hparams.num_translations_per_input)
