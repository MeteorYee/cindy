# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 12/30/2017
#
# Last Modified at: 12/31/2017, by: Synrey Yee

#######################<=Description=>##########################
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

"""Utility functions for building models."""
from __future__ import print_function

import collections
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from . import attmodel
from .third_party import iterator_utils
from .third_party import misc_utils as utils
from .third_party import vocab_utils


class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
  pass


def create_train_model(hparams):
  """Create train graph, model, and iterator."""
  src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
  tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  graph = tf.Graph()

  with graph.as_default(), tf.container("train"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, share_vocab = None)

    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len)

    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,)

  return TrainModel(
      graph=graph,
      model=model,
      iterator=iterator)


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "src_file_placeholder",
                            "tgt_file_placeholder", "iterator"))):
  pass


def create_eval_model(hparams):
  """Create train graph, model, src/tgt file holders, and iterator."""
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file
  graph = tf.Graph()

  with graph.as_default(), tf.container("eval"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, share_vocab = None)
    src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    src_dataset = tf.data.TextLineDataset(src_file_placeholder)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len_infer,
        tgt_max_len=hparams.tgt_max_len_infer)
    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.EVAL,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table)
  return EvalModel(
      graph=graph,
      model=model,
      src_file_placeholder=src_file_placeholder,
      tgt_file_placeholder=tgt_file_placeholder,
      iterator=iterator)


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "src_placeholder",
                            "batch_size_placeholder", "iterator"))):
  pass


def create_infer_model(hparams):
  """Create inference model."""
  graph = tf.Graph()
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  with graph.as_default(), tf.container("infer"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, share_vocab = None)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)

    src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

    src_dataset = tf.data.Dataset.from_tensor_slices(
        src_placeholder)
    iterator = iterator_utils.get_infer_iterator(
        src_dataset,
        src_vocab_table,
        batch_size=batch_size_placeholder,
        eos=hparams.eos,
        src_max_len=hparams.src_max_len_infer)
    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.INFER,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        reverse_target_vocab_table=reverse_tgt_vocab_table)
  return InferModel(
      graph=graph,
      model=model,
      src_placeholder=src_placeholder,
      batch_size_placeholder=batch_size_placeholder,
      iterator=iterator)


def create_emb_for_encoder_and_decoder(src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32):
  """Create embedding matrix for both encoder and decoder.

  Args:
    src_vocab_size: An integer. The source vocab size.
    tgt_vocab_size: An integer. The target vocab size.
    src_embed_size: An integer. The embedding dimension for the encoder's
      embedding.
    tgt_embed_size: An integer. The embedding dimension for the decoder's
      embedding.
    dtype: dtype of the embedding matrix. Default to float32.

  Returns:
    embedding_encoder: Encoder's embedding matrix.
    embedding_decoder: Decoder's embedding matrix.

  """
  with tf.variable_scope("embeddings", dtype=dtype) as scope:
    with tf.variable_scope("encoder"):
      embedding_encoder = tf.get_variable(
          "embedding_encoder", [src_vocab_size, src_embed_size], dtype)

    with tf.variable_scope("decoder"):
      embedding_decoder = tf.get_variable(
          "embedding_decoder", [tgt_vocab_size, tgt_embed_size], dtype)

  return embedding_encoder, embedding_decoder


def create_rnn_cell(num_units, num_layers, forget_bias,
                dropout, mode, base_gpu=0):
  """Create multi-layer RNN cell.

  Args:
    num_units: the depth of each unit.
    num_layers: number of cells.
    forget_bias: the initial forget bias of the RNNCell(s).
    dropout: floating point value between 0.0 and 1.0:
      the probability of dropout.  this is ignored if `mode != TRAIN`.
    mode: either tf.contrib.learn.TRAIN/EVAL/INFER
    base_gpu: The gpu device id to use for the first RNN cell in the
      returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
      as its device id.
  Returns:
    An `RNNCell` instance.
  """
  cell_list = _cell_list(num_units=num_units,
                         num_layers=num_layers,
                         forget_bias=forget_bias,
                         dropout=dropout,
                         mode=mode,
                         base_gpu=base_gpu)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list)


def _cell_list(num_units, num_layers, forget_bias,
            dropout, mode, base_gpu=0):
  """Create a list of RNN cells."""

  cell_list = []
  for i in range(num_layers):
    utils.print_out("  cell %d" % i, new_line=False)

    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
    single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units,
        forget_bias=forget_bias)

    if dropout > 0.0:
      single_cell = tf.contrib.rnn.DropoutWrapper(
          cell=single_cell, input_keep_prob=(1.0 - dropout))
      utils.print_out("  %s, dropout=%g " %(type(single_cell).__name__, dropout),
                      new_line=False)

    utils.print_out("")
    cell_list.append(single_cell)

  return cell_list
