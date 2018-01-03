# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 12/30/2017
#
# Description: Training script
#
# Last Modified at: 12/30/2017, by: Synrey Yee

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

import tensorflow as tf

from .third_party import misc_utils as utils

utils.check_tensorflow_version()

def train(hparams):
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_stats
  if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval
