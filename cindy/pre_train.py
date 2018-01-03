# -*- coding: utf-8 -*-
#
# Author: Synrey Yee
#
# Created at: 12/28/2017
#
# Description: generate training relevant files for translation
#
# Last Modified at: 12/29/2017, by: Synrey Yee

from __future__ import print_function

from collections import defaultdict

import codecs
import argparse
import pickle

import subprocess as sbp
import threading
import os
import platform


def CorpusCheck(src_raw, tgt_raw):
  src_num_lines = 0
  tgt_num_lines = 0
  # count the number of lines in both files,
  # which needs a linux shell command, if could.
  isLinux = (platform.system() == "Linux")
  if isLinux:
    args = ['wc', '-l', src_raw, tgt_raw]
    cmd = sbp.Popen(args, stdout = sbp.PIPE)
    res, _ = cmd.communicate()
    src_num_lines = int(res.split()[0])
    tgt_num_lines = int(res.split()[2])

  else:
    with open(src_raw, 'r') as src_file:
      for line in src_file:
        src_num_lines += 1

    with open(tgt_raw, 'r') as tgt_file:
      for line in tgt_file:
        tgt_num_lines += 1

  assert (src_num_lines == tgt_num_lines)
  return src_num_lines


def GenerateFiles(raw_file, vob_path, vob_dict_path,
              train_file, eval_file, test_file,
              step, freq):

  tr_file = codecs.open(train_file, 'w', "utf-8")
  ev_file = codecs.open(eval_file, 'w', "utf-8")
  tt_file = codecs.open(test_file, 'w', "utf-8")

  vob_dict = defaultdict(int)

  inp = codecs.open(raw_file, 'r', "utf-8")
  with tr_file, ev_file, tt_file:
    iseval = True
    for i, line in enumerate(inp):
      line = line.strip()

      if (i % step) == 0:
        if iseval:
          ev_file.write(line + u'\n')
          iseval = False
        else:
          tt_file.write(line + u'\n')
          iseval = True
      else:
        tr_file.write(line + u'\n')

      words = line.split()
      for wd in words:
        vob_dict[wd] += 1

  inp.close()

  with open(vob_dict_path, 'wb') as opt:
    pickle.dump(vob_dict, opt)

  with codecs.open(vob_path, 'w', "utf-8") as vob_file:
    vob_file.write(u"<unk>\n")
    vob_file.write(u"<s>\n")
    vob_file.write(u"</s>\n")
    # Traverse the vob_dict
    for key, value in vob_dict.items():
      # We need the words with each frequency greater than "freq".
      if value > freq:
        vob_file.write(key + u'\n')

  print("Generating file %s finished" % raw_file)


def main(args):
  src_raw = args.src_tgt_raw + '.' + args.src_suffix
  tgt_raw = args.src_tgt_raw + '.' + args.tgt_suffix

  src_vob_path = args.vob_path + '.' + args.src_suffix
  tgt_vob_path = args.vob_path + '.' + args.tgt_suffix

  src_vob_dict_path = args.vob_dict_path + '_' + args.src_suffix + ".pk"
  tgt_vob_dict_path = args.vob_dict_path + '_' + args.tgt_suffix + ".pk"

  src_train = args.train_file + '.' + args.src_suffix
  tgt_train = args.train_file + '.' + args.tgt_suffix

  src_eval = args.eval_file + '.' + args.src_suffix
  tgt_eval = args.eval_file + '.' + args.tgt_suffix

  src_test = args.test_file + '.' + args.src_suffix
  tgt_test = args.test_file + '.' + args.tgt_suffix

  assert os.path.exists(src_raw)
  assert os.path.exists(tgt_raw)

  num_sents = CorpusCheck(src_raw, tgt_raw)
  num_eval = num_sents * args.percentage
  num_test = num_sents * args.percentage
  step = num_sents / (num_eval + num_test)

  src_args = (src_raw, src_vob_path, src_vob_dict_path, src_train,
          src_eval, src_test, step, args.src_freq)
  tgt_args = (tgt_raw, tgt_vob_path, tgt_vob_dict_path, tgt_train,
          tgt_eval, tgt_test, step, args.tgt_freq)

  t1 = threading.Thread(target = GenerateFiles, args = src_args)
  t2 = threading.Thread(target = GenerateFiles, args = tgt_args)

  t1.setDaemon(True)
  t2.setDaemon(True)
  t1.start()
  t2.start()

  t1.join()
  t2.join()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # input
  parser.add_argument(
    "--src_tgt_raw",
    type = str,
    default = "/home/synrey/data/casia2015",
    help = "source or target file prefix")

  # output, training files
  parser.add_argument(
    "--vob_path",
    type = str,
    default = "/home/synrey/data/Snmt/vocab",
    help = "vector's file prefix")
  parser.add_argument(
    "--vob_dict_path",
    type = str,
    default = "/home/synrey/data/Snmt/vob_dict",
    help = "vector's python dict prefix, dumped by pickle")

  parser.add_argument(
    "--train_file",
    type = str,
    default = "/home/synrey/data/Snmt/train",
    help = "training file prefix")

  # output, eval file
  parser.add_argument(
    "--eval_file",
    type = str,
    default = "/home/synrey/data/Snmt/eval",
    help = "eval file prefix")

  # output, test file
  parser.add_argument(
    "--test_file",
    type = str,
    default = "/home/synrey/data/Snmt/test",
    help = "test file prefix")

  # parameters
  parser.add_argument(
    "--src_suffix",
    type = str,
    default = "ch",
    help = "source file suffix")
  parser.add_argument(
    "--tgt_suffix",
    type = str,
    default = "en",
    help = "target file suffix")
  parser.add_argument(
    "--src_freq",
    type = int,
    default = 2,
    help = "source file word frequency")
  parser.add_argument(
    "--tgt_freq",
    type = int,
    default = 9,
    help = "target file word frequency")

  parser.add_argument(
    "--percentage",
    type = float,
    default = 0.01,
    help = "the eval or test file percentage with respect to original corpora")

  args = parser.parse_args()
  main(args)