# Cindy
* For Chinese-English machine translation.
* Seq2seq + Luong Attention, encoder: bi-LSTM, decoder: 2-layer LSTM.
* A mini version of Google's seq2seq [nmt](https://github.com/tensorflow/nmt).

**Still updating...**

## Demo
### Corpus Handling
**I.**  Get [casia2015 corpus](http://nlp.nju.edu.cn/cwmt-wmt/). Rename the parallel corpus to casia2015.ch (for Chinese) and casia2015.en (for English). For Chinese part, segment the sentences using a word segmentation tool, for example: [foolNLTK](https://github.com/rockyzhengwu/FoolNLTK); For English part, make words and punctuations separated by space, such as "word, => word ,".

**II.**  Generate training, test, validation, vocabulary files by cindy/pre_train.py. You may refer to it for arguments configuration. Note the file path should be absolute path.

### Demo Training
> python -m cindy.cindy \\<br>
    --src=ch --tgt=en \\<br>
    --vocab_prefix=/home/synrey/data/Snmt_word/vocab  \\<br>
    --train_prefix=/home/synrey/data/Snmt_word/train \\<br>
    --dev_prefix=/home/synrey/data/Snmt_word/eval  \\<br>
    --test_prefix=/home/synrey/data/Snmt_word/test \\<br>
    --out_dir=/home/synrey/data/Snmt_word/nmt_model \\<br>
    --num_train_steps=82000 \\<br>
    --steps_per_stats=100 \\<br>
    --num_layers=2 \\<br>
    --num_units=512 \\<br>
    --dropout=0.2 \\<br>
    --metrics=bleu

"/home/synrey/data/Snmt_word" is where I put my corpus files.

### Result (BLEU value)
Translation examples:<br>
**src**:  这样 的 幽默 对 一个 热情 活力 四 射 的 金发 女郎 来说 ， 意思 可能 有点 晦涩 ， 但是 我 知道 你 明白 我 的 意思 的 。<br>
**ref**:  The humor here may be a bit dark for a bubbly blond but I think you get my drift .<br>
**nmt**:  A sense of humor is a bit obscure for a <unk> blonde , but I know what I mean .<br>

**src**:  因为 人 屡次 用 脚镣 和 铁链 捆锁 他 ， 铁链 竟 被 他 挣断 了 ， 脚镣 也 被 他 弄 碎 了 。<br>
**ref**:  For he had often been chained hand and foot , but he tore the chains apart and broke the irons on his feet .<br>
**nmt**:  For men were bound with fetters and chains , and chains were crushed by him ; and the fetters were shattered .

* Current BLEU value is **16.3**.
<br>
In the future, a more accurate word segmentation tool is needed for Chinese sentences, add subword option, ...
