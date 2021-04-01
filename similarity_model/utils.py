import codecs
import math
import os

import numpy as np
from similarity_model.train_bert import *
from bert import tokenization


def timeSince(since):
    # 功能:获取每次打印的时间消耗, since是训练开始的时间
    # 获取当前的时间
    now = time.time()

    # 获取时间差, 就是时间消耗
    s = now - since

    # 获取时间差的分钟数
    m = math.floor(s / 60)

    # 获取时间差的秒数
    s -= m * 60

    return '%dm %ds' % (m, s)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def do_evaluation(model, eval_dataloader, no_cuda):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    eval_steps, eval_examples = 0, 0
    eval_logits_all_batchs = None
    with torch.no_grad():
        for _,(input_ids, input_mask, segment_ids, label_ids) in enumerate(eval_dataloader):
            if not no_cuda:
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()
                label_ids = label_ids.cuda()
            input_ids_flat = input_ids.view(-1, input_ids.size(-1))
            input_mask_flat = input_mask.view(-1, input_mask.size(-1))
            segment_ids_flat = segment_ids.view(-1, segment_ids.size(-1))
            outputs = model(input_ids_flat, token_type_ids=segment_ids_flat, attention_mask=input_mask_flat,
                            labels=label_ids)
            batch_eval_loss, logits = outputs[:2]
            logits = logits.view(-1, 2)
            # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            # logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            if eval_logits_all_batchs is None:
                eval_logits_all_batchs = logits.copy()
            else:
                eval_logits_all_batchs = np.vstack((eval_logits_all_batchs, logits))  # 将logits数组按垂直方向叠加
            label_ids = label_ids.to('cpu').numpy()
            # 累加损失、验证的准确率
            batch_eval_accuracy = accuracy(logits, label_ids)

            print("本次eval预测了%d个,正确数量%d"%(input_ids.size(0),batch_eval_accuracy))
            eval_loss += batch_eval_loss.mean().item()
            eval_accuracy += batch_eval_accuracy

            eval_examples += input_ids.size(0)
            eval_steps += 1
    print("eval_examples = %d , right = %d"%(eval_examples,eval_accuracy))
    eval_loss = eval_loss / eval_steps  # 评估loss=平均batch loss=Σbatch loss/batch数量
    eval_accuracy = eval_accuracy / eval_examples  # 评估accuracy=预测对的样本数量/所有预测样本数量
    return eval_loss, eval_accuracy


def select_field(InputFeatures, field):
    return [
        [
            feature[field]
            for feature in InputFeature.features_arr
        ]
        for InputFeature in InputFeatures
    ]  # [total_sample_len , 1 , max_seq_len]


def _truncate_seq_pair(tokens_a, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            return tokens_a
        else:
            tokens_a = tokens_a[:max_length]
            return tokens_a




def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads iterable data into a list of `InputBatch`s."""

    # roc is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for i, example in enumerate(examples):
        text_a=example.text_a
        text_b=example.text_b
        label =int(example.label)
        text_a_tokens = tokenizer.tokenize(text_a)
        text_b_tokens = tokenizer.tokenize(text_b)

        entity_features = []

        # Modifies `entity_tokens` and `ending_tokens` in
        # place so that the total length is less than the
        # specified length.  Account for [CLS], [SEP], [SEP] with
        # "- 3"
        text_a_tokens = _truncate_seq_pair(text_a_tokens, max_seq_length -len(text_b)- 3)# 优先对text_a截断

        entity_tokens_with_cls_sep = ["[CLS]"] + text_a_tokens + ["[SEP]"]+text_b_tokens+["[SEP]"]

        segment_ids = [0] * (len(text_a_tokens) + 2)
        segment_ids +=[1]*(len(text_b_tokens) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(entity_tokens_with_cls_sep)  # max_seq_length个tokenId ，不足的补0
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))  # padding补0
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        entity_features.append((entity_tokens_with_cls_sep, input_ids, input_mask, segment_ids))

        features.append(
            InputFeature(
                entity_features=entity_features,
                label=label
            )
        )
    return features


class InputFeature(object):
    def __init__(self, entity_features, label):
        # with gzip.open(filename, 'rt')  as f:
        #     reader = csv.reader(f)
        #     rows = list(reader)
        self.features_arr = [
            {'tokens_with_cls_sep': tokens_with_cls_sep,
             'input_ids': input_ids,
             'input_mask': input_mask,
             'segment_ids': segment_ids
             }
            for tokens_with_cls_sep, input_ids, input_mask, segment_ids in entity_features
        ]

        self.label = label

def get_examples(data_dir):
    return create_examples(
        read_data(os.path.join(data_dir)))

def read_data(input_file):
    with codecs.open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f:
          lines.append(line.strip().split(" "))
        return lines

def create_examples(lines):
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        text_a = tokenization.convert_to_unicode(line[0])
        text_b = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[-1])
        examples.append(
            InputExample(text_a=text_a, text_b=text_b, label=label))
    return examples

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label