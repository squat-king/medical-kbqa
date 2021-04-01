import os

import pandas as pd
from collections import Counter
import random
import logging
from logging import handlers

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification,BertConfig
from tqdm import tqdm, trange
from similarity_model.utils import *
# from bert_chinese_encode import get_bert_encode_for_single
import torch
import torch.nn as nn
import math
import datetime
import matplotlib.pyplot as plt
import argparse
import time

if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_train",
                        required=False,
                        default=True,
                        action='store_true',
                        help="do predict")
    parser.add_argument("--do_eval",
                        required=False,
                        default=True,
                        action='store_true',
                        help="do eval")
    parser.add_argument("--do_predict",
                        required=False,
                        default=False,
                        action='store_true',
                        help="do predict")
    parser.add_argument("-lr",
                        "--learning_rate",
                        required=False,
                        default=0.0001,
                        type=float,
                        help="learning_rate")
    parser.add_argument("--epoch",
                        required=False,
                        default=500,
                        type=int,
                        help="epoch")
    parser.add_argument("-bs",
                        "--batch_size",
                        required=False,
                        default=500,
                        type=int,
                        help="batch size")
    parser.add_argument("--num_atten_head",
                        required=False,
                        default=12,
                        type=int,
                        help="num attention heads")
    parser.add_argument("--num_layers",
                        required=False,
                        default=12,
                        type=int,
                        help="num hidden layers")
    parser.add_argument("--seed",
                        required=False,
                        default=1,
                        type=int,
                        help="seed")
    parser.add_argument("--patience",
                        required=False,
                        default=100,
                        type=int,
                        help="seed")
    parser.add_argument("--output_dir",
                        default="./output",
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        required=False,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--hidden_size",
                        required=False,
                        default=768,
                        type=int,
                        help="num hidden layers")
    args = parser.parse_args()

    # 读取数据
    train_data_path = 'data/train.txt'
    eval_data_path = 'data/dev.txt'

    hidden_size = 768
    # 预训练模型bert输出的维度
    input_size = 768
    n_categories = 3
    BATCH_SIZE = 500
    # 把学习率设定为0.005
    learning_rate = 0.005
    EPOCH = 500
    MAX_SEQ_LENGTH = 128
    PATIENTS=100

    num_attention_heads=12
    num_hidden_layers=12
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        EPOCH = args.epoch
    if args.max_seq_length is not None:
        MAX_SEQ_LENGTH = args.max_seq_length
    if args.patience is not None:
        PATIENTS = args.patience
    if args.hidden_size is not None:
        hidden_size = args.hidden_size
    if args.num_atten_head is not None:
        num_attention_heads=args.num_atten_head
    if args.num_layers is not None:
        num_hidden_layers = args.num_layers

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = handlers.RotatingFileHandler(
        filename='log.similarity_model.' + datetime.datetime.now().strftime('%Y-%m-%d') + '.' + str(
            time.time()) + '_epoch_' + str(EPOCH) + '_bs_' + str(BATCH_SIZE) + '_lr_' + str(learning_rate)
                 + '_max_seq_length_' + str(MAX_SEQ_LENGTH),
        mode='w', backupCount=3, encoding='utf-8')
    format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    handler.setLevel(logging.INFO)
    handler.setFormatter(format_str)
    logger.addHandler(handler)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    config=BertConfig.from_pretrained(args.bert_model)
    config.hidden_size=hidden_size
    config.num_hidden_layers=num_hidden_layers
    config.num_attention_heads=num_attention_heads

    model = BertForSequenceClassification(config)
    model = trans_to_cuda(model)

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = Adam(optimizer_grouped_parameters,
                     lr=learning_rate)

    if args.do_train:
        train_examples = get_examples(train_data_path)
        train_features = convert_examples_to_features(
           train_examples, tokenizer, MAX_SEQ_LENGTH)
        logger.info("***** Loading train data*****")
        logger.info("  Num train examples = %d", len(train_features))
        train_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        train_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        train_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        train_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_dataset = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # 打印正负标签比例
        # print(dict(Counter(train_data[0].values)))
    if args.do_eval:
        eval_examples = get_examples(eval_data_path)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, MAX_SEQ_LENGTH)
        logger.info("***** Loading eval data*****")
        logger.info("  Num eval examples = %d", len(eval_features))
        eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        eval_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        eval_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_dataset = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label)
        eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_eval_acc = 0.0
    global_step = 0  # 表示总共训练了多少个batch
    # 获取整个训练的开始时间
    start = time.time()
    # 为后续的画图做准备，存储每次打印间隔之间的平均损失和平均准确率
    all_train_loss = []
    all_train_acc = []
    all_eval_loss = []
    all_eval_acc = []

    patience=0
    for epoch in trange(int(EPOCH), desc="Epoch"):
        model.train()
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        logger.info("Epoch = " + str(epoch) + "| TimeSince: " + timeSince(start))
        train_loss = 0
        train_accuracy = 0
        train_examples, train_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.cuda() if not args.no_cuda else t for t in batch)  # 如果指定了no_cuda参数，就用cpu类型
            input_ids, input_mask, segment_ids, label_ids = batch  # batch={tuple:4}
            # flatten,[batch_size , 1 , max_seq_len] --> [batch_size , max_seq_len]
            input_ids_flat = input_ids.view(-1, input_ids.size(-1))
            input_mask_flat = input_mask.view(-1, input_mask.size(-1))
            segment_ids_flat = segment_ids.view(-1, segment_ids.size(-1))
            # 2020-12-02 basilwang Migrating from pytorch-pretrained-bert to transformers
            if args.no_cuda:
                loss, train_logits = model(input_ids_flat, token_type_ids=segment_ids_flat,
                                           attention_mask=input_mask_flat,
                                           labels=label_ids)
            else:
                loss, train_logits = model(input_ids_flat.cuda(), token_type_ids=segment_ids_flat.cuda(),
                                       attention_mask=input_mask_flat.cuda(),
                                       labels=label_ids.cuda())
            # 第一个输出是loss损失值，第二个输出是全连接层的输出值，即logits,shape=[batch_size,num_labels]

            train_logits = train_logits.view(-1, 2)
            train_logits = train_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            acc_tmp = accuracy(train_logits, label_ids)
            train_accuracy += acc_tmp
            train_loss += loss.item()
            train_examples += input_ids.size(0)
            train_steps += 1
            if step % 10 == 0:  # 每训练10个batch输出一次loss
                logger.info('Batch %d , train_loss = %.4f, train_accuracy = %.4f' % (global_step + 1 , loss , acc_tmp))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if args.do_eval and step % 10 == 0:  # 每训练100个batch进行一次eval,计算一次accuracy
                eval_loss, eval_accuracy = do_evaluation(model, eval_dataloader, args.no_cuda)
                # 将损失，准确率的结果保存起来，为后续的画图使用
                train_accuracy = train_accuracy / train_examples
                train_loss = train_loss / train_steps

                all_train_loss.append(train_loss)
                all_train_acc.append(train_accuracy)
                all_eval_loss.append(eval_loss)
                all_eval_acc.append(eval_accuracy)

                train_accuracy = 0
                train_loss = 0
                train_examples = 0
                train_steps = 0
                # logger.info('Batch %d , train_loss = %.4f, train_accuracy = %.4f' % (global_step, eval_loss, eval_accuracy))
                logger.info('Batch %d , eval_loss  = %.4f, eval_accuracy  = %.4f' % (global_step + 1, eval_loss, eval_accuracy))

                if best_eval_acc < eval_accuracy:
                    best_eval_acc = eval_accuracy
                    logger.info('best_eval_acc= %.4f at training batch = %.4f' % (best_eval_acc, global_step))
                    save_name = f"pytorch_model_best_acc_%.4f.bin" % best_eval_acc
                    # torch.save(model.state_dict(), os.path.join(args.output_dir, f"pytorch_model_best_acc.bin"))
                    torch.save(model.state_dict(), os.path.join(args.output_dir, save_name))
                    patience=0
                else:
                    patience += 1
                if patience > PATIENTS:
                    break
        if patience > PATIENTS:
            break
    model_save_dir = os.path.join(args.output_dir, f'model{global_step}')
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_dir, f"pytorch_model.bin"))
    logger.info("after " + str(global_step) + " batches , model has been saved to " + model_save_dir + " bingo!!")

    text_save("./output/all_train_acc.txt", all_train_acc)
    text_save("./output/all_train_loss.txt", all_train_loss)
    text_save("./output/all_eval_acc.txt", all_eval_acc)
    text_save("./output/all_eval_loss.txt", all_eval_loss)

    # plt.figure(0)
    # # plt.plot(all_train_loss, label="Train Loss")
    # plt.plot(all_eval_loss, color="red", label="Eval Loss")
    # plt.legend(loc="upper left")
    # plt.grid(True)
    # plt.savefig("./loss.png")
    #
    # plt.figure(1)
    # # plt.plot(all_train_acc, label="Train Acc")
    # plt.plot(all_eval_acc, color="red", label="Eval Acc")
    # plt.legend(loc="upper left")
    # plt.grid(True)
    # plt.savefig("./acc.png")

    # 模型的保存，首先给定保存的路径
    # MODEL_PATH = './BERT_RNN.pth'

    # torch.save(rnn.state_dict(), MODEL_PATH)


def text_save(filename, data):
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print(filename + " saved!")


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


if __name__ == "__main__":
    main()
