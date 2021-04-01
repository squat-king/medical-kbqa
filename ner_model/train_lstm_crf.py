# 导入关键的工具包
import argparse
import torch.optim as optim
# 导入自定义的类和评估函数
from ner_model.bilstm_crf import BiLSTM_CRF
from ner_model.utils import *
# 导入工具包
from tqdm import tqdm, trange
import json
import time
import logging
from logging import handlers

time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def do_evaluate(model, eval_data_list, char_to_id, label_to_idx, id_to_char, id_to_tag):
    # 每一个epoch, 训练结束后直接在验证集上进行验证
    total_acc_num, total_predict_num, totao_gold_num, total_loss = 0, 0, 0, 0
    # 进入验证阶段最重要的一步就是保持模型参数不变, 不参与反向传播和参数更新
    with torch.no_grad():
        for _,eval_data in enumerate(eval_data_list):
            # 直接提取验证集的特征和标签, 并进行数字化映射
            tokens, tags = eval_data.get("text"), eval_data.get("label")
            idx_of_tokens = prepare_sequence(tokens, char_to_id)
            labels = torch.tensor([label_to_idx[t] for t in tags], dtype=torch.long)
            # 验证阶段的损失值依然是通过neg_log_likelihood函数进行计算
            loss = model.neg_log_likelihood(idx_of_tokens, labels)
            # 验证阶段的解码依然是通过直接调用model中的forward函数进行
            score, best_path_list = model(idx_of_tokens)
            # 累加每一个样本的损失值
            step_loss = loss.data.numpy()
            total_loss += step_loss
            idx_of_tokens_unq = idx_of_tokens.unsqueeze(0)
            labels_unq = labels.unsqueeze(0)
            best_path_list_up = [best_path_list]
            step_acc, step_recall, f1_score, num_acc_entities, num_predict_entities, num_gold_entities = evaluate(
                idx_of_tokens_unq.tolist(), labels_unq.tolist(), best_path_list_up, id_to_char, id_to_tag)
            # 累加三个重要的统计量, 预测正确的实体数, 预测的总实体数, 真实标签的实体数
            total_acc_num += num_acc_entities
            total_predict_num += num_predict_entities
            totao_gold_num += num_gold_entities

    # logger.info("eval:", total_acc_num, total_predict_num, totao_gold_num)
    eval_mean_loss = total_loss / len(eval_data_list)
    # 当准确预测的数量大于0, 并且总的预测标签量大于0, 计算验证集上的准确率, 召回率, F1值
    if total_acc_num > 0 and total_predict_num > 0:
        eval_epoch_acc = total_acc_num / total_predict_num
        eval_epoch_recall = total_acc_num / totao_gold_num
        eval_epoch_f1 = 2 * eval_epoch_acc * eval_epoch_recall / (eval_epoch_acc + eval_epoch_recall)
        return eval_mean_loss, eval_epoch_acc, eval_epoch_recall, eval_epoch_f1
    else:
        return eval_mean_loss,0, 0, 0

def text_save(filename, data):
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    logger.info(filename + " saved!")


# 构建文本到id的映射函数
def prepare_sequence(seq, char_to_id):
    # 初始化空列表
    char_ids = []
    # 遍历中文字符串, 完成数字化的映射
    for i, ch in enumerate(seq):
        # 判断当前字符是都在映射字典中, 如果不在取UNK字符的编码来代替, 否则取对应的字符编码数字
        if char_to_id.get(ch):
            char_ids.append(char_to_id[ch])
        else:
            char_ids.append(char_to_id["UNK"])
    # 将列表封装成Tensor类型返回
    return torch.tensor(char_ids, dtype=torch.long)


# 指定字符编码表的文件路径
json_file = 'data/char_to_id.json'


# 添加获取训练数据和验证数据的函数
def get_data():
    # 设置训练数据的路径和验证数据的路径
    train_data_file_path = "data/train_data.txt"
    eval_data_file_path = "data/dev_data.txt"
    # 初始化数据的空列表
    train_data_list = []
    valid_data_list = []
    # 因为每一行都是一个样本, 所以按行遍历文件即可
    for line in open(train_data_file_path, mode='r', encoding='utf-8'):
        # 每一行数据都是json字符串, 直接loads进来即可
        data = json.loads(line)
        train_data_list.append(data)
    # 同理处理验证数据集
    for line in open(eval_data_file_path, mode='r', encoding='utf-8'):
        data = json.loads(line)
        valid_data_list.append(data)
    # 最后以列表的形式返回训练数据集和验证数据集
    return train_data_list, valid_data_list  # 每个list里存放了data_num个dict,每个dict里有2对key-value，{'text':[],'label':[]}

# 构建函数评估模型的准确率和召回率, F1值
def evaluate(idx_of_tokens, labels, predict_labels, id2char, id2tag):
    # idx_of_tokens: 代表待评估 的原始样本,shape=[1,seq_len]
    # labels: 真实的标签序列,shape=[1,seq_len]
    # predict_labels: 模型预测出来的标签序列,即best_path_list_up,shape=[1,seq_len]
    # id2char: 代表数字化映射字典
    # id2tag: 代表标签的数字换映射字典
    # 初始化真实实体的集合以及每隔实体的字+标签列表
    gold_entities_list, gold_entity = [], []
    # 初始化模型预测实体的几个以及每个实体的字+标签列表
    predict_entities_list, predict_entity = [], []
    # 遍历当前待评估的每个句子
    for line_no, token_idx_of_one_sentence in enumerate(idx_of_tokens):
        # 迭代句子的每个字符
        for i in range(len(token_idx_of_one_sentence)):
            # 判断: 弱句子的某一个id值等于0, 意味着是<PAD>, 后面将全部是0, 直接跳出内循环
            if token_idx_of_one_sentence[i] == 0:
                break
            # 获取当前句子中的每一个文字字符
            token_text = id2char[token_idx_of_one_sentence[i]]
            # 获取当前字符的真实实体标签类型
            gold_token_label_text = id2tag[labels[line_no][i]]
            # 获取当前字符的预测实体标签类型
            predict_token_label_text = id2tag[predict_labels[line_no][i]]
            # 首先判断真实实体是否可以加入列表中
            # 首先判断id2tag的第一个字符是否为B, 表示一个实体的开始
            if gold_token_label_text[0] == "B":
                # 将实体字符和类型加入实体列表中
                gold_entity = [token_text + "/" + gold_token_label_text]
            # 判断id2tag第一个字符是否为I, 表示一个实体的中间到结尾
            # 总体的判断条件:1.类型要以I开始 2.entity不为空 3.与上一个字符的子类型相同
            elif gold_token_label_text[0] == "I" and len(gold_entity) != 0 \
                  and gold_entity[-1].split("/")[1][1:] == gold_token_label_text[1:]:
                # 满足条件的部分追加进实体列表中
                gold_entity.append(token_text + "/" + gold_token_label_text)
            # 判断id2tag的第一个字符是否为O, 并且entity非空, 实体已经完成了全部的判断
            elif gold_token_label_text[0] == "O" and len(gold_entity) != 0:
                # 增加一个唯一标识
                gold_entity.append(str(line_no) + "_" + str(i))
                # 将一个完整的命名实体追加进最后的列表中
                gold_entities_list.append(gold_entity)
                # 将gold_eneity清空, 以便判断下一个命名实体
                gold_entity = []
            else:
                gold_entity = []

            # 接下来判断预测出来的命名实体
            # 第一步首先判断id2tag的第一个字符是否为B, 表示实体的开始
            if predict_token_label_text[0] == "B":
                predict_entity = [token_text + "/" + predict_token_label_text]
            # 判断第一个字符是否是I, 并且entity非空, 并且实体类型相同
            elif predict_token_label_text[0] == "I" and len(predict_entity) != 0 \
                 and predict_entity[-1].split("/")[1][1:] == predict_token_label_text[1:]:
                predict_entity.append(token_text + "/" + predict_token_label_text)
            # 判断第一个字符是否为O, 并且entity非空, 代表一个完整的实体已经识别完毕, 可以追加进列表中
            elif predict_token_label_text[0] == "O" and len(predict_entity) != 0:
                # 增加一个唯一标识
                predict_entity.append(str(line_no) + "_" + str(i))
                # 将识别出来的完整实体追加进最终的列表中
                predict_entities_list.append(predict_entity)
                # 将predict_entity清空, 以便判断下一个命名实体
                predict_entity = []
            else:
                predict_entity = []

    # 预测对了的实体,predict实体和gold实体的交集
    acc_entities = [entity for entity in predict_entities_list if entity in gold_entities_list]
    # 预测对的实体数
    num_acc_entities = len(acc_entities)
    # 预测出来的实体数
    num_predict_entities = len(predict_entities_list)
    # 真实的实体数
    num_gold_entities = len(gold_entities_list)

    # 如果准确实体的个数大于0, 则计算准确率,召回率, F1值。防止分母为0报错 ZeroDivisionError: division by zero
    if num_acc_entities > 0:
        # 精确率=正确预测的实体数量/被预测出来的实体数量
        step_acc = float(num_acc_entities / num_predict_entities)
        # 召回率=正确预测的实体数量/真实的实体数量
        step_recall = float(num_acc_entities / num_gold_entities)
        # f1 = (2*精确率*召回率)/(精确率+召回率)
        f1_score = 2.0 * step_acc * step_recall / (step_acc + step_recall)
        return step_acc, step_recall, f1_score, num_acc_entities, num_predict_entities, num_gold_entities
    else:
        return 0, 0, 0, num_acc_entities, num_predict_entities, num_gold_entities

# 训练函数的主要代码部分
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument("-lr",
                        "--learning_rate",
                        required=False,
                        default=0.001,
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
                        default=100,
                        type=int,
                        help="batch size")
    parser.add_argument("--num_layers",
                        required=False,
                        default=1,
                        type=int,
                        help="num hidden layers")
    parser.add_argument("--seed",
                        required=False,
                        default=1,
                        type=int,
                        help="seed")
    parser.add_argument("--patience",
                        required=False,
                        default=1000,
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
                        default=100,
                        type=int,
                        help="num hidden layers")
    parser.add_argument("--emb_size",
                        required=False,
                        default=200,
                        type=int,
                        help="num hidden layers")
    parser.add_argument("-bi",
                        "--bidirectional",
                        required=False,
                        default=True,
                        action='store_true',
                        help="bidirectional")
    args = parser.parse_args()
    # 首先确定超参数和训练数据的导入
    EMBEDDING_DIM = 200 if args.emb_size is None else args.emb_size
    HIDDEN_DIM = 100 if args.hidden_size is None else args.hidden_size
    NUM_LAYERS = 1 if args.num_layers is None else args.num_layers
    Bidirectional = args.bidirectional
    learning_rate = 0.001 if args.learning_rate is None else args.learning_rate
    EPOCH = 10 if args.epoch is None else args.epoch
    BATCH_SIZE = 100 if args.batch_size is None else args.batch_size
    PATIENTS = 500 if args.patience is None else args.patience
    handler = handlers.RotatingFileHandler(
        filename='log.ner_train.' + time_str + '.' + '_epoch_' + str(EPOCH) + '_bs_' + str(BATCH_SIZE) + '_lr_' + str(
            learning_rate) + '_emb_dim_' + str(EMBEDDING_DIM) + '_hidden_dim_' + str(HIDDEN_DIM),
        mode='w', backupCount=3, encoding='utf-8')
    format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    handler.setLevel(logging.INFO)
    handler.setFormatter(format_str)
    logger.addHandler(handler)

    # train_data_list, eval_data_list = get_data()
    train_data_list = read_data("./data/train.txt")
    eval_data_list = read_data("./data/dev.txt")
    # 字符到id的映射已经提前准备好了, 直接读取文件即可
    char_to_id = json.load(open('data/char_to_id.json', mode='r', encoding='utf-8'))
    # 直接将标签到id的映射字典作为超参数写定, 因为这个字典只和特定的任务有关系
    label_to_idx = {"O": 0, "B": 1, "I": 2, "<START>": 3, "<STOP>": 4}

    # 直接构建模型
    model = BiLSTM_CRF(vocab_size=len(char_to_id), tag_to_idx=label_to_idx, embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, bidirectional=Bidirectional)
    # 直接选定优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.85, weight_decay=1e-4)

    # 调转字符标签和id值
    id_to_tag = {v: k for k, v in label_to_idx.items()}
    # 调转字符编码中的字符和Id
    id_to_char = {v: k for k, v in char_to_id.items()}

    # 获取时间戳, 用于模型，图片，日志文件的名称
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    # model_saved_path = "./output/model/bilstm_crf_state_dict_%s.pt" % (time_str)
    # model_saved_path = "./output/model/pytorch_model.bin" % (time_str)

    # train_history_image_path = "log/bilstm_crf_train_plot_%s.png" % (time_str)
    # log_file = open("log/train_%s.log" % (time_str), mode='w', encoding='utf-8')

    # 获取整个训练的开始时间
    start = time.time()
    # 将几个重要的统计量做初始化
    train_loss_history, train_acc_history, train_recall_history, train_f1_history = [], [], [], []
    eval_loss_history, eval_acc_history, eval_recall_history, eval_f1_history = [], [], [], []
    global_step = 0
    patience = 0
    best_eval_acc = 0.0
    # 按照epochs进行轮次的训练和验证
    for epoch in trange(int(EPOCH), desc="Epoch"):
        logger.info("Epoch = " + str(epoch) + "| TimeSince: " + timeSince(start))
        total_acc_num, total_predict_num, totao_gold_num, total_loss = 0, 0, 0, 0
        # 对于任意一轮epoch训练, 我们先进行训练集的训练
        for step, train_data in enumerate(train_data_list):
            model.zero_grad()
            # 取出训练样本
            tokens, tags = train_data.get("text"), train_data.get("label")
            # 完成数字化编码
            idx_of_tokens = prepare_sequence(tokens, char_to_id)  # [seq_len],每个字符在字典中的index
            labels = torch.tensor([label_to_idx[t] for t in tags], dtype=torch.long)  # [seq_len],每个label的index
            # 计算损失值, model中计算损失值是通过neg_log_likelihood()函数完成的
            loss = model.neg_log_likelihood(idx_of_tokens, labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            # 累加每一个样本的损失值
            step_loss = loss.data.numpy()
            total_loss += step_loss
            # 评估一下模型的表现
            score, best_path_list = model(idx_of_tokens)
            # 得到降维的张量和最佳路径
            idx_of_tokens_unq = idx_of_tokens.unsqueeze(0)  # [seq_len] --> [1,seq_len]
            labels_unq = labels.unsqueeze(0)  # [seq_len] --> [1,seq_len]
            best_path_list_up = [best_path_list]  # [seq_len] --> [1,seq_len]
            step_acc, step_recall, f1_score, \
            num_acc_entities, num_predict_entities, num_gold_entities = \
                evaluate(idx_of_tokens_unq.tolist(), labels_unq.tolist(), best_path_list_up, id_to_char, id_to_tag)
            if step % 10 == 0:
                logger.info("Batch = %d , loss = %.4f , acc = %.4f , recall = %.4f , f1= %.4f  ---per batch" % (
                global_step, loss.item(), step_acc, step_recall, f1_score))

            # 累加三个重要的统计量, 预测正确的实体数, 预测出来的总实体数, 真实的实体个数
            total_acc_num += num_acc_entities
            total_predict_num += num_predict_entities
            totao_gold_num += num_gold_entities

            if step % 10 == 0:
                # 如果真实预测出来的实体数大于0, 则计算平均损失、准确率, 召回率, F1分值
                if total_predict_num > 0 and total_acc_num>0:
                    train_mean_loss = total_loss / len(train_data_list)
                    train_epoch_acc = total_acc_num / total_predict_num
                    train_epoch_recall = total_acc_num / totao_gold_num
                    train_epoch_f1 = 2.0 * train_epoch_acc * train_epoch_recall / (train_epoch_acc + train_epoch_recall)
                else:
                    train_mean_loss = total_loss / len(train_data_list)
                    train_epoch_acc = 0
                    train_epoch_recall = 0
                    train_epoch_f1 = 0
                eval_mean_loss, eval_acc, eval_recall, eval_f1 = do_evaluate(model, eval_data_list, char_to_id,
                                                                             label_to_idx, id_to_char, id_to_tag)
                log_text = "Batch = %s | train loss: %.5f |train acc: %.3f |train recall: %.3f | train f1: %.3f" \
                           " | eval loss: %.5f | eval acc: %.3f | eval recall: %.3f | eval f1: %.3f" % \
                           (global_step, train_mean_loss, train_epoch_acc, train_epoch_recall, train_epoch_f1,
                            eval_mean_loss, eval_acc, eval_recall, eval_f1)
                logger.info(log_text)

                if best_eval_acc < eval_acc:
                    best_eval_acc = eval_acc
                    logger.info('best_eval_acc= %.4f  at  batch = %.4f' % (best_eval_acc, global_step + 1))
                    save_name = f"pytorch_model_best_acc_%.4f.bin" % best_eval_acc
                    # torch.save(model.state_dict(), os.path.join(args.output_dir, f"pytorch_model_best_acc.bin"))
                    torch.save(model.state_dict(), os.path.join(args.output_dir, save_name))
                    patience = 0
                else:
                    patience += 1
                if patience > PATIENTS:
                    break
                    # 将当前epoch的重要统计量添加进画图列表中
                train_loss_history.append(train_mean_loss)
                train_acc_history.append(train_epoch_acc)
                train_recall_history.append(train_epoch_recall)
                train_f1_history.append(train_epoch_f1)
                eval_loss_history.append(eval_mean_loss)
                eval_acc_history.append(eval_acc)
                eval_recall_history.append(eval_recall)
                eval_f1_history.append(eval_f1)
        if patience > PATIENTS:
            break

    # 当整个所有的轮次结束后, 说明模型训练完毕, 直接保存模型
    model_save_dir = os.path.join(args.output_dir, f'model{global_step}')
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_dir, f"pytorch_model.bin"))
    logger.info("after " + str(global_step) + " batches , model has been saved to " + model_save_dir + " bingo!!")

    # torch.save(model.state_dict(), model_saved_path)
    text_save("data/train_loss_history.txt", train_loss_history)
    text_save("data/eval_loss_history.txt", eval_loss_history)

    text_save("data/train_acc_history.txt", train_acc_history)
    text_save("data/eval_acc_history.txt", eval_acc_history)

    text_save("data/train_recall_history.txt", train_recall_history)
    text_save("data/eval_recall_history.txt", eval_recall_history)

    text_save("data/train_f1_history.txt", train_recall_history)
    text_save("data/eval_f1_history.txt", eval_recall_history)

    # 完成画图的功能代码
    # 将loss的历史数据画成图片
    # save_train_history_image(train_loss_history, eval_loss_history, train_history_image_path, "Loss")
    #
    # # 将准确率的历史数据画成图片
    # save_train_history_image(train_acc_history, eval_acc_history, train_history_image_path, "Acc")
    #
    # # 将召回率的历史数据画成图片
    # save_train_history_image(train_recall_history, eval_recall_history, train_history_image_path, "Recall")
    #
    # # 将F1值的历史数据画成图片
    # save_train_history_image(train_f1_history, eval_f1_history, train_history_image_path, "F1")

    logger.info("Train Finished".center(100, "-"))

    # 训练结束后, 最后打印一下训练好的模型中的各个组成模块的参数详情
    for name, parameters in model.named_parameters():
        logger.info(name, ":", parameters.size())
