# 导入若干工具包
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

# 设置随机种子
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(1)


# 编写3个辅助函数, 未来在类中会调用这几个函数
# 第1个: 求最大值所在的下标
def argmax(vec):
    # 返回最大值对应的下标, 以Python整形返回
    _, idx = torch.max(vec, 1)
    return idx.item()


# 第2个: 文本字符串转换成数字化张量, 以Tensor长整形返回
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# 第3个: 计算log-sum-exp的值
def log_sum_exp(vec):
    # 引入代码技巧: 为了防止指数运算溢出, 先减去最大值, 外层再加上最大值
    # 先把最大值取出
    max_score = vec[0, argmax(vec)]
    # 使用广播变量的形式进行扩充
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # 直接返回要求的值
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# 首先设置若干超参数
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 200
HIDDEN_DIM = 100
VOCAB_SIZE = 30

# 初始化标签字典
tag_to_idx = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, START_TAG: 5, STOP_TAG: 6}


# 编写函数, 完成中文文本信息的数字化编码, 变成张量
def sentence_map(sentence_list):
    # sentence_list: 代表待转换的中文字符列表
    # 初始化映射字典
    word_to_idx = {}
    for sentence, tags in sentence_list:
        for word in sentence:
            # 只要任意的一个中文字符不在字典中, 追加进字典, 并且将值依次递增
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    return word_to_idx


sentence_list = [
    (["女", "性", "，", "8", "8", "岁", "，", "农", "民", "，", "双", "滦", "区", "应", "营", "子", "村", "人", "，", "主", "因", "右", "髋",
      "部", "摔", "伤", "后", "疼", "痛", "肿", "胀", "，", "活", "动", "受", "限", "5", "小", "时", "于", "2", "0", "1", "6", "-", "1",
      "0", "-", "2", "9", "；", "1", "1", "：", "1", "2", "入", "院", "。"],
     ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O",
      "O", "O", "O", "O", "B-sym", "I-sym", "B-sym", "I-sym", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O",
      "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
     ),
    (["处", "外", "伤", "后", "疼", "痛", "伴", "有", "头", "晕", "2", "小", "时", "于", "2", "0", "1", "6", "-", "-", "1", "0", "-",
      "-", "0", "2", "1", "2", ":", "0", "2", "收", "入", "院", "。"],
     ["O", "O", "O", "O", "B-sym", "I-sym", "O", "O", "B-sym", "I-sym", "O", "O", "O", "O", "O", "O", "O", "O", "O",
      "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
     )]


# if __name__ == '__main__':
#     word_to_idx = sentence_map(sentence_list)
#     print(word_to_idx)

# 创建类
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim, num_layers=1, bidirectional=True):
        super(BiLSTM_CRF, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        # 将参数传入类中
        self.embedding_dim = embedding_dim
        # 将LSTM网络中的隐藏层维度传入
        self.hidden_dim = hidden_dim
        # 将词汇总数传入
        self.vocab_size = vocab_size
        # 将标签字典传入
        self.tag_to_idx = tag_to_idx
        # 将标签数量传入类中, label_num决定了转移矩阵的维度
        self.label_num = len(tag_to_idx)

        # 定义词嵌入层, 传入的两个参数是vocab_size, embedding_dim
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 定义BiLSTM层, 只设定一个隐藏层, 注意一点: 因为采用了双向网络所以隐藏层的维度要除以2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=self.num_layers, bidirectional=bidirectional)
        # 定义全连接层, 作用是将BiLSTM网络的输出维度从hidden_dim变换到转移矩阵的维度label_num
        self.hidden2tag = nn.Linear(hidden_dim, self.label_num)

        # 定义转移矩阵并初始化参数, 采用正态分布的随机初始化值
        # 转移矩阵是一个方阵, 维度是[label_num,label_num]
        # 转移矩阵的任意位置(i, j), 代表从j转移到i的概率
        self.transitions = nn.Parameter(torch.randn(self.label_num, self.label_num))

        # 设定两个限定条件, 任意合法的转移不会转移到起始标签START, 并且不会从结束标签STOP转移出去
        # 即不可能由其他tag转移到start,也不可能由end转移到其他tag
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

        # 初始化LSTM网络的两个初始输入参数h0, c0
        self.hidden = self.init_hidden()# 是一个二元组,(h0,c0) h0 = [num_layer * num_direction , batch_size , hidden_size/2]
                                        #                    c0 = [num_layer * num_direction , batch_size , hidden_size/2]

    def init_hidden(self):
        # 明确一下LSTM网络的初始化输入张量的维度意义
        # 三个参数代表 (num_layers * num_directions, batch_size, hidden_size)
        return (torch.randn(self.num_layers*self.num_directions, 1, self.hidden_dim // 2), torch.randn(self.num_layers*self.num_directions, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):# 输入句子,返回发射分数,[seq_len , (batch_size ,) num_labels]
        # sentence: 代表文本向量化后的输入张量
        # 首先获取一个初始化的隐藏层张量h0, c0
        self.hidden = self.init_hidden() # (h0、c0 = [num_layers * num_directions , batch_size, hidden_size])
        # 进行词嵌入层的处理, 并将一句样本变形成三维张量
        embedding = self.word_embeds(sentence).view(len(sentence), 1, -1) # [seq_length , batch_size , input_size]
        # 直接将输入张量和h0,c0输入进LSTM网络中
        seq_lengths=[embedding.size(0)]
        lstm_input = pack_padded_sequence(embedding, seq_lengths)
        lstm_out, self.hidden = self.lstm(lstm_input, self.hidden) # in: embeds+(h0,c0)  out : output-[seq_len,batch_size,hidden_size*2]、 hidden=(ht,ct),ht=[num_directions*num_layers,batch_size,hidden_size]
        # 将输出张量重新变形成二维张量, 以便后续送入全连接层处理
        lstm_out = lstm_out.data.view(len(sentence), self.hidden_dim)
        # 进行全连接层的映射处理, 作用是将LSTM的输出维度从hidden_dim变换成转移矩阵的维度label_num
        logits = self.hidden2tag(lstm_out)
        # 返回发射分数
        return logits # [seq_len , (batch_size ,) num_labels]

    def _forward_alg(self, logits):# 根据发射分数,计算所有可能路径的分数总和
        # logits: 代表经过BiLSTM网络处理后的发射矩阵张量
        # 初始化一个(1, label_num)的二维张量, 并且全部赋值给-10000
        init_alphas = torch.full((1, self.label_num), -10000.)
        # 只把起始标签的位置赋值成0, 意思是只能从起始标签开始转移
        init_alphas[0][self.tag_to_idx[START_TAG]] = 0.

        # 初始化前向传播张量, 也要求最初合法的转移只能从起始标签开始
        forward_var = init_alphas

        # 遍历样本句子中的每一个字符
        for feat in logits:
            # 初始化空列表, 存储当前时间步的前向传播张量
            alphas_t = []
            # 遍历所有可能的标签类型
            for next_tag in range(self.label_num):
                # 初始化发射分数的张量, 无论前一个标签是什么, 经过广播变量后的每一个分量值都相同
                emit_score = feat[next_tag].view(1, -1).expand(1, self.label_num)
                # 第i个输入的转移分数代表第i个字符转移到next_tag标签的转移分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # 将前向传播张量 + 转移矩阵张量 + 发射矩阵张量, 表示第i个字符转移到next_tag标签的总分数
                next_tag_var = forward_var + trans_score + emit_score
                # 经过log_sum_exp的处理后得到的分数添加进每一个时间步的列表中
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 将前向传播张量变形成1行后, 赋值给forward_var, 等待下一个时间步的处理
            forward_var = torch.cat(alphas_t).view(1, -1)
        # 所有的时间步处理结束后, 代表整个句子完成了遍历, 需要在最后添加结束标签的转移分数
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        # 同理, 对最后的张量进行log_sum_exp的计算
        alpha = log_sum_exp(terminal_var)
        # 最终返回整个句子的前向传播分数
        return alpha

    def _score_sentence(self, logits, tags):# 输入发射分数、真实标签序列，计算得分
        # logits: 代表经过BiLSTM网络处理后的发射矩阵张量
        # tags: 代表真实的标签序列而且经过数字化的处理
        # 初始化分数为0
        score = torch.zeros(1)
        # 模拟NLP的处理流程, 将起始标签START_TAG添加到真实标签序列的最前面
        tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long), tags])# [1]+[seq_len]=[seq_len+1]

        # 遍历发射矩阵的每一个时间步, 累加分数
        for i, feat in enumerate(logits):
            # 每一个时间步进行分数的累加, 加的是从第i个时间步的标签转移到第i+1个时间步标签的分数,
            # 还要加上第i+1个时间步的发射分数
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]

        # for循环结束后, 还是要将整个字符遍历后的结果分数加上转移到结束标签的分数
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
        # 最后返回带标签的真实语句的分数值
        return score

    def _viterbi_decode(self, logits):
        # logits: 代表经过BiLSTM网络处理后的发射矩阵张量
        # 初始化回溯指针列表, 里面存放的是每一个时间步的解析标签结果
        backpointers = []

        # 初始化变量为1行7列的张量, 全部赋值为-10000
        init_vvars = torch.full((1, self.label_num), -10000.)
        # 只将起始标签的位置设置为0, 意思是所有合法的转移都要从START_TAG开始
        init_vvars[0][self.tag_to_idx[START_TAG]] = 0.

        # 初始化前向传播张量, 第i个时间步的forward_var中存放的是第i-1个时间步的维特比变量
        forward_var = init_vvars

        # 遍历样本语句中的每一个时间步, 也就是遍历序列中的每一个字符
        for feat in logits:
            # 初始化当前时间步的回溯指针
            bptrs_t = []
            # 初始化当前时间步的维特比变量
            viterbivars_t = []

            # 遍历当前时间步下, 所有可能的标签, 本项目中只有7种可能的标签
            for next_tag in range(self.label_num):
                # 将上一个时间步标签i对应的维特比变量, 加上从标签i转移到标签next_tag的转移分数, 作为next_tag_var
                # 注意: 这里没有加发射矩阵的分数, 试音发射矩阵经过广播变量处理后所有值都相同, 不影响最大值的求解
                next_tag_var = forward_var + self.transitions[next_tag]
                # 采用贪心算法, 求解最大值最为解析结果
                best_tag_id = argmax(next_tag_var)
                # 将最优解析id值追加进当前时间步的回溯指针列表中
                bptrs_t.append(best_tag_id)
                # 将维特比变量值追加金当前时间步的列表中
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 内层for循环结束后, 需要再加上发射矩阵张量, 作为更新后的前向传播张量, 准备下一个时间步的解析
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # 将当前时间步的解析指针列表追加进总的回溯指针列表中
            backpointers.append(bptrs_t)

        # 外层for循环结束后, 说明已经遍历了整个序列, 在最后需要追加上转移到结束标签的转移分数
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        # 求解最大值标签, 作为最后一个时间步的解析结果
        best_tag_id = argmax(terminal_var)
        # 采用最佳tag对应的分数作为整个路径解析的分数值
        path_score = terminal_var[0][best_tag_id]

        # 遍历总的回溯指针列表, 依次解析出最佳路径标签
        # 初始化最终的结果列表, 并且最后一步得到的标签首先加入结果中
        best_path = [best_tag_id]
        # 逆向遍历总的回溯列表
        for bptrs_t in reversed(backpointers):
            # 每次通过最佳id解析出来的最佳标签追加进结果列表中
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # 解析的最后一步对应的一定是开始标签START_TAG, 在这里删除掉
        # 因为START_TAG和STOP_TAG并不是真实的标签, 只是人为添加的辅助程序运行的标签
        start = best_path.pop()
        # 为了代码的鲁棒性, 增加正确性检测, 确认一下这个标签是起始标签
        assert start == self.tag_to_idx[START_TAG]
        # 因为是从后向前追加进的结果列表, 所以需要逆序排列就是从前向后的解析结果
        best_path.reverse()
        # 返回解析的语句分数, 和解析出的标签最佳路径
        return path_score, best_path

    def neg_log_likelihood(self, sentence, labels):# crf的损失函数=所有预测路径的总分数-真实路径的分数,LOSS越小越好
        # sentence: 代表输入语句经过数字化编码后的张量
        # labels: 真实标签样本经历数字化编码后的张量
        # 从BiLSTM网络中得到发射矩阵张量
        logits = self._get_lstm_features(sentence)#[seq_len, (batch_size ,) num_labels]
        # 第一步调用前向传播函数得到前向传播的预测值
        forward_score = self._forward_alg(logits)
        # 第二步调用真实句子的计算函数得到真实句子的分数值
        gold_score = self._score_sentence(logits, labels)
        # 用预测分数 - 真实分数, 这个差值就是损失值loss
        return forward_score - gold_score

    def forward(self, sentence):
        # 首先从BiLSTM网络中得到发射矩阵的张量
        logits = self._get_lstm_features(sentence)
        # 利用发射矩阵张量, 通过维特比算法直接解码得到最佳标签路径
        score, tag_seq = self._viterbi_decode(logits)
        # 返回句子的分数值, 以及最佳解码标签序列
        return score, tag_seq


# 直接调用类
# if __name__ == '__main__':
#     # 获取所有字符的映射字典
#     word_to_idx = sentence_map(sentence_list)
#     # 创建类对象
#     # model = BiLSTM_CRF(vocab_size=len(word_to_idx), tag_to_idx=tag_to_idx,
#     #                    embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
#     model = BiLSTM_CRF(vocab_size=22303, tag_to_idx=tag_to_idx,
#                        embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
#     # print(model)
#     model.load_state_dict(torch.load(r"F:\毕业设计\QASystem\offline\ner_model\model\bilstm_crf_state_dict_20201025_153550.pt"))
#     # 遍历样本数据
#     for sentence, tags in sentence_list:
#         # 完成自然语言文本到数字化张量的映射
#         sentence_in = prepare_sequence(sentence, word_to_idx)
#         # 调用类内的函数完成发射矩阵的张量计算
#         logits = model._get_lstm_features(sentence_in)
#         # print(logits)
#         # print(logits.shape)
#         # print('*******')
#         forward_score = model._forward_alg(logits)
#         # print(forward_score)
#         # 首先将tags标签进行数字化映射, 然后封装成torch.long类型的Tensor张量
#         targets = torch.tensor([tag_to_idx[t] for t in tags], dtype=torch.long)
#         # 将发射矩阵张量, 和真实标签作为参数传入函数中, 计算真实句子的分数
#         # gold_score = model._score_sentence(logits, targets)
#         # print(gold_score)
#         # 直接将发射矩阵张量送入维特比解析函数中
#         # score, tag_seq = model._viterbi_decode(logits)
#         # print(tag_seq)
#         # print(score)
#         # print('******')
#
#         # 直接调用损失函数计算loss值
#         loss = model.neg_log_likelihood(sentence_in, targets)
#         print('loss=', loss)
#
#         # 进入预测阶段, 检验一下forward函数的解码能力
#         with torch.no_grad():
#             # 模型默认调用forward()函数, 功能是预测阶段的维特比解码函数
#             score, tag_seq = model(sentence_in)
#             print(score)
#             print(tag_seq)
#             print('******')
