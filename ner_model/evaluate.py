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
