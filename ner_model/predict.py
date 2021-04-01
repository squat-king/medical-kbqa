# 导入工具包
import os
import torch
import json
from tqdm import tqdm
from ner_model.bilstm_crf import BiLSTM_CRF

# 完成中文文本数字化映射的函数
def prepare_sequence(seq, char_to_id):
    char_ids = []
    # 遍历中文文本进行映射
    for idx, ch in enumerate(seq):
        # 判断当前字符是否在映射字典中, 如果不在取UNK作为替代
        if char_to_id.get(ch):
            char_ids.append(char_to_id[ch])
        else:
            char_ids.append(char_to_id["<UNK>"])
    # 返回封装成Tensor类型的列表
    return torch.tensor(char_ids, dtype=torch.long)

# 单样本预测的函数构建
def single_predict(model_path, content, char_to_id_json_path, embedding_dim,
                   hidden_dim, target_type_list, tag_to_id):
    # 函数功能: 单样本语句的命名实体识别预测, 返回识别出来的实体列表
    # model_path: 模型的存储路径
    # content: 待遇测的单条样本
    # char_to_id_json_path: 字符映射字典的存储路径
    # embedding_dim: 词嵌入的维度
    # hidden_dim: 隐藏层的维度
    # target_type_list: 待识别的类型列表
    # tag_to_id: 标签映射字典
    # 加载数字化映射字典
    char_to_id = json.load(open(char_to_id_json_path, mode='r', encoding='utf-8'))
    # 实例化类对象
    model = BiLSTM_CRF(len(char_to_id), tag_to_id, embedding_dim, hidden_dim)
    # 加载已经训练好的模型
    model.load_state_dict(torch.load(model_path))
    # 获取tag到id的映射字典
    tag_id_dict = {v:k for k, v in tag_to_id.items() if k[2:] in target_type_list}
    # 初始化返回的结果列表
    entities = []
    # 预测阶段不求解梯度，不更新参数
    with torch.no_grad():
        # 将待预测的样本进行数字化映射
        sentence_in = prepare_sequence(content, char_to_id)
        # 直接调用模型获取维特比算法的解码结果
        score, best_path_list = model(sentence_in)
        # 初始化一个实体
        entity = None
        # 遍历解码后的列表, 一个一个提取实体
        for char_idx, tag_id in enumerate(best_path_list):
            # 如果预测结果的tag_id属于目标字典的key值
            if tag_id in tag_id_dict:
                # 取字典的第一个字符
                tag_index = tag_id_dict[tag_id][0]
                # 通过下标将真实中文字符取出
                current_char = content[char_idx]
                # 如果当前标签是B, 意味着实体的开始
                if tag_index == "B":
                    entity = current_char
                # 如果当前标签是I, 意味着实体的中间到结尾, 直接追加字符串即可
                elif tag_index == "I":
                    entity += current_char
            # 当实体不为空并且当前标签是O时, 意味着已经成功的提取了一个实体
            if tag_id == tag_to_id["O"] and entity:
                # 为了防止特殊字符的干扰, 做一些判断
                if "、" not in entity and "～" not in entity and "。" not in entity \
                    and "”" not in entity and "：" not in entity and ":" not in entity \
                    and "，" not in entity and "," not in entity and "." not in entity \
                    and ";" not in entity and "；" not in entity and "【" not in entity \
                    and "】" not in entity and "[" not in entity and "]" not in entity:
                    entities.append(entity)
                # 重置为空, 准备下一个实体的识别
                entity = None
    # 返回单条样本中的所有命名实体, 以集合的形式返回
    return set(entities)

# 编写批量预测代码函数
def batch_predict(data_path, model_path, char_to_id_json_path, embedding_dim,
                  hidden_dim, target_type_list, prediction_result_path, tag_to_id):
    # data_path: 数据文件的路径(文件夹)
    # model_path: 模型的存储路径
    # char_to_id_json_path: 字符映射字典的文件路径
    # embedding_dim: 词嵌入的维度
    # hidden_dim: 隐藏层的维度
    # target_type_list: 待匹配的标签类型
    # prediction_result_path: 预测结果的存储路径
    # tag_to_id: 标签的映射字典
    # 遍历所有的文件进行预测
    for fn in tqdm(os.listdir(data_path)):
        # 拼接完整的文件路径
        fullpath = os.path.join(data_path, fn)
        # 定义结果文件
        entities_file = open(os.path.join(prediction_result_path, fn.replace('txt', 'csv')),
                             mode='w', encoding='utf-8')
        # 读取文件进行预测
        with open(fullpath, mode='r', encoding='utf-8') as f:
            content = f.readline()
            # 调用单条样本预测函数即可
            entities = single_predict(model_path, content, char_to_id_json_path,
                                      embedding_dim, hidden_dim, target_type_list, tag_to_id)
            # 写入结果文件中
            entities_file.write("\n".join(entities))
    print("Batch Prediction Finished!".center(100, "-"))






























