from ner_model.bilstm_crf import *

class NerModel:
    def __init__(self,model_path,char_to_idx,label_to_idx,EMBEDDING_DIM,
                  HIDDEN_DIM, NUM_LAYERS,Bidirectional):
        # 直接构建模型
        self.char_to_idx=char_to_idx
        self.label_to_idx=label_to_idx
        self.id_to_tag = {v: k for k, v in self.label_to_idx.items()}
        self.model = BiLSTM_CRF(vocab_size=len(self.char_to_idx), tag_to_idx=self.label_to_idx, embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, bidirectional=Bidirectional)
        self.model.load_state_dict(torch.load(model_path))
        self.mask_char=['。','、',"~",'"','“','’','\'',':','：','，',',','`','`','<','>','《','》','《','》','?','？',';','[',']','{','}','【','】','=','_']
    def single_predict(self,content):
        # 函数功能: 单样本语句的命名实体识别预测, 返回识别出来的实体列表
        # content: 待遇测的单条样本
        entities=[]
        # 预测阶段不求解梯度，不更新参数
        with torch.no_grad():
            mask_characters=[]
            # 将待预测的样本进行数字化映射
            token_ids = prepare_sequence(content, self.char_to_idx)
            # 直接调用模型获取维特比算法的解码结果
            score, best_path_list = self.model(token_ids)
            # 初始化一个实体
            entity = None
            # 遍历解码后的列表, 一个一个提取实体
            for char_idx, tag_idx in enumerate(best_path_list):
                # 如果预测结果的tag_id属于id_to_tag字典的key值
                if tag_idx in self.id_to_tag:
                    # 取字典的第一个字符
                    tag_text = self.id_to_tag[tag_idx][0]
                    # 通过下标将真实中文字符取出
                    current_char = content[char_idx]
                    # 如果当前标签是B, 意味着实体的开始
                    if tag_text == "B":
                        entity = current_char
                    # 如果当前标签是I, 意味着实体的中间到结尾, 直接追加字符串即可
                    elif tag_text == "I":
                        entity += current_char
                    # 当实体不为空并且当前标签是O时, 意味着已经成功的提取了一个实体
                    elif tag_text == "O" and entity:
                        # 为了防止特殊字符的干扰,对entity进行过滤
                        for mask_char in self.mask_char:
                            entity.replace(mask_char,"")
                        entities.append(entity)
                        # 重置为空, 准备下一个实体的识别
                        entity = None
        # 返回单条样本中的所有命名实体, 以集合的形式返回
        return set(entities)

    # 编写批量预测代码函数
    def batch_predict(self,contents):
        batch_entities=[]
        for content in contents:
            entities=self.single_predict(content)
            batch_entities.append(entities)
        return batch_entities

