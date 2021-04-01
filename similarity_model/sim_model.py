import torch
from transformers import BertTokenizer, BertForSequenceClassification
from similarity_model.utils import *

class SimModel:
    def __init__(self,MODEL_PATH):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH)


    def predict(self, question, attributes):
        predict_example=[]
        for attribute in attributes:
            predict_example.append(InputExample(text_a=question,text_b=attribute,label=0))
        with torch.no_grad():
            features=convert_examples_to_features(predict_example,self.tokenizer,max_seq_length=128)
            input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
            input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
            segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
            input_ids_flat = input_ids.view(-1, input_ids.size(-1))
            input_mask_flat = input_mask.view(-1, input_mask.size(-1))
            segment_ids_flat = segment_ids.view(-1, segment_ids.size(-1))
            outputs = self.model(input_ids_flat, token_type_ids=segment_ids_flat, attention_mask=input_mask_flat)
            logits=outputs[0]
            # print(logits)
            sim_int=torch.argmax(logits,-1)
            # print("sim_int is ",sim_int)
            # print(len(sim_int))
            sim=[]
            for idx,value in enumerate(sim_int):
                if value==1:# 如果某attribute被判定为与问题相似度分类为1,就取它的(idx,logits值)
                    sim.append((idx,logits[idx][1]))
            sim_float=sorted(sim,key=lambda x: x[1],reverse=True)
            # print(len(sim_float))
            return sim_float[0]# 返回一个元组,对应最相似的那个属性,(idx,value)

