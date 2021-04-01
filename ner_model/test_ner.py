import json
import config
from ner_model.ner_model import *



char_to_id = json.load(open('data/char_to_id.json', mode='r', encoding='utf-8'))
label_to_id = {"O": 0, "B": 1,  "I": 2, "<START>": 3, "<STOP>": 4}
# label_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}

model=NerModel(model_path=config.ner_model,
               char_to_idx=char_to_id,
               label_to_idx=label_to_id,
               EMBEDDING_DIM=200,
               HIDDEN_DIM=100,
               NUM_LAYERS=1,
               Bidirectional=True)
while True:
    text=input("请输入\n")
    entities = model.single_predict(text)
    # entities=model.single_predict("《高等数学》的出版社是哪里？")
    # entities=model.single_predict("感冒的症状有哪些？")
    print("识别出的实体有")

    print(entities)
# print(len(entities))
# a=set(['1','2','3','3'])
# print(a[0])
