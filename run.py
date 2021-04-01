import json
import re
import config
import torch
import py2neo
from unit_model.unit_model import UnitModel
from ner_model.ner_model import NerModel
from similarity_model.sim_model import SimModel

def get_entities_relation_dict(graph,entities):
    rel_subj_dict = {}
    triples=[]
    for entity in entities:
        # 对实体名进行完全匹配，如果匹配不成功，就进行模糊匹配
        # 返回的是一个Cursor类型
        res_cursor = graph.run('MATCH (e1)-[r]->(e2) WHERE e1.name ="%s" return r' % entity)
        res_list=res_cursor.data()
        if len(res_list)!= 0:
            pass
        else:
            res_cursor = graph.run('MATCH (e1)-[r]->(e2) WHERE e1.name =~ ".*%s.*" return r' % entity)
            res_list = res_cursor.data()
        # res_list是一个dict类型
        for r in res_list:
            s = str(r['r'])
            str_obj = s[:s.index('[')]
            str_rel_subj = s[s.index('['):]
            obj = re.findall(r"\((.+?)\)", str_obj)
            rel = re.findall(r":(.+?) ", str_rel_subj)
            subj = re.findall(r"\((.+?)\)", str_rel_subj)
            triples.append((obj[0],rel[0],subj[0]))
        for triple in triples:
            # 如果dict中已经有了rel名作为key值,就append，反之就赋值一个list
            if triple[1] in rel_subj_dict.keys():
                rel_subj_dict[str(triple[1])].append(triple)
            else:
                rel_subj_dict[str(triple[1])]=[triple]

    return rel_subj_dict  # key=关系名,value=[(主体名,关系名,客体名1),(主体名,关系名,客体名2),......]


def list_to_str(subj_list):
    s_str=""
    for _, t in enumerate(set(subj_list)):
        if _ == 0:
            s_str += t
        else:
            s_str += ("、" + t)
    return s_str


def main():
    with torch.no_grad():
        char_to_id = json.load(open(config.char_to_id, mode='r', encoding='utf-8'))
        label_to_id = {"O": 0, "B": 1,  "I": 2, "<START>": 3, "<STOP>": 4}
        # label_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4, "<START>": 5, "<STOP>": 6}

        ner_model = NerModel(model_path=config.ner_model,
                             char_to_idx=char_to_id,
                             label_to_idx=label_to_id,
                             EMBEDDING_DIM=200,
                             HIDDEN_DIM=100,
                             NUM_LAYERS=1,
                             Bidirectional=True)

        sim_model=SimModel(config.sim_model)
        unit_model=UnitModel()

        graph = py2neo.Graph("http://localhost:7474", username="neo4j", password="root")

        while True:
            print("====="*10)
            question = input("问题：\n")
            question = question.strip()
            if ( "quit" == question ):
                print("quit")
                return
            try:

                entities=ner_model.single_predict(question)

                print("识别出的实体有:", entities)
                # 如果没有识别出实体，就调用Unit
                if len(entities)==0:
                    # print("未发现实体")
                    print(unit_model.unit_chat(question))
                    continue
                else: # 如果识别出了实体，就在数据库中查找 该实体的所有属性和关系

                    # 查询返回一个字典,key=关系名,value=[(主体名,关系名,客体名1),(主体名,关系名,客体名2),......]
                    att_rels_dict=get_entities_relation_dict(graph,entities)

                    #如果实体没有任何属性和关系，就调用unit
                    if len(att_rels_dict)==0:
                        # print("未发现属性和关系")
                        print(unit_model.unit_chat(question))
                        continue
                    else:# 如果识别出了属性和关系，就判断属性/关系与question的相似度，取最相似的属性/关系值返回
                        # print("匹配到的三元组有",att_rels_dict)

                        att_rels_names=list(att_rels_dict.keys())

                        print("与%s有关的信息有"%entities,att_rels_names)

                        (best_idx,sim_logit)=sim_model.predict(question,att_rels_names)

                        # print("最相似的属性名/关系名是 :",att_rels_names[best_idx],
                        #       " , 相似度为:",sim_logit)
                        # reply=att_rels_dict[att_rels_names[best_idx]]

                        triples_list=att_rels_dict[att_rels_names[best_idx]]

                        obj=triples_list[0][0]
                        rel=att_rels_names[best_idx]
                        subj_list=[]

                        for triple in triples_list:
                            subj_list.append(triple[2])
                        subjs=list_to_str(subj_list)

                        reply = "回答：\n{}的{}是{}".format(obj,rel, subjs)
                        print(reply)
            except:
                print(unit_model.unit_chat(question))

if __name__ == '__main__':
    main()