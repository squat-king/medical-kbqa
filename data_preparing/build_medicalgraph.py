#!/usr/bin/env python3
# coding: utf-8
# File: MedicalGraph.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-10-3
# Reviser: jhz〈jhzsdufe@163.com〉
# Modified on: 2021-3-30

import os
import json
from py2neo import Graph, Node
import config


class MedicalGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = config.medical_data
        # self.g = Graph(address="localhost:7474",auth=('neo4j', 'root'))
        self.g = Graph("http://localhost:7474", username="neo4j", password="root")

    '''读取文件'''

    def read_nodes(self):
        # 共8类节点
        self.drugs = []  # 药品
        self.recipes = []  # 菜谱
        self.foods = []  # 食物
        self.checks = []  # 检查
        self.departments = []  # 科室
        self.producers = []  # 药企
        self.diseases = []  # 疾病
        self.symptoms = []  # 症状

        self.disease_infos = []  # 疾病的相关属性,只有disease实体有属性

        # 构建节点实体关系
        self.rels_between_departments = []  # 科室－科室关系
        self.rels_not_eat = []  # 疾病－忌吃食物关系
        self.rels_do_eat = []  # 疾病－宜吃食物关系
        self.rels_recommand_food = []  # 疾病－推荐吃食物关系
        self.rels_common_drug = []  # 疾病－通用药品关系
        self.rels_recommand_drug = []  # 疾病－推荐药品关系
        self.rels_check = []  # 疾病－检查关系
        self.rels_drug_producer = []  # 厂商－药物关系

        self.rels_disease_symptom = []  # 疾病症状关系
        self.rels_disease_acompany = []  # 疾病并发关系
        self.rels_disease_department = []  # 疾病与科室之间的关系

        print("从json中抽取三元组")
        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                count += 1
                data_json = json.loads(line)
                disease_dict = {}
                disease = data_json['name']
                disease_dict['name'] = disease
                self.diseases.append(disease)
                # disease_dict['desc'] = ''
                # disease_dict['prevent'] = ''
                # disease_dict['cause'] = ''
                # disease_dict['easy_get'] = ''
                # disease_dict['cure_department'] = ''
                # disease_dict['cure_way'] = ''
                # disease_dict['cure_lasttime'] = ''
                # disease_dict['symptom'] = ''
                # disease_dict['cured_prob'] = ''

                if 'symptom' in data_json:
                    self.symptoms += data_json['symptom']
                    for symptom in data_json['symptom']:
                        self.rels_disease_symptom.append([disease, "症状", symptom])

                if 'acompany' in data_json:
                    for acompany in data_json['acompany']:
                        self.rels_disease_acompany.append([disease, "并发症", acompany])

                # if 'desc' in data_json:
                #     disease_dict['desc'] = data_json['desc']

                # if 'prevent' in data_json:
                #     disease_dict['prevent'] = data_json['prevent']

                # if 'cause' in data_json:
                #     disease_dict['cause'] = data_json['cause']

                # if 'get_prob' in data_json:
                #     disease_dict['get_prob'] = data_json['get_prob']

                # if 'easy_get' in data_json:
                #     disease_dict['easy_get'] = data_json['easy_get']

                if 'cure_department' in data_json:
                    cure_department = data_json['cure_department']
                    if len(cure_department) == 1:
                        self.rels_disease_department.append([disease, "治疗科室", cure_department[0]])
                    if len(cure_department) == 2:
                        big = cure_department[0]
                        small = cure_department[1]
                        self.rels_between_departments.append([small, "所属父科室", big])
                        self.rels_disease_department.append([disease, "治疗科室", small])

                    # disease_dict['cure_department'] = cure_department
                    self.departments += cure_department

                # if 'cure_way' in data_json:
                #     disease_dict['cure_way'] = data_json['cure_way']

                # if 'cure_lasttime' in data_json:
                #     disease_dict['cure_lasttime'] = data_json['cure_lasttime']

                # if 'cured_prob' in data_json:
                #     disease_dict['cured_prob'] = data_json['cured_prob']

                if 'common_drug' in data_json:
                    common_drug = data_json['common_drug']
                    for drug in common_drug:
                        self.rels_common_drug.append([disease, "常用药品", drug])
                    self.drugs += common_drug

                if 'recommand_drug' in data_json:
                    recommand_drug = data_json['recommand_drug']
                    self.drugs += recommand_drug
                    for drug in recommand_drug:
                        self.rels_recommand_drug.append([disease, "推荐药品", drug])

                if 'not_eat' in data_json:
                    not_eat = data_json['not_eat']
                    for _not in not_eat:
                        self.rels_not_eat.append([disease, "忌吃食物", _not])

                    self.foods += not_eat
                if 'do_eat' in data_json:
                    do_eat = data_json['do_eat']
                    for _do in do_eat:
                        self.rels_do_eat.append([disease, "宜吃食物", _do])
                    self.foods += do_eat

                if 'recommand_eat' in data_json:
                    recommand_eat = data_json['recommand_eat']
                    for _recommand in recommand_eat:
                        self.rels_recommand_food.append([disease, "推荐菜品", _recommand])
                    self.recipes += recommand_eat

                if 'check' in data_json:
                    check = data_json['check']
                    for _check in check:
                        self.rels_check.append([disease, "所需检查", _check])
                    self.checks += check

                if 'drug_detail' in data_json:
                    drug_detail = data_json['drug_detail']
                    for i in drug_detail:
                        if (str(i).endswith(")")):  # 如果以括号结尾，则认为以药品名结尾
                            tuples = str(i).rpartition("(")
                            drug_name = tuples[2].rstrip(")")
                            if (tuples[0].find(drug_name) > 0):
                                producer_name = tuples[0].rstrip(drug_name)
                        else:
                            drug_name = i
                            producer_name = i
                        self.drugs.append(drug_name)
                        self.producers.append(producer_name)
                        self.rels_drug_producer.append([producer_name, "生产药品", drug_name])
                self.disease_infos.append(disease_dict)

    '''建立节点'''

    def create_node(self, label, nodes):
        count = 0
        for node_name in nodes:
            self.g.run('MERGE (e:%s{name:"%s"})' % (label, node_name))
            count += 1
            print(label, count, len(nodes))
        return

    '''创建知识图谱中心疾病的节点'''

    def create_diseases_nodes(self, disease_infos):
        count = 0
        for disease_dict in disease_infos:
            # node = Node("Disease", name=disease_dict['name'], desc=disease_dict['desc'],
            #             prevent=disease_dict['prevent'], cause=disease_dict['cause'],
            #             easy_get=disease_dict['easy_get'], cure_lasttime=disease_dict['cure_lasttime'],
            #             cure_department=disease_dict['cure_department']
            #             , cure_way=disease_dict['cure_way'], cured_prob=disease_dict['cured_prob'])
            node = Node("Disease", name=disease_dict['name'])
            self.g.create(node)
            count += 1
            print("Disease", count, len(disease_infos))
        return

    '''创建知识图谱实体节点类型schema'''

    def create_graphnodes(self):
        self.create_diseases_nodes(self.disease_infos)
        self.create_node('Drug', self.drugs)
        print(len(self.drugs))
        self.create_node('Recipe', self.recipes)
        print(len(self.recipes))
        self.create_node('Food', self.foods)
        print(len(self.foods))
        self.create_node('Check', self.checks)
        print(len(self.checks))
        self.create_node('Department', self.departments)
        print(len(self.departments))
        self.create_node('Producer', self.producers)
        print(len(self.producers))
        self.create_node('Symptom', self.symptoms)
        return

    '''创建实体关系边'''

    # def create_graphrels(self):
    #     self.create_relationship('Disease', 'Recipe', self.rels_recommand_food, 'recommand_eat', '推荐食谱')
    #     self.create_relationship('Disease', 'Food', self.rels_not_eat, 'no_eat', '忌吃')
    #     self.create_relationship('Disease', 'Food', self.rels_do_eat, 'do_eat', '宜吃')
    #     self.create_relationship('Department', 'Department', self.rels_between_departments, 'belongs_to', '属于')
    #     self.create_relationship('Disease', 'Drug', self.rels_common_drug, 'has_common_drug', '常用药品')
    #     self.create_relationship('Producer', 'Drug', self.rels_drug_producer, 'production', '生产药品')
    #     self.create_relationship('Disease', 'Drug', self.rels_recommand_drug, 'has_recommand_drug', '推荐药品')
    #     self.create_relationship('Disease', 'Check', self.rels_check, 'need_to_check', '所需检查')
    #     self.create_relationship('Disease', 'Symptom', self.rels_disease_symptom, 'has_symptom', '症状')
    #     self.create_relationship('Disease', 'Disease', self.rels_disease_acompany, 'accompany_with', '并发疾病')
    #     self.create_relationship('Disease', 'Department', self.rels_disease_department, 'belongs_to', '所属科室')

    def create_graphrels(self):
        self.create_relationship('Disease', 'Recipe', self.rels_recommand_food, '推荐菜品')
        self.create_relationship('Disease', 'Food', self.rels_not_eat, '忌吃食物')
        self.create_relationship('Disease', 'Food', self.rels_do_eat,  '宜吃食物')
        self.create_relationship('Department', 'Department', self.rels_between_departments, '所属父科室')
        self.create_relationship('Disease', 'Drug', self.rels_common_drug, '常用药品')
        self.create_relationship('Producer', 'Drug', self.rels_drug_producer, '生产药品')
        self.create_relationship('Disease', 'Drug', self.rels_recommand_drug,'推荐药品')
        self.create_relationship('Disease', 'Check', self.rels_check,  '所需检查')
        self.create_relationship('Disease', 'Symptom', self.rels_disease_symptom,  '症状')
        self.create_relationship('Disease', 'Disease', self.rels_disease_acompany,  '并发疾病')
        self.create_relationship('Disease', 'Department', self.rels_disease_department,  '治疗科室')

    '''创建实体关联边'''

    # def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
    #     count = 0
    #     # 去重处理
    #     set_edges = []
    #     for edge in edges:
    #         set_edges.append('###'.join(edge))
    #     relation_num = len(set(set_edges))  # 去重后的relation长度
    #     for edge in set(set_edges):
    #         edge = edge.split('###')
    #         p = edge[0]
    #         q = edge[2]
    #         query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
    #             start_node, end_node, p, q, rel_type, rel_name)
    #         try:
    #             self.g.run(query)
    #             count += 1
    #             print(rel_type, count, relation_num)
    #         except Exception as e:
    #             print(e)
    #     return
    def create_relationship(self, start_node, end_node, edges, rel_type):
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        relation_num = len(set(set_edges))  # 去重后的relation长度
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[2]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s]->(q)" % (
                start_node, end_node, p, q, rel_type)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, relation_num)
            except Exception as e:
                print(e)
        return

    '''导出数据'''

    def export_data(self):
        # Drugs, Recipes, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommand_food, rels_not_eat, rels_do_eat, rels_between_departments, rels_common_drug, rels_drug_producer, rels_recommand_drug, rels_disease_symptom, rels_disease_acompany, rels_disease_department = self.read_nodes()
        f_drug = open('drug.txt', 'w+')
        f_recipe = open('recipe.txt', 'w+')
        f_food = open('food.txt', 'w+')
        f_check = open('check.txt', 'w+')
        f_department = open('department.txt', 'w+')
        f_producer = open('producer.txt', 'w+')
        f_symptom = open('symptoms.txt', 'w+')
        f_disease = open('disease.txt', 'w+')

        f_drug.write('\n'.join(list(self.drugs)))
        f_recipe.write('\n'.join(list(self.recipes)))
        f_food.write('\n'.join(list(self.foods)))
        f_check.write('\n'.join(list(self.checks)))
        f_department.write('\n'.join(list(self.departments)))
        f_producer.write('\n'.join(list(self.producers)))
        f_symptom.write('\n'.join(list(self.symptoms)))
        f_disease.write('\n'.join(list(self.diseases)))

        f_drug.close()
        f_recipe.close()
        f_food.close()
        f_check.close()
        f_department.close()
        f_producer.close()
        f_symptom.close()
        f_disease.close()

        return


if __name__ == '__main__':
    handler = MedicalGraph()
    handler.read_nodes()
    handler.create_graphnodes()
    handler.create_graphrels()
    # handler.export_data()
