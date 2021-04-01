# d2 = {'spam': 2, 'ham': 1, 'eggs': 3}
# arr=d2.keys()
# print(list(arr))
import re

from py2neo import *

graph = Graph("http://localhost:7474", username="neo4j", password="root")
graph.delete_all()

a = Node('Entity', name='感冒')
b = Node('Entity', name='感冒灵颗粒')
c = Node('Entity', name='哈药六厂')
d = Node('Entity', name='流鼻涕')
e = Node('Food', name='苹果')
f = Node('Department', name='感冒科')
g = Node('Department', name='疾病科')
r1 = Relationship(a, '宜吃药物', b)
r2 = Relationship(b, '生产厂家', c)
r3 = Relationship(a, '症状', d)
r4 = Relationship(a, '推荐食物', e)
r5 = Relationship(a, '所属科室', f)
r6 = Relationship(f, '父科室', g)

graph.create(a)
graph.create(b)
graph.create(c)
graph.create(d)
graph.create(e)
graph.create(f)
graph.create(g)

graph.create(r1)
graph.create(r2)
graph.create(r3)
graph.create(r4)
graph.create(r5)
graph.create(r6)

matcher = NodeMatcher(graph)

node = matcher.match('Entity', name='感冒')

# 查询name为感冒的Entity节点
data1 = graph.run('MATCH (e:Entity{name:"感冒"}) return e')
# 返回一个Cursor类型，使用data.data()方法返回一个list,
# [{'e': Node('Entity', name='感冒')}]
# 每个节点用一个dict表示，key='e',value=Node(实体名,属性值)
print(data1.data())
# 查询name为感冒的Entity节点的所有属性（暂时有问题）
data1 = graph.run('MATCH (e:Entity{name:"感冒"}) return e')
Node = data1.data()[0]['e']
print()
# 查询name为感冒的Entity节点的所有关系，保存为{key,value}形式
# data1 = graph.run('MATCH (e1:Entity{name:"感冒"})-[r]-(e2) WHERE e1.name =~ ".*感冒.*" return r')
print('********************')
data1 = graph.run('MATCH (e1)-[r]-(e2) WHERE e1.name =~ ".*感冒.*" return r')
r_list = data1.data()
print(r_list)
print('********************')
rel_subj={}
for r in r_list:
    s=str(r['r'])
    s = s[s.index('['):]
    print(r)
    rel = re.findall(r":(.+?) ", s)
    subj = re.findall(r"\((.+?)\)", s)
    rel_subj[rel[0]]=subj[0]
print(rel_subj)
# 查询name包含'感'字的Entity节点的所有关系
data1 = graph.run('MATCH (e1:Entity)-[r]-(e2:Entity) WHERE e1.name =~ ".*感冒.*" return r')
r_list = data1.data()
print(r_list)
# matcher = NodeMatcher(graph)

# node = matcher.match('Person')
# print list(node)
