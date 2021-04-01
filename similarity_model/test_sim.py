import random
from similarity_model.sim_model import *
import config
import torch
# a=torch.tensor([
#     [0.1,0.2],
#     [0.1,-0.2],
#     [0.3,0.4]
# ])
# c=torch.softmax(a,dim=-1)
# print(c)
# b=torch.argmax(c,-1)
# print(torch.argmax(b,-1))
# arr=[0,1,2,3,4,5,6,7,8,9,10]
# for i in range(100):
#     print(random.choice(arr))
# arr=[(0,0.03),(0,0.01),(0,0.012),(0,0.05)]
# print(sorted(arr,key=lambda x:x[1],reverse=True))
# str='《高等数学》的出版社是哪里？'
# char='《'
# print(str.replace(char,""))

# sim_model=SimModel(r"F:\毕业设计\QASystem\online\doctor_online\similarity_model\output\model1002")
sim_model=SimModel(config.ner_model)
idx,value=sim_model.predict("飞盘是什么时候发明的？",["发明时间","菜品口味","管辖权归属","研发时间"])

# idx,value=sim_model.predict("《计算机应用基础》这本书的出版社是哪个？",["疾病","宜吃","应该吃","《计算机应用基础》这本书的出版社是哪个","生产","症状"])
print("最相似的属性的下标是 "+str(idx)+" 相似度是 : "+str(value))

# import numpy as np
# arr=[1,2,3,4,5,6,7,8,9]
# arr=np.array(arr)
# print(arr.argmax(0))

# a = torch.tensor([
#     [1, 1],
#     [1, 1],
#     [1, 1],
#     [1, 1],
#     [1, 1]
# ],dtype=float)
# b = torch.tensor([
#     [1, 1],
#     [1, 1],
#     [1, 1],
#     [1, 1],
#     [1, 1]]
# ,dtype=float)
# a = a.view(5, 2)
# b = b.view(5, 2)
# c = torch.add(a, b)
# d=torch.exp(c)
# e=torch.sum(c,1).view(-1,1)
#
# print(c)
# print(d)
# print(e)