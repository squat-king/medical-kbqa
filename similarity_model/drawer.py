import matplotlib.pyplot as plt
import numpy as np

def text_read(filename):
  arr=[]
  f = open(filename,'r')
  lines = f.readlines()  # 读取全部内容
  for line in lines:
    arr.append(float(line.rstrip('\n')))
  return arr


all_train_loss=np.array(text_read('output/all_train_loss.txt'))
all_train_acc=np.array(text_read('output/all_train_acc.txt'))
all_eval_loss=np.array(text_read('output/all_eval_loss.txt'))
all_eval_acc=np.array(text_read('output/all_eval_acc.txt'))
print("最大的train_acc是%.4f,下标是%d"%(max(all_train_acc),all_train_acc.argmax(0)))
print("最大的eval_acc是%.4f,下标是%d"%(max(all_eval_acc),all_eval_acc.argmax(0)))
print("最大的train_loss是%.4f,下标是%d"%(max(all_train_loss),all_train_loss.argmax(0)))
print("最大的eval_loss是%.4f,下标是%d"%(max(all_eval_loss),all_eval_loss.argmax(0)))
print(all_train_loss[1105:1110])
print(all_eval_loss[1105:1110])
plt.figure(0)
plt.plot(all_train_loss, label="Train Loss")
plt.plot(all_eval_loss, color="red", label="Eval Loss")
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig("./loss_.png")

plt.figure(1)
plt.plot(all_train_acc, label="Train Acc")
plt.plot(all_eval_acc, color="red", label="Eval Acc")
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig("./acc_.png")

# print(arr)
# plt.figure(0)
# # plt.plot(all_train_loss, label="Train Loss")
# plt.plot(all_eval_loss, color="red", label="Eval Loss")
# plt.legend(loc="upper left")
# plt.grid(True)
# plt.savefig("./loss.png")
#
# plt.figure(1)
# # plt.plot(all_train_acc, label="Train Acc")
# plt.plot(all_eval_acc, color="red", label="Eval Acc")
# plt.legend(loc="upper left")
# plt.grid(True)
# plt.savefig("./acc.png")