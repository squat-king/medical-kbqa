import matplotlib.pyplot as plt
import numpy as np

def text_read(filename):
  arr=[]
  f = open(filename,'r')
  lines = f.readlines()  # 读取全部内容
  for line in lines:
    arr.append(float(line.rstrip('\n')))
  return arr


train_loss=np.array(text_read('data/train_loss_history.txt'))
train_acc=np.array(text_read('data/train_acc_history.txt'))
train_recall=np.array(text_read('data/train_recall_history.txt'))
train_f1=np.array(text_read('data/train_f1_history.txt'))

eval_loss=np.array(text_read('data/eval_loss_history.txt'))
eval_acc=np.array(text_read('data/eval_acc_history.txt'))
eval_recall=np.array(text_read('data/eval_recall_history.txt'))
eval_f1=np.array(text_read('data/eval_f1_history.txt'))

print("最大的train_acc是%.4f,下标是%d"%(max(train_acc),train_acc.argmax(0)))
print("最大的eval_acc是%.4f,下标是%d"%(max(eval_acc),eval_acc.argmax(0)))
print("最大的train_recall是%.4f,下标是%d"%(max(train_recall),train_recall.argmax(0)))
print("最大的eval_recall是%.4f,下标是%d"%(max(eval_recall),eval_recall.argmax(0)))
print("最大的train_f1是%.4f,下标是%d"%(max(train_f1),train_f1.argmax(0)))
print("最大的eval_f1是%.4f,下标是%d"%(max(eval_f1),eval_f1.argmax(0)))
# print(all_train_loss[1105:1110])
# print(all_eval_loss[1105:1110])

plt.figure(0)
plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, color="red", label="Eval Loss")
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig("./loss_.png")

plt.figure(1)
plt.plot(train_acc, label="Train Acc")
plt.plot(eval_acc, color="red", label="Eval Acc")
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig("./acc_.png")

plt.figure(1)
plt.plot(train_f1, label="Train F1")
plt.plot(eval_f1, color="red", label="Eval F1")
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig("./f1_.png")
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