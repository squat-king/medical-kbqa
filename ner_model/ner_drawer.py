from datetime import time

import matplotlib.pyplot as plt


# 添加绘制损失曲线和评估曲线的函数
def save_train_history_image(train_history_list, eval_history_list,
                             history_image_path, data_type):
    # train_history_list: 训练历史结果数据
    # eval_history_list: 验证历史结果数据
    # history_image_path: 历史数据生成图像的保存路径
    # data_type: 数据类型
    # 直接开始画图
    plt.plot(train_history_list, label="Train %s History" % (data_type))
    plt.plot(eval_history_list, label="Validate %s History" % (data_type))
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel(data_type)
    plt.savefig(history_image_path.replace("plot", data_type))
    plt.close()


def text_read(filename):
    arr = []
    f = open(filename, 'r')
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        arr.append(float(line.rstrip('\n')))
    return arr


time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
model_saved_path = "model/bilstm_crf_state_dict_%s.pt" % (time_str)
image_save_path = "data/bilstm_crf_train_plot_%s.png" % (time_str)

train_loss_history = text_read("data/train_loss_history.txt")
train_acc_history = text_read("data/train_acc_history.txt")
train_recall_history = text_read("data/train_recall_history.txt")
train_f1_history = text_read("data/train_f1_history.txt")

eval_loss_history = text_read("data/eval_loss_history.txt")
eval_acc_history = text_read("data/eval_acc_history.txt")
eval_recall_history = text_read("data/eval_recall_history.txt")
eval_f1_history = text_read("data/eval_f1_history.txt")

save_train_history_image(train_loss_history, eval_loss_history, image_save_path, "Loss")

# 将准确率的历史数据画成图片
save_train_history_image(train_acc_history, eval_acc_history, image_save_path, "Acc")

# 将召回率的历史数据画成图片
save_train_history_image(train_recall_history, eval_recall_history, image_save_path, "Recall")

# 将F1值的历史数据画成图片
save_train_history_image(train_f1_history, eval_f1_history, image_save_path, "F1")
