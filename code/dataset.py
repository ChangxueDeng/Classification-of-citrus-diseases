import os
import random

# 类型标签
class_label = -1

# 训练集与验证集读取
train_datalist = []
valid_datalist = []
train_datalist_path = "./data/train/"
valid_datalist_path = "./data/valid/"
# 训练与验证数据列表
train_datalist_txt = "train.txt"
valid_datalist_txt = "test.txt"

# 读取文件存进列表
def data_in(data_l, data_p, label):
    for root,dirs,filenames in os.walk(data_p):
        for i in filenames:
            data = root+"/"+i+"\t"+str(label)+"\n"
            #print(data)
            data_l.append(data)   # 依次添加，不清空
        label += 1
    random.shuffle(data_l)# 打乱顺序
    return

# 写入数据文件
def write_txt(txt, datalist):
    with open(txt, 'w', encoding='UTF-8') as f:
        for data_img in datalist:
            f.write(str(data_img))
    return

data_in(train_datalist,train_datalist_path,class_label)
data_in(valid_datalist,valid_datalist_path,class_label)

write_txt(train_datalist_txt, train_datalist)
write_txt(valid_datalist_txt, valid_datalist)