
import os 
import random 

import shutil


def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        if os.path.isfile(dstpath + fname):
            print("%s existed!"%(srcfile))
        else:
            shutil.copy(srcfile, dstpath + fname)          # 复制文件
            print ("copy %s -> %s"%(srcfile, dstpath + fname))
 
def accomplish(data_class, data_list):  #完成数据划分
    c0=c1=c2=c3 = 0
    if data_class == "train":
        cp_dir0 = './dataset/train/Citrus Black spot/'
        cp_dir1 = './dataset/train/Citrus canker/'
        cp_dir2 =  './dataset/train/Citrus greening/'
        cp_dir3 = './dataset/train/Citrus Healthy/'
    elif data_class == "test":
        cp_dir0 = './dataset/test/Citrus Black spot/'
        cp_dir1 = './dataset/test/Citrus canker/'
        cp_dir2 =  './dataset/test/Citrus greening/'
        cp_dir3 = './dataset/test/Citrus Healthy/'
    for i in range(len(data_list)):
        srcfile = data_list[i].split("\t")[0]
        if data_list[i].split("\t")[1] == "0\n":
            c0 += 1
            mycopyfile(srcfile, cp_dir0)
        elif data_list[i].split("\t")[1] == "1\n":
            c1 += 1
            mycopyfile(srcfile, cp_dir1)
        elif data_list[i].split("\t")[1] == "2\n":
            c2 += 1
            mycopyfile(srcfile, cp_dir2)
        elif data_list[i].split("\t")[1] == "3\n":
            c3 += 1
            mycopyfile(srcfile, cp_dir3)
    print(c0," ", c1, " ", c2, " ", c3)

#训练 ： 测试 = 9：1
train_ratio = 0.9 
test_ratio = 1 - train_ratio
 
rootdata = r"data/train" #根目录
 
train_list, test_list = [], [] 
data_list = []
 
class_flag = -1
for a, b, c in os.walk(rootdata):
    print(a)
    for i in range(len(c)):
        data_list.append(os.path.join(a, c[i]))
    for i in range(0, int(len(c) * train_ratio)):
        train_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'    # os.path.join 拼接起来，给一个从0开始逐一编号的标签
        train_list.append(train_data)
 
    for i in range(int(len(c) * train_ratio), len(c)):
        test_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
        test_list.append(test_data)
 
    class_flag += 1

accomplish("train", train_list)
accomplish("test", test_list)
