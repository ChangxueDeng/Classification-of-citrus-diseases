import os
import shutil
from sklearn.model_selection import train_test_split

# 配置参数
dataset_path = "./dataset/all"  # 数据集根目录
output_path = "./dataset"  # 输出目录
test_ratio = 0.15    # 测试集比例
val_ratio = 0.15     # 验证集比例（从训练集剩余部分划分）
seed = 42            # 随机种子

# 创建目标目录结构
for split in ['train', 'val', 'test']:
    for cls in ['Citrus Black spot', 'Citrus Canker', 'Citrus Greening', 'Citrus Healthy']:
        os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

# 遍历每个类别进行划分
for class_name in os.listdir(dataset_path):
    if not os.path.isdir(os.path.join(dataset_path, class_name)):
        continue
    
    # 获取该类别所有图像路径
    class_dir = os.path.join(dataset_path, class_name)
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
    
    # 第一次划分：分离测试集
    train_val, test = train_test_split(
        images,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True
    )
    
    # 第二次划分：分离验证集
    train, val = train_test_split(
        train_val,
        test_size=val_ratio/(1-test_ratio),  # 调整比例
        random_state=seed
    )
    
    # 复制文件到目标目录
    def copy_files(files, split):
        for f in files:
            shutil.copy(f, os.path.join(output_path, split, class_name))
    
    copy_files(train, 'train')
    copy_files(val, 'val')
    copy_files(test, 'test')
    
    # 打印统计信息
    print(f"Class {class_name}:")
    print(f"  Total: {len(images)}")
    print(f"  Train: {len(train)} ({len(train)/len(images):.1%})")
    print(f"  Val:   {len(val)} ({len(val)/len(images):.1%})")
    print(f"  Test:  {len(test)} ({len(test)/len(images):.1%})\n")