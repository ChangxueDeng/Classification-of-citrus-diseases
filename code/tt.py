import os

# 定义要统计的类别和数据集划分
categories = ["Citrus Black spot", "Citrus Canker", "Citrus Greening", "Citrus Healthy"]
splits = ["train", "val", "test"]

# 遍历每个数据集划分和类别
for split in splits:
    print(f"\n{split.upper()}数据集统计:")
    for category in categories:
        folder_path = os.path.join("dataset", split, category)
        try:
            # 获取文件夹中的文件列表
            files = [f for f in os.listdir(folder_path) 
                    if os.path.isfile(os.path.join(folder_path, f))]
            # 统计文件数量
            count = len(files)
            print(f"{category}: {count}张图片")
        except FileNotFoundError:
            print(f"警告: 文件夹 {folder_path} 不存在")
        except Exception as e:
            print(f"处理 {folder_path} 时出错: {e}")