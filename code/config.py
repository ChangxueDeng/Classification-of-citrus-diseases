import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
import os
from torchvision.datasets import ImageFolder

class BalancedAugmentationDataset(ImageFolder):
    def __init__(self, root, minority_classes, majority_transform, minority_transform ,save_augmented_dir=None):
        super().__init__(root)
        self.minority_class_ids = {
            self.class_to_idx[cls] for cls in minority_classes
        }
        self.majority_transform = majority_transform
        self.minority_transform = minority_transform


        self.save_augmented_dir = save_augmented_dir
        if self.save_augmented_dir:
            os.makedirs(self.save_augmented_dir, exist_ok=True)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        
        # 根据类别选择增强策略
        if target in self.minority_class_ids:
            img = self.minority_transform(img)
        else:
            img = self.majority_transform(img)
            
        return img, target

class Config:
    # 模型配置
    model_name = "mobilenetv3_large"  # 手动修改需要训练的模型
    supported_models = ["mobilenetv3_large", "mobilenetv2", "mobilenetv3_small",
                        "shufflenetv2_x2.0","mobilenetv4_conv_small","efficientnet_b0","shufflenetv2_x1.0","shufflenetv2_x1.5"]
    
    # 训练参数
    batch_size = 32
    epochs = 100
    lr = 0.001
    input_size = 224
    
    # 路径配置
    timm_cache_dir = "./cache"
    data_root = "./dataset"
    model_save_dir = "./saved_models"
    result_dir = "./training_results"
    test_result_dir = "./test_results"
    
    # 设备配置
    device= torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
    # 数据增强
    
    train_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        T.RandomApply([transforms.RandomRotation(degrees=30)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 公共基础增强（所有类别共用）
    base_transform = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # 针对少数类的激进增强（仅应用在训练集的少数类样本）
    aggressive_transform = T.Compose([
        transforms.Resize([input_size, input_size]),
        T.RandomHorizontalFlip(p=0.3),  # 合理水平镜像
        T.RandomVerticalFlip(p=0.3),
        T.RandomApply([T.RandomRotation(degrees=30)], p=0.3),    # 限制旋转角度
        T.RandomApply([T.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5)], p=0.3),  # 降低平移和剪切幅度
        # 颜色扰动（保持病斑颜色特征）
        T.RandomApply([T.ColorJitter(
            brightness=0.15,  # 降低亮度扰动
            contrast=0.15,    # 保持对比度差异
            saturation=0.1    # 最小化饱和度变化
        )], p=0.3),

        # 噪声与模糊（模拟成像差异）
        T.RandomApply([T.GaussianBlur(
            kernel_size=5, 
            sigma=(0.1, 1.0)  # 轻微模糊
        )], p=0.2),
        transforms.ToTensor(),
        # 局部遮挡（模拟自然遮挡）
        T.RandomErasing(
            p=0.3,
            scale=(0.02, 0.1),  # 小比例遮挡
            ratio=(0.5, 2),      # 保持合理形状
            value='random'
        ),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 普通增强（用于多数类Citrus Greening）
    train_transform = T.Compose([
        transforms.Resize([input_size, input_size]),
        T.RandomHorizontalFlip(p=0.3),
        T.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    minority_classes = ['Citrus Black spot', 'Citrus Canker', 'Citrus Healthy']

