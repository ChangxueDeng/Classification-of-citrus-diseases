import torch
import torch.nn as nn
import json
import os
import copy
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from config import Config
from config import BalancedAugmentationDataset
import torchvision
from tqdm import tqdm
import time
import timm
from torchvision.datasets import ImageFolder
from test import test_model
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_model():
    model = None
    if Config.model_name == "mobilenetv3_large":
        model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
        #修改分类器最后一层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        # 修改分类器最后一层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "mobilenetv3_small":
        model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        # 修改分类器最后一层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "shufflenetv2_x2.0":
        model = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
        # 修改分类器最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "shufflenetv2_x1.0":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        # 修改分类器最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "shufflenetv2_x1.5":
        model = torchvision.models.shufflenet_v2_x1_5(pretrained=True)
        # 修改分类器最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "mobilenetv4_conv_small":
        model = timm.create_model('mobilenetv4_conv_small', pretrained=True, cache_dir = Config.timm_cache_dir)
        # 修改分类器最后一层
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        # 修改分类器最后一层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "efficientnet_b1":
        model = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT)
        # 修改分类器最后一层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "efficientnet_b3":
        model = torchvision.models.efficientnet_b3(weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
    elif Config.model_name == "resnet50":
        model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)

    return model.to(Config.device)
    model = models.efficientnet_b3(weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 4)
    # 冻结教师参数
    for param in model.parameters():
        param.requires_grad = False
            # 解冻全连接层参数
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model.to(Config.device)

def save_training_log(log_data):
    Path(Config.result_dir).mkdir(exist_ok=True)
    log_path = os.path.join(Config.result_dir, f"{Config.model_name}_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f)

def train():
    start_time = time.time()
    # 初始化

    # train_dataset = torchvision.datasets.ImageFolder(
    #     os.path.join(Config.data_root, "train"),
    #     transform=Config.train_transform
    # )
    train_dataset = BalancedAugmentationDataset(
    os.path.join(Config.data_root, "train"),
    minority_classes=Config.minority_classes,
    majority_transform=Config.train_transform,     # 多数类用普通增强
    minority_transform=Config.aggressive_transform # 少数类用激进增强
    )

    # --- 使用 sklearn 计算类别权重 ---
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # 获取所有样本的标签（整数索引）
    y = np.array(train_dataset.targets)  # 直接使用数据集的 targets

    # 计算类别权重（自动处理类别频率）
    class_weights = compute_class_weight(
        class_weight="balanced",  # 权重反比于类别频率
        classes=np.unique(y),      # 所有存在的类别索引（按顺序）
        y=y                        # 样本标签数组
    )
    class_weights = class_weights / np.sum(class_weights) * len(class_weights)
    # 转换为 PyTorch Tensor 并分配到设备
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(Config.device)

    # 输出结果验证
    class_names = train_dataset.classes
    print("Class weights (按类别顺序):")
    for cls, weight in zip(class_names, class_weights):
        print(f"{cls}: {weight:.2f}")
    print(class_weights_tensor)
    num_samples = len(train_dataset)
    print(f"num_samples: {num_samples}")
    train_loader = DataLoader(
        train_dataset,
        # shuffle=True,
        sampler=ImbalancedDatasetSampler(dataset = train_dataset, num_samples= num_samples),
        batch_size=Config.batch_size,
        num_workers=8
    )

    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(Config.data_root, "val"),
        transform=Config.val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=8
    )
    
    model = get_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr,weight_decay=0.01)
    
    #     optimizer = torch.optim.SGD(
    #     model.parameters(), 
    #     lr=0.001, 
    #     momentum=0.9, 
    #     weight_decay=1e-4
    # )
    #criterion = torch.nn.CrossEntropyLoss()
    #criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    #criterion = torch.nn.CrossEntropyLoss(weight= normalized_weights)
    #criterion = MultiClassFocalLoss(alpha=class_weights_tensor)
    #criterion = MultiClassFocalLoss(alpha=normalized_weights)
    #criterion = MultiClassFocalLoss(alpha=class_weights_tensor)
    # 方案1：余弦退火（推荐）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=Config.epochs,  # 周期长度（一般等于总epoch数）
        eta_min=1e-5          # 最小学习率
    )

    # 训练记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],  # 新增：验证集损失记录
        "val_acc": [],
        "val_macro_f1": [],
        "best_epoch": 0,
        "epochs": Config.epochs,
        "batch_size": Config.batch_size,
        "init_learning_rate": Config.lr,
        "learning_rate":[]
    }
    # 新增：初始化最佳指标跟踪
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    best_epoch = 0
    # 训练循环
    for epoch in range(Config.epochs):
        # 训练步骤...
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for Input, target in tqdm(train_loader, desc=f"Training {Config.model_name}"):
            Input = Input.to(Config.device,non_blocking=True)
            target = target.to(Config.device,non_blocking=True)
            Output = model(Input)  # 前向预测，获得当前 batch 的预测结果
            loss = criterion(Output, target)  # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数
            train_loss += loss.item() * Input.size(0)
            _, preds = torch.max(Output.data, 1)  # 获得当前 batch 的预测结果
            train_acc += torch.sum(preds == target).item()  # 计算当前 batch 的准确率
            optimizer.zero_grad()
            loss.backward()  # 损失函数对神经网络权重反向传播求梯度
            optimizer.step()
        # train_loss = train_loss / len(train_loader.dataset)
        # train_acc = train_acc / len(train_loader.dataset)
        train_loss = train_loss / num_samples
        train_acc = train_acc / num_samples
        # 验证步骤
        model.eval()
        val_preds = []  # 新增：存储所有预测结果
        val_labels = []  # 新增：存储所有真实标签
        val_acc = 0.0
        val_loss = 0.0  # 新增：验证集损失初始化
        with torch.no_grad():
            for Input, target in val_loader:
                Input = Input.to(Config.device)
                target = target.to(Config.device)
                Output = model(Input)
                loss = criterion(Output, target)  # 计算当前 batch 的验证损失
                val_loss += loss.item() * Input.size(0)
                _, preds = torch.max(Output.data, 1)
                val_acc += torch.sum(preds == target).item()
                # 新增：收集全部预测结果和标签
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(target.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)  # 计算验证集平均损失
        val_acc = val_acc / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')  # 关键指标
        # 新增：保存最佳模型逻辑
        if  val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model, os.path.join(Config.model_save_dir, f"{Config.model_name}_best.pt"))
            history["best_epoch"] = epoch + 1
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # 保存记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)  # 新增：保存验证集损失
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["learning_rate"].append(current_lr)  # 新增：保存当前学习率
        print(f'Epoch {epoch + 1}/{Config.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_f1:.4f}, Learning Rate: {current_lr:.2e}')
    
    # 保存最终模型和记录
    torch.save(model, os.path.join(Config.model_save_dir, f"{Config.model_name}.pt"))
    save_training_log(history)

    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    # 将总训练时间转换为小时和分钟
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
     # 打印总训练时间
    print(f'Total training time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds')
    print(f"Test best model: {Config.model_name}")
    test_model(Config.model_name)

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(Config.model_save_dir, exist_ok=True)
    os.makedirs(Config.result_dir, exist_ok=True)
    
    # 开始训练
    train()