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
import torchvision
from tqdm import tqdm
import time

a = True

def get_model():
    model = None
    if Config.model_name == "mobilenetv3_large":
        model = torchvision.models.mobilenet_v3_large(weigth=torchvision.models.MobileNet_V3_Large_Weights)
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        #修改分类器最后一层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
        # 解冻分类器参数（可选，若需训练整个分类器）
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif Config.model_name == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(weigth=torchvision.models.MobileNet_V2_Weights)
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        # 修改分类器最后一层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
        # 解冻分类器参数（可选，若需训练整个分类器）
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif Config.model_name == "mobilenetv3_small":
        model = torchvision.models.mobilenet_v3_small(weigth=torchvision.models.MobileNet_V3_Small_Weights)
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        # 修改分类器最后一层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 4)
        # 解冻分类器参数（可选，若需训练整个分类器）
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif Config.model_name == "resnet18":
        model = torchvision.models.resnet18(weigth=torchvision.models.ResNet18_Weights)
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        # 修改分类器最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)
        # 解冻全连接层参数
        for param in model.fc.parameters():
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
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(Config.data_root, "train"),
        transform=Config.train_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=Config.batch_size
    )

    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(Config.data_root, "val"),
        transform=Config.val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False
    )
    
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 训练记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1" : []
    }

    # 训练循环
    for epoch in range(Config.epochs):
        # 训练步骤...
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for Input, target in tqdm(train_loader, desc=f"Training {Config.model_name}"):
            Input = Input.to(Config.device)
            target = target.to(Config.device)
            Output = model(Input) # 前向预测，获得当前 batch 的预测结果
            loss = criterion(Output, target) # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数
            train_loss += loss.item() * Input.size(0)

            _, preds = torch.max(Output.data,1) # 获得当前 batch 的预测结果
            train_acc += torch.sum(preds == target).item() # 计算当前 batch 的准确率

            optimizer.zero_grad()
            loss.backward()     # 损失函数对神经网络权重反向传播求梯度
            optimizer.step()
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)

        # 验证步骤
        model.eval()
        val_preds = []    # 新增：存储所有预测结果
        val_labels = []   # 新增：存储所有真实标签
        val_acc = 0.0
        with torch.no_grad():
            for Input, target in val_loader:
                Input = Input.to(Config.device)
                target = target.to(Config.device)
                Output = model(Input)
                _, preds = torch.max(Output.data, 1)
                val_acc += torch.sum(preds == target).item()
                
                # 新增：收集全部预测结果和标签
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(target.cpu().numpy())

        val_acc = val_acc / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')  # 关键指标
        # # 早停判断与模型保存
        # if val_f1 > best_f1:
        #     best_f1 = val_f1
        #     no_improve = 0
        #     best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝最佳参数
        #     torch.save(best_model_wts,  # 立即保存最佳模型
        #             os.path.join(Config.model_save_dir, f"{Config.model_name}_best.pth"))
        # else:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         print(f'Early stopping at epoch {epoch+1}, best f1: {best_f1:.4f}')
        #         model.load_state_dict(best_model_wts)  # 恢复最佳模型参数
        #         break

        # 保存记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(f'Epoch {epoch+1}/{Config.epochs}, Train Loss: {train_loss:.4f},Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_f1:.4f}')
    
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

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(Config.model_save_dir, exist_ok=True)
    os.makedirs(Config.result_dir, exist_ok=True)
    
    # 开始训练
    train()