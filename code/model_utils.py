import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torchvision
import os
from config import Config
from tqdm import tqdm

def count_parameters(model):
    """统计PyTorch模型的参数量，单位为M"""
    return round(sum(p.numel() for p in model.parameters()) / 1_000_000, 2)

def load_model(model_path):
    # 加载模型
    model = torch.load(os.path.join(Config.model_save_dir, f"{model_name}_best.pt"), map_location=Config.device, weights_only=False)
    model = model.to(Config.device)
    model.eval()
        
    model.eval()
    return model
models = ["resnet18","efficientnet_b0","mobilenetv2","mobilenetv3_large","mobilenetv3_small","shufflenetv2_x1.0","shufflenetv2_x1.5","shufflenetv2_x2.0"]  # 需要对比的模型列表
for model_name in models:
    model = load_model(model_name)
    print(f"Model: {model_name}, Parameters: {count_parameters(model)}")