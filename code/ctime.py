import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torchvision
import os
import time
from config import Config
from tqdm import tqdm
from print_model import CBAM
def test_model(model_name):
    # 加载模型
    model = torch.load(os.path.join(Config.model_save_dir, f"{model_name}_best.pt"), 
                      map_location=Config.device, 
                      weights_only=False)
    model = model.to(Config.device)
    model.eval()
    
    # 加载测试数据
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(Config.data_root, "test"),
        transform=Config.val_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8)  # 修改batch_size为1以测量单张图像
    
    # 测量推理时间
    total_time = 0.0
    num_images = 0
    
    # 执行测试
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            inputs = inputs.to(Config.device)
            
            # 预热
            if num_images == 0:
                for _ in range(10):
                    _ = model(inputs)
            
            # 测量时间 - 兼容CPU和GPU
            if str(Config.device) == 'cpu':
                start_time = time.time()
                outputs = model(inputs)
                elapsed_ms = (time.time() - start_time) * 1000
            else:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                outputs = model(inputs)
                end_time.record()
                torch.cuda.synchronize()
                elapsed_ms = start_time.elapsed_time(end_time)
            
            total_time += elapsed_ms
            num_images += 1
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    avg_time_ms = total_time / num_images
    
    # 保存时间结果
    time_result = {
        "model": model_name,
        "average_inference_time_ms": avg_time_ms,
        "total_images": num_images
    }
    
    if not os.path.exists(Config.test_time_result_dir):
        os.makedirs(Config.test_time_result_dir)
    with open(f"{Config.test_time_result_dir}/{model_name}_time.json", "w") as f:
        json.dump(time_result, f)
    print(f"Time results saved to {Config.test_time_result_dir}/{model_name}_time.json")
    
    return time_result

if __name__ == "__main__":
    model_name = "efficientnet_b3"  # 修改为需要测试的模型
    #models = ["resnet18","efficientnet_b0","mobilenetv2","mobilenetv3_large","mobilenetv3_small","shufflenetv2_x1.0","shufflenetv2_x1.5","shufflenetv2_x2.0"]  # 需要对比的模型列表
    models = ["mobilenetv3_large_CBMA"]
    for model_name in models:
        print(f"Model: {model_name}")
        test_model(model_name)