import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torchvision
import os
from config import Config
from tqdm import tqdm



def test_model(model_name):
    # 加载模型
    if Config.device == "cpu":
        model = torch.load(os.path.join(Config.model_save_dir, f"{model_name}_best.pt"), map_location="cpu",weights_only=False)
    else:
        model = torch.load(os.path.join(Config.model_save_dir, f"{model_name}_best.pt"), weights_only=False, map_location='cuda:0')
    model = model.to(Config.device)
    model.eval()
    
    # 加载测试数据
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(Config.data_root, "test"),
        transform=Config.val_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size,num_workers=8)
    
    # 执行测试
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            outputs = model(inputs.to(Config.device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 生成报告
    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 保存结果
    result = {
        "model": model_name,
        "test_accuracy": report["accuracy"],
        "classification_report": report,
        
        # "confusion_matrix": cm.tolist(),
        "confusion_matrix_percent": (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
    }
    
    if not os.path.exists(Config.test_result_dir):
        os.makedirs(Config.test_result_dir)
    with open(f"{Config.test_result_dir}/{model_name}_test.json", "w") as f:
        json.dump(result, f)
    print(f"Test results saved to {Config.test_result_dir}/{model_name}_test.json")
    return result

if __name__ == "__main__":
    model_name = "efficientnet_b3"  # 修改为需要测试的模型
    test_model(model_name)