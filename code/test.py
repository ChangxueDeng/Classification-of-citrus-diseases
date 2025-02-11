import torch
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from config import Config

def test_model(model_name):
    # 加载模型
    model = get_model()  # 需要实现模型加载逻辑
    model_path = os.path.join(Config.model_save_dir, f"{model_name}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 加载测试数据
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(Config.data_root, "test"),
        transform=Config.val_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
    
    # 执行测试
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
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
        "confusion_matrix": cm.tolist()
    }
    
    with open(f"test_results/{model_name}_test.json", "w") as f:
        json.dump(result, f)
    
    return result

if __name__ == "__main__":
    model_name = "mobilenetv3_large"  # 修改为需要测试的模型
    test_model(model_name)