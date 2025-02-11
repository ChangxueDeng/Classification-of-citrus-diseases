import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results(model_names):
    results = {}
    for name in model_names:
        try:
            with open(f"training_results/{name}_log.json") as f:
                results[name] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: No training data found for {name}")
    return results

def compare_models(models_to_compare):
    results = load_results(models_to_compare)
    
    plt.figure(figsize=(20, 5))  # 调整图形大小以容纳新的子图
    
    # 准确率对比
    plt.subplot(1, 3, 1)  # 修改子图位置
    for name, data in results.items():
        plt.plot(data["val_acc"], label=name)
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # 损失对比
    plt.subplot(1, 3, 2)  # 修改子图位置
    for name, data in results.items():
        plt.plot(data["train_loss"], label=name)
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Macro F1 对比
    plt.subplot(1, 3, 3)  # 新增子图
    for name, data in results.items():
        plt.plot(data["val_macro_f1"], label=name)  # 增加 macro_f1 绘制
    plt.title("Validation Macro F1 Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()

def plot_confusion_matrix(model_name):
    with open(f"test_results/{model_name}_test.json") as f:
        result = json.load(f)
    
    cm = np.array(result["confusion_matrix"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{model_name}_confusion.png")
    plt.show()

if __name__ == "__main__":
    # 示例使用
    models = ["mobilenetv3_large"]  # 需要对比的模型列表
    compare_models(models)
    
    # 查看单个模型的混淆矩阵
    #plot_confusion_matrix("mobilenetv3_large")