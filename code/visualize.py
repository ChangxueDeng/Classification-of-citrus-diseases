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
    
    plt.figure(figsize=(30, 10))  # 调整图形大小以容纳新的子图
    
    # 准确率对比
    plt.subplot(1, 5, 1)  # 修改子图位置
    for name, data in results.items():
        plt.plot(data["train_acc"], label=name)
    plt.title("Training Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 损失对比
    plt.subplot(1, 5, 2)  # 修改子图位置
    for name, data in results.items():
        plt.plot(data["train_loss"], label=name)
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 5, 3)  # 修改子图位置
    for name, data in results.items():
        plt.plot(data["val_acc"], label=name)
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
        # 损失对比
    plt.subplot(1, 5, 4)  # 修改子图位置
    for name, data in results.items():
        plt.plot(data["val_loss"], label=name)
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Macro F1 对比
    plt.subplot(1, 5, 5)  # 新增子图
    for name, data in results.items():
        plt.plot(data["val_macro_f1"], label=name)  # 增加 macro_f1 绘制
    plt.title("Validation Macro F1 Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()

# def plot_confusion_matrix(model_name):
#     with open(f"test_results/{model_name}_test.json") as f:
#         result = json.load(f)
    
#     cm = np.array(result["confusion_matrix_percent"])
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#     plt.title(f"Confusion Matrix - {model_name}")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.savefig(f"{model_name}_confusion.png")
#     plt.show()
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(model_name):
    # 打开指定模型的测试结果 JSON 文件
    with open(f"test_results/{model_name}_test.json") as f:
        # 加载 JSON 数据到 result 变量
        result = json.load(f)

    # 从结果中提取混淆矩阵百分比数据并转换为 numpy 数组
    cm_percent = np.array(result["confusion_matrix_percent"])

    # 设置图形大小
    plt.figure(figsize=(8, 6))

    # 绘制热力图，显示百分比数据
    sns.heatmap(cm_percent, annot=True, fmt=".2%", cmap="Blues")

    # 设置图形标题，包含模型名称
    plt.title(f"Confusion Matrix - {model_name}")
    # 设置 x 轴标签为 "Predicted"
    plt.xlabel("Predicted")
    # 设置 y 轴标签为 "Actual"
    plt.ylabel("Actual")

    # 保存绘制好的混淆矩阵图，文件名包含模型名称
    plt.savefig(f"{model_name}_confusion.png")

    # 显示绘制好的图形
    plt.show()


if __name__ == "__main__":
    # 示例使用
    # models = ["mobilenetv3_large","mobilenetv2","efficientnet-b0","mobilenetv3_small","mobilenetv4_conv_small"]  # 需要对比的模型列表
    # compare_models(models)
    
    # 查看单个模型的混淆矩阵
    #plot_confusion_matrix("mobilenetv2")
    plot_confusion_matrix("mobilenetv3_large")
    # plot_confusion_matrix("efficientnet-b0")
    # plot_confusion_matrix("mobilenetv3_small")
    #plot_confusion_matrix("mobilenetv4_conv_small")