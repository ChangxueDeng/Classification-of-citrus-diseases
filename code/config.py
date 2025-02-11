import torch
import torchvision.transforms as transforms

class Config:
    # 模型配置
    model_name = "mobilenetv3_large"  # 手动修改需要训练的模型
    supported_models = ["mobilenetv3_large", "mobilenetv2", 
                       "mobilenetv3_small", "resnet18"]
    
    # 训练参数
    batch_size = 10
    epochs = 20
    lr = 0.001
    input_size = 224
    
    # 路径配置
    data_root = "./dataset/temp"
    model_save_dir = "./saved_models"
    result_dir = "./training_results"
    
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])