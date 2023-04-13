import time
import torch
import torch.utils.data as D
import torchvision
from efficientnet_pytorch import EfficientNet
from tensorboardX import SummaryWriter
import ranger21
import os
from torchsampler import ImbalancedDatasetSampler

# 模型训练函数
def eff_train(train_loader, model, criterion, optimizer, epoch,device):
    model.train()
    #运行时的准确率
    running_corrects = 0.0
    for i, (Input, target) in enumerate(train_loader):
        Input = Input.to(device)
        target = target.to(device)
        Output = model(Input)
        loss = criterion(Output, target)
        _, preds = torch.max(Output.data,1)
        running_corrects += torch.sum(preds == target).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Training loss = ", loss.item())    # 每轮中的100次输出一次loss
    epoch_acc = running_corrects / len(train_loader.dataset)        #计算每轮的准确率
    print("Training Accuracy = ", epoch_acc)          # 输出每轮的准确率
    
    writer.add_scalar('efficientnet-b4', epoch_acc, global_step=epoch)     # 将准确率写入到tensorboard中
#验证
def valid(valid_loader, model,device):
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for i ,(Input, target) in enumerate(valid_loader):
            Input = Input.to(device)
            target = target.to(device)
            Output = model(Input)
            _,preds = torch.max(Output.data,1)
            correct += torch.sum(preds == target).item()
    valid_correct = correct / len(valid_loader.dataset)
    print("valid Accuracy = ", valid_correct)
    return valid_correct

#加载预训练模型："efficientnet-b4"
def Model():
    model_ft = EfficientNet.from_pretrained("efficientnet-b4")
    num_ftrs = model_ft._fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs,4)
    return model_ft



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    epochs = 20
    batch_size = 64
    lr = 0.001
    print(device)
    #--------------------------transforms数据增强----------------------------
    transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224]),
            torchvision.transforms.RandomHorizontalFlip(), #随机水平翻转
            torchvision.transforms.RandomVerticalFlip(), #随机垂直翻
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
    transforms_valid = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224,224]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
    #--------------------------------------加载数据集---------------------------------------
    #训练集
    train_dataset = torchvision.datasets.ImageFolder(os.path.join("./dataset/train"), transforms_train)
    print(train_dataset.class_to_idx)
    #验证集
    valid_dataset = torchvision.datasets.ImageFolder(os.path.join("./dataset/valid"),transforms_valid)

    #-------------------------------------生成dataloader-----------------------------------------
    train_loader = D.DataLoader(train_dataset,sampler=ImbalancedDatasetSampler(train_dataset) ,batch_size=batch_size)
    valid_loader = D.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    #---------------------------- 加载模型------------------------------------
    model = Model()
    model = model.to(device)
    #print(model)
    # 采用ranger21优化器，交叉熵损失函数
    optimizer = ranger21.Ranger21(model.parameters(),lr=lr,num_epochs=epochs,num_batches_per_epoch=len(train_loader))
    #optimizer = torch.optim.SGD()
    criterion = torch.nn.CrossEntropyLoss()

    # 将tensorboard文件写入runs文件夹中
    writer = SummaryWriter('./runs')

    # 定义一个开始时间，用于查看整个模型训练耗时
    start_time = time.time()

    #------------------------------ 开始训练-------------------------------------
    valid_acc = 0   #验证集准确率，用于判断是否保存模型
    for epoch in range(epochs):
        print("*********************   Epoch ", epoch, " ************************")
        eff_train(train_loader, model, criterion, optimizer, epoch,device)  # 调用前面定义的训练方法
        validing_acc = valid(valid_loader,model,device)
        if valid_acc <= validing_acc :
            torch.save(model,"best_model/model.pkl")
            valid_acc = validing_acc #更新
            print("Save PyTorch Model to best_model/model.pkl")
        epoch = epoch + 1
    # 定义一个结束时间
    end_time = time.time()
    # 用开始时间-结束时间=总耗时
    time = end_time - start_time
    print(time)
    # 关闭tensorboard写入
    writer.close()

    #classes = { 0 : "Citrus Black spot 0", 1: "Citrus canker 1", 2 : "Citrus greening 2",3:"Citrus Healthy 3"}