import time
import pandas as pd
import torch
import torch.utils.data as D
import torchvision
from efficientnet_pytorch import EfficientNet
import os
from pycm import*
import matplotlib.pyplot as plt

#测试
def test(test_loader, model):
    correct = 0
    res = []
    y_true = []
    with torch.no_grad():
        for i ,(Input, target) in enumerate(test_loader):
            Input = Input.to(device)
            target = target.to(device)
            Output = model(Input)
            _,preds = torch.max(Output,1)
            print("target: ",target,"   preds: ", preds)
            res += preds.cpu().numpy().tolist()
            y_true += target.cpu().numpy().tolist()
            correct += torch.sum(preds == target).item()
    test_correct = correct / len(test_loader.dataset)
    cm = ConfusionMatrix(y_true,res)
    print(cm)
    cm.print_matrix()
    cm.print_normalized_matrix()
    cm.plot(normalized=True,one_vs_all=True,cmap='Blues')
    plt.show()
    
    print("test Accuracy = ", test_correct)
    return res

def getlist(dataset_samples):
    filelist = []
    tager = []
    for i in range(len(dataset_samples)):
        filelist.append(dataset_samples[i][0])
        tager.append(dataset_samples[i][1])
    return [filelist,tager]

def Model():
    model_ft = EfficientNet.from_pretrained("efficientnet-b4")
    num_ftrs = model_ft._fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs,4)
    return model_ft

transforms_valid = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224,224]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    ])

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(device)

batch_size = 1

#测试集
test_dataset = torchvision.datasets.ImageFolder(os.path.join("./dataset/test"),transforms_valid)
print(test_dataset.class_to_idx)
test_loader = D.DataLoader(test_dataset, batch_size=batch_size)

# 进行预测
#model = Model()
#model.load_state_dict(torch.load("best_model/model.pth",map_location=torch.device('cpu')))
if device == 'cpu':
    model = torch.load("Model/model5.pkl",map_location='cpu')
else :
    
    model = torch.load("Model/model5.pkl")
model = model.to(device)
model.eval()
res = test(test_loader,model)

#写入csv
filelist, target_list = getlist(test_dataset.samples)
result = [list(i) for i in zip(filelist,target_list,res)]
result = pd.DataFrame(result,columns=['filename','target','pred'])
result.to_csv('Result.csv')