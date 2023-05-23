import torch
import torch.utils.data.distributed
from torch.utils.mobile_optimizer import optimize_for_mobile
device = "cuda" if torch.cuda.is_available() else 'cpu'
#model_pth = './Model/model8.pth'
mobile_pt = './mobile.pt'
# if device == 'cpu':
#     model = torch.load("Model/model8.pth",map_location='cpu')
# else :
#     model = torch.load("Model/model8.pth")
model = torch.load('Model/model8.pth',map_location=torch.device('cpu'))
model.eval() # 模型设为评估模式
# 1张3通道224*224的图片
input_tensor = torch.rand(1, 3, 224, 224) # 设定输入数据格式
mobile = torch.jit.trace(model,input_tensor) # 模型转化
#optimized_traced_model = optimize_for_mobile(mobile)
mobile.save(mobile_pt) # 保存文件