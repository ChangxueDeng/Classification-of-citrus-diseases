from torchvision import models
import torch.nn as nn
modes = models.shufflenet_v2_x2_0(pretrained=True )
n = modes.fc.in_features
print(n)
modes.fc = nn.Linear(n, 4)
print(modes)
#print(modes)
# model = models.resnet18(pretrained=True)

# for param in model.parameters():
#     param.requires_grad = False

# for param in model.fc.parameters():
#     param.requires_grad = True
#print(model)
# 打印全连接层参数的 requires_grad 属性，验证是否设置成功
# for name, param in model.fc.named_parameters():
#     print(f"Parameter {name}: requires_grad = {param.requires_grad}")