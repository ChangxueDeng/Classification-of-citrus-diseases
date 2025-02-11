import torch
import torchvision
from PIL import Image

def get_img(img_path, img_transforms):
    img = Image.open(img_path).convert('RGB')
    img = img_transforms(img)
    img = img.unsqueeze(0)
    return img

def test(img, model):
    with torch.no_grad():
        Input = img
        Input = Input.to(device)
        Output = model(Input)
        _,pred = torch.max(Output,1)
        print(pred)
        pred = pred.cpu().numpy().tolist()
    return pred

img_transforms= torchvision.transforms.Compose([
        torchvision.transforms.Resize([224,224]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    ])


device = "cuda" if torch.cuda.is_available() else 'cpu'
#单张照片路径
img_path = "dataset\\test\\Citrus Greening\\CitrusGreening(7).png"
if device == 'cpu':
    model = torch.load("./mobileNetModel/model17.pth",map_location='cpu')
else :
    model = torch.load("./mobileNetModel/model17.pth")
model = model.to(device)
img = get_img(img_path, img_transforms)
pred = test(img,model)
print(pred)
if pred[0] == 0:
    print('Citrus Black spot')
elif pred[0] == 1:
    print('Citrus Canker')
elif pred[0] == 2:
    print('Citrus Greening')
else:
    print('Citrus Healthy')


