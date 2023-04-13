import torch
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torch.utils.data as D

class Data_Loader(Dataset):
    def __init__(self, txt_path, data_type):
        self.imgs_info = self.get_images(txt_path)
        self.data_type = data_type
        self.targetsize = 224
        #训练集
        self.train_tf = transforms.Compose([
            transforms.Resize(self.targetsize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        #测试集
        self.valid_tf = transforms.Compose([
            transforms.Resize(self.targetsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

        # 通过读取txt文档，返回信息
    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info

    def __get_labels__(self, imgs_info):
        return imgs_info[1]

    def padding_black(self, img):
        w, h = img.size
        scale = 224 / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

        # 我们在遍历数据集中返回的每一条数据
    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]  # 读取每一条数据，得到图片路径和标签值
        img = Image.open(img_path)  # 利用 Pillow打开图片
        img = img.convert('RGB')  # 将图片转变为RGB格式
        # img = self.padding_black(img)
        label = int(label)
        if self.data_type == 0:  # 对训练集和测试集分别处理
            img = self.train_tf(img)
        elif self.data_type == 1:
            img = self.valid_tf(img)
        return img, label
    def __len__(self):
        return len(self.imgs_info)





if __name__ == "__main__":
    test_data = "test.txt"
    test_loader = Data_Loader(test_data,1)
    size = len(test_loader)
    print(size)
    for i,(img,target) in enumerate(test_loader):
        print(f"----------------{i}---------------")
        print(img)
        print(target)