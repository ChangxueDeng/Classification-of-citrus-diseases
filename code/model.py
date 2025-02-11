# import torch
# import torch.utils.data as D
# import torchvision
# from efficientnet_pytorch import EfficientNet

# # def Model():
# #     model_ft = EfficientNet.from_pretrained("efficientnet-b4")
# #     num_ftrs = model_ft._fc.in_features
# #     model_ft.fc = torch.nn.Linear(num_ftrs,4)
# #     return model_ft
# device = 'cpu'
# model = Model()
# model.load_state_dict(torch.load("best_model/model.pth",map_location=torch.device('cpu')))
# model = model.to(device)
# torch.save(model, "./Model/model.pkl")