# 计算模型的计算量和参数量
# from thop import profile
# from torchsummary import summary
import os
import torch

from thop import clever_format

from model_code.AlexNet import model_AlexNet
from model_code.MobileNet import model_MobileNet
from model_code.ResNet import model_ResNet
from model_code.VGG import model_VGG

def CreateModel(input_network="VGG"):

    # 模型列表
    ResNet_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    VGG_list = ["VGG11", "VGG13", "VGG16", "VGG19"]

    if input_network == "AlexNet":
        Model, Macs, Params = model_AlexNet()

    elif input_network == "MobileNet":
        Model, Macs, Params = model_MobileNet()

    elif input_network == "ResNet":
        # if input_network not in ResNet_list:
        #     input_network = "ResNet18"
        default_input_network = "ResNet18"
        Model, Macs, Params = model_ResNet(default_input_network)

    elif input_network == "VGG":
        # if input_network not in VGG_list:
        #     input_network = "VGG11"
        default_input_network = "VGG11"
        Model, Macs, Params = model_VGG(default_input_network)

    else:
        print("network error in computation")

    torch.save(Model, "./model/" + str(input_network) + ".pth")
    print("Saving model successfully!")

    return Macs, Params

def LoadModel(input_network):
    model = torch.load("./model/" + str(input_network) + ".pth")
    print("Loading model successfully")
    return model

def model_computation_predefined(input_network="VGG"):
    # 输入模型代码, 并保存模型到本地, 获得参数量和计算量
    Macs, Params = CreateModel(input_network)
    # 
    Storage = os.path.getsize("./model/" + str(input_network) + ".pth")
    Storage = Storage / 2**20
    Storage = clever_format(Storage, "%.2f")

    # # 加载保存的模型
    # model = LoadModel()

    # 计算模型的计算量和参数量

    print('Macs', Macs)
    print('Params', Params)
    print('Storage ', Storage, 'MB')

def model_computation_user():
    pass

def HMT_computation(cmd='Predefined_model', network='VGG'):
    if cmd == 'Predefined_model':
        # 系统预存模型 V1.0
        model_computation_predefined(network)
    elif cmd == 'User_defined_model':
        # 需要用户上传模型代码 后续补充
        model_computation_user()
    else:
        print("error in computation!")

if __name__ == "__main__":
    
    HMT_computation()
