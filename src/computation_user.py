# @FunctionName: Model evaluation Computation User
# @Author: Wangyuzhan
# @Time: 2023/01/09

# 计算模型的计算量和参数量
import os
import torch

from thop import clever_format
from profile_my import profile

from model_code_user.Model_user import model_user

def model_computation_user_ext():
    # 定义参数量、计算量、存储量
    Macs = 0
    Params = 0
    Storage = 0

    Model, input = model_user()
    
    Macs, Params = profile(Model, inputs=(input, ))
    Macs, Params = clever_format([Macs, Params], "%.2f")

    torch.save(Model, "./model_user/" + "model_user" + ".pth")

    Storage = os.path.getsize("./model_user/" + "model_user" + ".pth")
    Storage = Storage / 2**20
    # Storage = clever_format(Storage, "%.2f")

    return Macs, Params, Storage

if __name__ == "__main__":
    
    model_computation_user_ext()

    # print(model_computation_user_ext())
