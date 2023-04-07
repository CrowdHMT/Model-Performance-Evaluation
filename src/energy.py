from ast import Return
import os

import torch
import torch.nn as nn

from profile_my import profile

from model_code.AlexNet import model_AlexNet
from model_code.MobileNet import model_MobileNet
from model_code.ResNet import model_ResNet
from model_code.VGG import model_VGG
from model_code.Resnet_ope import *

import time

def CreateModel(input_network="VGG", compress_ope="null"):
    
    # 模型列表
    ResNet_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "ResNet_ope"]
    VGG_list = ["VGG11", "VGG13", "VGG16", "VGG19"]

    if input_network == "AlexNet":
        Model, input = model_AlexNet()

    elif input_network == "MobileNet":
        Model, input = model_MobileNet()

    elif input_network == "ResNet":
        # if input_network not in ResNet_list:
        #     input_network = "ResNet18"
        default_input_network = "ResNet18"
        Model, input = model_ResNet(default_input_network)
    elif input_network == "ResNet_ope":
        input = torch.randn(1, 3, 32, 32)
        if compress_ope == "svd":
            Model = resnet18svd()
            print("svd")
        elif compress_ope == "dpconv":
            Model = resnet18dpconv()
        elif compress_ope == "fire":
            Model = resnet18fire()
        elif compress_ope == "inception1":
            Model = resnet18inception1()
        elif compress_ope == "inception2":
            Model = resnet18inception2()
        else:
            print("error compress_ope")
            Model = resnet18svd()
    elif input_network == "VGG":
        # if input_network not in VGG_list:
        #     input_network = "VGG11"
        default_input_network = "VGG11"
        Model, input = model_VGG(default_input_network)

    else:
        print("network error in computation")

    # torch.save(Model, "./model/" + str(input_network) + ".pth")
    # print("Saving model successfully!")

    return Model, input

def HMT_energy(cmd='Predefined_model', network='VGG', compress_ope="null"):
        # 获得模型
    if cmd == 'Predefined_model':
        # 系统预存模型 V1.0
        Energy = model_energy_predefined(network, compress_ope)
    elif cmd == 'User_defined_model':
        # 需要用户上传模型代码 后续补充
        Energy = model_energy_user()
    else:
        print("error in energy!")

    return Energy

def model_energy_user():
    pass

def model_energy_predefined(network, compress_ope="null"):
    
    Model, input = CreateModel(network, compress_ope)

    # print("Model: ", Model)
    # print("input: ", )

    # 计算 Cl：计算量
    Macs, Params, Model_list = profile(Model, inputs=(input, ))
    # 获得Cl
    Cl = Macs

    # 计算 Ml：访问量
        # 对于每一层：
                    # 输入大小 x 字节
                    # 权重大小 x 字节
                    # 输出大小 x 字节
            # 内存访问量：（输入张量大小 + 输出张量大小 + 权重大小）x 数据类型字节数
        # 计算每一层，求和

    # 1. 获得每一层的名称
    net_list = {'input': input.shape}

    for key_i in Model_list.keys():
        net_list.setdefault(str(key_i), {})

    # 2. 获得每一层的weight和bias大小

    for name, param in Model.named_parameters():
        # print(name, param.shape)
        layer_name = name.split(".")[0]
        layer_name_para = name.split(".")[1]
        if layer_name in net_list:
            net_list[layer_name][layer_name_para] = param.shape



    # print("net_list: ", net_list)

    # input = torch.randn(2, 3, 32, 32)
    # 获得输入
    num_sample = net_list["input"][0]
    C1 = net_list["input"][1]
    W1 = net_list["input"][2]
    H1 = net_list["input"][3]
    input_size = W1 * H1 * C1

    # 初始化参数
    input_size_totle = 0
    output_size_totle = 0
    weight_size_totle = 0

    # 定义字节
    byte_size_float64 = 8
    byte_size_float32 = 4

    # 定义单元能耗
    # 单位 pJ
    energy_access = 100     # 内存访问: 100 pJ
    energy_access_gpu = 0.05 * 10 ** 3   # GPU访存：0.05 mJ = 0.5 * 10 ** 9 pJ
    energy_access_cache = 0.05      # 缓存访问：0.05 pJ
    energy_mutpily_cpu = 5 * 10 ** 3    # 乘法操作：5 mJ = 5 * 10 ** 9 pJ
    cache_rate = 0.5        # 初始化命中率：50%

    for name, layer in Model.named_modules():
        # 卷积层
        if isinstance(layer, nn.Conv2d):
            # 获取卷积核数量、输入大小、步长和填充
            out_channels = layer.out_channels
            in_channels = layer.in_channels
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding

            # 获得参数
            K = kernel_size[0]
            P = padding[0]
            S = stride[0]
            C2 = out_channels
            # 计算输出大小
            W2 = (W1 - K + 2 * P) / S + 1
            H2 = (H1 - K + 2 * P) / S + 1
            output_size = W2 * H2 * C2

            # 考虑偏置
            if layer.bias is not None:
                # 该层包含偏置参数
                # K * K * C1 * C2 + C2 = (K * K * C1 + 1)* C2
                weight_size = (K * K * C1 + 1) * C2
            else:
                # 该层不包含偏置参数
                weight_size = K * K * C1 * C2

            # 累加大小
            weight_size_totle += weight_size
            input_size_totle += input_size
            output_size_totle += output_size

            # 更新输入大小和长、宽
            input_size = output_size
            C1 = C2
            W1 = W2
            H1 = H2

        # 全连接层
        elif isinstance(layer, nn.Linear):
            # 获得输入输出大小
            output_features = layer.out_features
            input_features = layer.in_features

            # 计算输出大小
            output_size = W1 * H1 * output_features

            # 考虑偏置
            if layer.bias is not None:
                weight_size = input_size * output_size + output_size
            else:
                weight_size = input_size * output_size
            # 累加大小

            input_size_totle += input_size
            output_size_totle += output_size
            weight_size_totle += weight_size

            # 更新输入大小、通道数
            input_size = output_size
            C1 = output_features

        # 其他层，没有权重
        else:
            # 输出层信息
            # print("name: ", name, "\tlayer: ", layer)
            pass

    # 计算：内存访问量 = ( 输入张量大小 + 输出张量大小 + 权重大小 ) × 数据类型字节数 × 每次样本输入数量
    mem_access = (input_size_totle + output_size_totle + weight_size_totle) * byte_size_float32 * num_sample
    # print("Totle Memory Access: ", mem_access)
    # Cl已经获得，获得Ml
    Ml = mem_access

    # 创建一个设备对象
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        is_GPU = 1
    else:
        is_GPU = 0

    # 获得cache命中率
    cache_rate_get = getcacherate(Model, input, device)
    if cache_rate_get < 1:
        cache_rate = cache_rate_get
    else:
        # print("cache_rate_get: ", cache_rate_get)
        pass

    # 总能耗
    
    # print("Cl: ", Cl)
    # print("Ml: ", Ml)
    energy_total = energy_mutpily_cpu * Cl + cache_rate * energy_access_cache * Ml + (1-cache_rate) * energy_access + is_GPU * Ml * energy_access_gpu
    # energy_total = energy_mutpily_cpu * Cl + cache_rate * energy_access_cache * Ml + (1-cache_rate) * energy_access
    # energy_total = round(energy_total * 10 ** (-12) , 2)
    energy_total = energy_total * 10 ** (-12 + 3)
    
    # print("energy_total: ", energy_total)

    return energy_total

def getcacherate(model, input, device):

    network = model.to(device)

    input_tensor = input.to(device)

    # 在 GPU 上运行网络并打印输出
    with torch.no_grad():
        output = network(input_tensor)
        # print(output.shape)

    # 测试模型运行时间
    for i in range(100):
        
        time_taken = measure_model_time(network, input_tensor, device)
        # print('Model took {:.6f} seconds to run on device {}'.format(time_taken * 1000, device))
        if i == 0:
            time1 = time_taken
        elif i == 1:
            time2 = time_taken

    rate_cache_1 = 1 - (time1 - time2) / time1

    # print("time1: ", time1)
    # print("time2: ", time2)
    # print("rate_cache_1: ", rate_cache_1 * 100, "%")
    return rate_cache_1

def measure_model_time(model, input_tensor, device):
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()

    return end_time - start_time

if __name__ == "__main__":
    
    HMT_energy()
