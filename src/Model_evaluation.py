# @FunctionName: Model evaluation
# @Author: Wangyuzhan
# @Time: 2022/12/19

import argparse
from msilib.schema import Class
from operator import truediv

# my function
from computation import HMT_computation
from latency import HMT_latency

class EvaluationValue:
    """ 存储评估参数的结果 """
    def __init__(self):
        self.reset()

    def reset(self):
        # 初始化参数值
        # Computation', 'Parameter', 'Storage', 'Latency', 'Energy', 'Accuracy'
        self.Macs = 0
        self.Params = 0
        self.Storage = 0
        self.Latency = 0
        self.Energy = 0
        self.Accuracy = 0

def parse_args():

    # 计算量参数量存储量、时延、能耗、精度
    parser = argparse.ArgumentParser(
        description='CrowdHMT Model Evaluation')
    # 选择预定于模型或者用户模型
    # 1. 预定义模型则选择系统提供的模型
    # 2. 用户上传模型, 需要按照模型定义规则完成模型并上传, 进行评估
    parser.add_argument('cmd', choices=['Predefined_model', 'User_defined_model'],
                        default='Predefined_model')
    
    # 系统与定义模型的参数

    parser.add_argument('--network', type=str, default='VGG',
                        choices=['AlexNet', 'MobileNet', 'ResNet', 'VGG'],
                        help='Select the model to evaluate')

    # parser.add_argument('--performance', type=str, default='Computation',
    #                     choices=['Computation', 'Parameter', 'Storage', 'Latency', 'Energy', 'Accuracy'],
    #                     help='Select the model performance to evaluate')

    # 自定义模型的参数: 网络名称, 文件名称

    parser.add_argument('--User_network', type=str, default='none',
                        help='Please input your network')

    parser.add_argument('--User_netfilename', type=str, default='none',
                        help='Please input your network file name')

    # 评估的性能超参数

    parser.add_argument('--Computation', type=str, default=False, 
                        choices=['True', 'False'], help='Evaluate the model Computation')

    parser.add_argument('--Latency', type=str, default=False, help='Evaluate the model Latency')

    parser.add_argument('--Energy', type=str, default=False, help='Evaluate the model Energy')

    parser.add_argument('--Accuracy', type=str, default=False, help='Evaluate the model Accuracy')

    args = parser.parse_args()
    return args

def ResultTransmission(Value_e: EvaluationValue): 
    # Show the result
    print("*" * 30)
    print("Macs: ", Value_e.Macs)
    print("Params: ", Value_e.Params)
    print("Storage:  %.2f" %float(Value_e.Storage) + "MB")
    print("*" * 30)

def main():

    # 超参数
    args = parse_args()
    # 
    Value_e = EvaluationValue()

    # 选择标准模型，即系统预存模型，AlexNet, MobileNet, ResNet, VGG
    if args.cmd == 'Predefined_model':
        if args.network:
            # 获得网络名称信息
            input_net = args.network
            # 将参数初始化
            Value_e.reset()
            # print("Net is :", input_net)
            if args.Computation == "True":
                # 模型计算量、参数量、存储量
                # print("Value_e: ", Value_e.Accuracy)
                # Value_e.Accuracy = 1
                Value_e.Macs, Value_e.Params, Value_e.Storage = HMT_computation(cmd=args.cmd, network=input_net)
            if args.Latency:
                # 
                pass
            if args.Energy:
                pass
            if args.Accuracy:
                pass
        pass

    # 选择自定义模型，需要上传模型文件，规定格式
    elif args.cmd == 'User_defined_model':
        # 测试网络结构是否正确
        # 需要网络名称, 代码文件名称
        Value_e.reset()
        # print("Net is :", input_net)
        if args.Computation == "True":
            # 模型计算量、参数量、存储量
            # print("Value_e: ", Value_e.Accuracy)
            # Value_e.Accuracy = 1
            Value_e.Macs, Value_e.Params, Value_e.Storage = HMT_computation(cmd=args.cmd)
        if args.Latency:
            # 
            pass
        if args.Energy:
            pass
        if args.Accuracy:
            pass

    # 错误选择，报错
    else:
        pass

    # 发送数据到前端展示
    ResultTransmission(Value_e)

if __name__ == "__main__":

    main()
    