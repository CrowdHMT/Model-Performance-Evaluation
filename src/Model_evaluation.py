# author wyz
# date 2022.12.19

import argparse

# my function
from computation import HMT_computation

def parse_args():

    parser = argparse.ArgumentParser(
        description='CrowdHMT Model Evaluation')
    # 计算量参数量存储量、时延、能耗、精度
    parser.add_argument('cmd', choices=['Predefined_model', 'User_defined_model'],
                        default='Predefined_model')

    parser.add_argument('--network', type=str, default='VGG',
                        choices=['AlexNet', 'MobileNet', 'ResNet', 'VGG'],
                        help='Select the model to evaluate')

    parser.add_argument('--performance', type=str, default='Computation',
                        choices=['Computation', 'Parameter', 'Storage', 'Latency', 'Energy', 'Accuracy'],
                        help='Select the model performance to evaluate')

    args = parser.parse_args()
    return args

def main():

    # 超参数
    args = parse_args()

    # 选择标准模型，即系统预存模型，AlexNet, MobileNet, ResNet, VGG
    if args.cmd == 'Predefined_model':
        if args.network:
            input_net = args.network
            # print("Net is :", input_net)
            HMT_computation(cmd=args.cmd, network=input_net)
        pass

    # 选择自定义模型，需要上传模型文件，规定格式
    elif args.cmd == 'User_defined_model':
        pass

    # 错误选择，报错
    else:
        pass

if __name__ == "__main__":

    main()
    