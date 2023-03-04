'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():

    net = ResNet18()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()

def model_ResNet(input_network="ResNet18"):

    # ResNet_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

    if input_network == "ResNet18":
        model = ResNet18()
    elif input_network == "ResNet34":
        model = ResNet34()
    elif input_network == "ResNet50":
        model = ResNet50()
    elif input_network == "ResNet101":
        model = ResNet101()
    elif input_network == "ResNet152":
        model = ResNet152()
    else:
        model = ResNet18()

    input = torch.randn(1, 3, 32, 32)

    return model, input
# model_ResNet()

def test_runtime():
    import time
    model, input = model_ResNet()
    starttime = time.time()
    for i in range(20):
        out = model(input)
    endtime = time.time()
    execution_t = (endtime - starttime) / 20
    print("execution_t: ", execution_t)


def measure_model_time(model, input_tensor, device):
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()

    return end_time - start_time



if __name__ == '__main__':

    # 创建一个设备对象
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化自定义的 Mobilenet 网络
    network = ResNet18().to(device)

    # 创建一个随机的输入图像张量
    input_tensor = torch.randn(2, 3, 32, 32).to(device)

    # 在 GPU 上运行网络并打印输出
    with torch.no_grad():
        output = network(input_tensor)
        print(output.shape)
    
    time_list = []
    
    time_in_cache = 0

    # 测试模型运行时间
    for i in range(100):
        
        time_taken = measure_model_time(network, input_tensor, device)
        # print('Model took {:.6f} seconds to run on device {}'.format(time_taken * 1000, device))
        time_list.append(time_taken * 1000)
        if i == 0:
            time1 = time_taken
        elif i == 1:
            time2 = time_taken
        else:
            time_in_cache += time_taken


    time_inchache_avg = time_in_cache / 99

    rate_cache_all = 1 - (time1 - time_inchache_avg) / time1
    rate_cache_1 = 1 - (time1 - time2) / time1

    print("time1: ", time1)
    print("time2: ", time2)
    print("time_inchache_avg_all: ", time_inchache_avg)
    print("rate_cache_all: ", rate_cache_all * 100, "%")
    print("rate_cache_1: ", rate_cache_1 * 100, "%")
    print("Avg time", sum(time_list[1:]) / ( len(time_list) -1 ))