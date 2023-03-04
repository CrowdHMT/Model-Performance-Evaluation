'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():

    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()

def model_MobileNet():

    model = MobileNet()
    input = torch.randn(2, 3, 32, 32)

    return model, input

# model_MobileNet()

def test_runtime():
    import time
    model, input = model_MobileNet()
    starttime = time.time()
    for i in range(20):
        out = model(input)
    endtime = time.time()
    execution_t = (endtime - starttime) / 20
    print("execution_t: ", execution_t)

def measure_model_time(model, input_tensor):
    model.eval()
    input_tensor = input_tensor
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()

    return end_time - start_time



if __name__ == '__main__':

    # 创建一个设备对象
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化自定义的 Mobilenet 网络
    network = MobileNet()

    # 创建一个随机的输入图像张量
    input_tensor = torch.randn(2, 3, 32, 32)

    # 在 GPU 上运行网络并打印输出
    with torch.no_grad():
        output = network(input_tensor)
        # print(output.shape)
    
    time_list = []

    time_in_cache = 0

    # 测试模型运行时间
    for i in range(100):
        
        time_taken = measure_model_time(network, input_tensor)
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