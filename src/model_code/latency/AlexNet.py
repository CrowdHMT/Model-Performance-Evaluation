import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.Conv2d_1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=96, padding=1)
        self.bn_1 = nn.BatchNorm2d(96)
        self.maxpool_1 = nn.MaxPool2d((3, 3), stride=2, padding=1)

        self.Conv2d_2 = nn.Conv2d(kernel_size=5, in_channels=96, out_channels=256, padding=2)
        self.bn_2 = nn.BatchNorm2d(256)
        self.maxpool_2 = nn.MaxPool2d((3, 3), stride=2, padding=1)

        self.Conv2d_3 = nn.Conv2d(kernel_size=3, in_channels=256, out_channels=384, padding=1)
        self.Conv2d_4 = nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, padding=1)
        self.Conv2d_5 = nn.Conv2d(kernel_size=3, in_channels=384, out_channels=256, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.maxpool_3 = nn.MaxPool2d((3, 3), stride=2, padding=1)

        self.fc_1 = nn.Linear(4*4*256, 2048)
        self.dp_1 = nn.Dropout()
        self.fc_2 = nn.Linear(2048, 1024)
        self.dp_2 = nn.Dropout()
        self.fc_3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)

        x = self.Conv2d_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)

        x = F.relu(self.Conv2d_3(x))
        x = F.relu(self.Conv2d_4(x))
        x = F.relu(self.Conv2d_5(x))
        x = self.bn_3(x)
        x = F.relu(x)
        x = self.maxpool_3(x)

        x = x.view(-1, 4*4*256)
        x = F.relu(self.fc_1(x))
        x = self.dp_1(x)
        x = F.relu(self.fc_2(x))
        x = self.dp_2(x)
        x = self.fc_3(x)
        return x

def test():

    net = AlexNet()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()

def model_AlexNet():

    model = AlexNet().to(device)
    input = torch.randn(2, 3, 32, 32).to(device)

    return model, input

def test_runtime():
    import time
    model, input = model_AlexNet()
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
    network = AlexNet().to(device)

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