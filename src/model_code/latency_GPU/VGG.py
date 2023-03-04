'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

import time

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def test():
    
    net = VGG("VGG11")
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()

def model_VGG(input_network = "VGG11"):

    VGG_list = ["VGG11", "VGG13", "VGG16", "VGG19"]

    if input_network in VGG_list:
        model = VGG(input_network)
    else:
        model = VGG("VGG11")

    input = torch.randn(2, 3, 32, 32)

    return model, input

def test_runtime():
    import time
    model, input = model_VGG()
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
    network = VGG("VGG11").to(device)

    # 创建一个随机的输入图像张量
    input_tensor = torch.randn(2, 3, 32, 32).to(device)

    # 在 GPU 上运行网络并打印输出
    with torch.no_grad():
        output = network(input_tensor)
        # print(output.shape)
    
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

    torch.cuda.empty_cache()