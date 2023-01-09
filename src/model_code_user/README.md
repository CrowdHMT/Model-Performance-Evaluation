# 自定义网络结构

## 一、头文件

目前仅支持Pytorch架构的网络模型，需要添加以下头文件：pytorch环境

``` python
import torch
import torch.nn as nn
```
**注意：除此之外，符合Pytorch官方函数库的网络定义均可以使用，例如：**
``` python
import torch.nn.functional as F
```

## 二、网络定义

符合Pytorch的网络结构定义均可作为自定义网络结构上传来进行模型性能评估

例如：AlexNet定义为Model_user, 其文件名为
``` python
class Model_user(nn.Module):
    def __init__(self):
        super(Model_user, self).__init__()
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
```

## 三、测试函数

首先需要用户验证函数定义是否正确，以Model_user为例：
``` python
def test():
    net = Model_user()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
```

输出应该和模型定义输出大小相等：
``` bash
torch.Size([2, 10])
```

## 四、模型返回函数

用于网络性能评估，以AlexNet网络为例，需要定义模型、输入张量以及模型的名称：
``` python
# 注意定义返回函数时，需要定义为"model_User"
def model_user():
    # 定义模型和输入
    model = AlexNet()
    input = torch.randn(2, 3, 32, 32)
    # 返回模型和输入
    return model, input
```

**注意：在模型定义时，需要先使用test()函数通过函数测试，再定义model_user()超参数函数**

## 五、使用方法

主函数接口为"User_defined_model"和"--Computation True"：
``` python
python Model_evaluation.py User_defined_model --Computation True
```
