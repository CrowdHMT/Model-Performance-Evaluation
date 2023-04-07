"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class SVD(nn.Module):
    def __init__(self, in_planes, beta, kernel_1_out):
        super(SVD, self).__init__()
        # 1x1 conv -> 3x3 conv branch
        self.kernel_3_in = int(in_planes * beta)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, self.kernel_3_in, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(self.kernel_3_in, kernel_1_out, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(kernel_1_out),
            nn.ReLU(True),
        )

    def forward(self, x):
        y = self.b1(x)
        return y

def svd_block(in_list, out_list, pooling_k, pooling_s):
    layers = [SVD(in_list[i], 0.25, out_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


class ResNet18SVD(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, divided_by=1):
        super(ResNet18SVD, self).__init__()

        self.in_planes = 64//divided_by

        self.conv1 = nn.Conv2d(3, 64//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//divided_by)
        self.layer1 = self._make_layer(block, 64//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        self.layer3 = svd_block([128, 256, 256], [256, 256, 256], 2, 2)
        self.layer4 = svd_block([256, 512, 512], [512, 512, 512], 1, 1)
        # add
        # self.layer5 = svd_block([512, 512, 512], [512, 512, 512], 1, 1)
        # add
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())
        out = self.conv1(x)
        # print(out.size())
        out = self.bn1(out)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        
        # out = self.layer5(out)
        
        # print(out.size())
        out = self.avg_pool(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.linear(out)
        # print(out.size())
        return out

class Fire(nn.Module):
    def __init__(self, in_planes, beta, out):
        super(Fire, self).__init__()
        self.kernel_in = int(in_planes * beta)
        self.branch = out // 2
        self.squeeze = nn.Conv2d(in_planes, self.kernel_in, kernel_size=1)
        self.expand1 = nn.Conv2d(self.kernel_in, self.branch, kernel_size=1)
        self.expand2 = nn.Conv2d(self.kernel_in, self.branch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.squeeze(x)
        y1 = self.expand1(x)
        y2 = self.expand2(x)
        y = torch.cat((y1, y2), dim=1)
        y = self.bn(y)
        y = self.relu(y)
        return y

def fire_block(in_list, out_list, pooling_k, pooling_s):
    layers = [Fire(in_list[i], 0.25, out_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)

class ResNet18Fire(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, divided_by=1):
        super(ResNet18Fire, self).__init__()

        self.in_planes = 64//divided_by

        self.conv1 = nn.Conv2d(3, 64//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//divided_by)
        self.layer1 = self._make_layer(block, 64//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        
        self.layer3 = fire_block([128, 256, 256], [256, 256, 256], 2, 2)
        self.layer4 = fire_block([256, 512, 512], [512, 512, 512], 1, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())
        out = self.conv1(x)
        # print(out.size())
        out = self.bn1(out)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = self.avg_pool(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.linear(out)
        # print(out.size())
        return out

def deep_conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_in, kernel_size=k_size, groups=chann_in, padding=p_size),
        nn.Conv2d(chann_in, chann_out, kernel_size=1),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

class ResNet18DpConv(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, divided_by=1):
        super(ResNet18DpConv, self).__init__()

        self.in_planes = 64//divided_by

        self.conv1 = nn.Conv2d(3, 64//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//divided_by)
        self.layer1 = self._make_layer(block, 64//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        
        self.layer3 = deep_conv_layer(128, 256, 3, 1)
        self.layer4 = deep_conv_layer(256, 512, 2, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Inception1(nn.Module):
    def __init__(self, in_planes, beta, kernel_1_out):
        super(Inception1, self).__init__()
        # 1x1 conv -> 3x3 conv branch
        self.kernel_3_in = int(in_planes * beta)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, self.kernel_3_in, kernel_size=1),
            nn.Conv2d(self.kernel_3_in, self.kernel_3_in, kernel_size=3, padding=1),
            nn.Conv2d(self.kernel_3_in, kernel_1_out, kernel_size=1),
            nn.BatchNorm2d(kernel_1_out),
            nn.ReLU(True),
        )

    def forward(self, x):
        y = self.b1(x)
        return y

def inception1_block(in_list, out_list, pooling_k, pooling_s):
    layers = [Inception1(in_list[i], 0.25, out_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)

class ResNet18Inception1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, divided_by=1):
        super(ResNet18Inception1, self).__init__()

        self.in_planes = 64//divided_by

        self.conv1 = nn.Conv2d(3, 64//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//divided_by)
        self.layer1 = self._make_layer(block, 64//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        
        self.layer3 = inception1_block([128, 256, 256], [256, 256, 256], 2, 2)
        self.layer4 = inception1_block([256, 512, 512], [512, 512, 512], 1, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Inception2(nn.Module):
    def __init__(self, in_planes, out):
        super(Inception2, self).__init__()
        self.branch = out // 2
        self.son1 = nn.Conv2d(in_planes, self.branch, kernel_size=(1, 3), padding=(0, 1))
        self.son2 = nn.Conv2d(in_planes, self.branch, kernel_size=(3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        y1 = self.son1(x)
        y2 = self.son2(x)
        y = torch.cat((y1, y2), dim=1)
        y = self.bn(y)
        y = self.relu(y)
        return y

def inception2_block(in_list, out_list, pooling_k, pooling_s):
    layers = [Inception2(in_list[i], out_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)

class ResNet18Inception2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, divided_by=1):
        super(ResNet18Inception2, self).__init__()

        self.in_planes = 64//divided_by

        self.conv1 = nn.Conv2d(3, 64//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//divided_by)
        self.layer1 = self._make_layer(block, 64//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        
        self.layer3 = inception2_block([128, 256, 256], [256, 256, 256], 2, 2)
        self.layer4 = inception2_block([256, 512, 512], [512, 512, 512], 1, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet18inception2():
    """ return a ResNet 18 Deep inception2 object
    """
    return ResNet18Inception2(BasicBlock, [2, 2, 2, 2])

def resnet18inception1():
    """ return a ResNet 18 inception1 object
    """
    return ResNet18Inception1(BasicBlock, [2, 2, 2, 2])

def resnet18dpconv():
    """ return a ResNet 18 Deep Conv object
    """
    return ResNet18DpConv(BasicBlock, [2, 2, 2, 2])

def resnet18fire():
    """ return a ResNet 18 FIRE object
    """
    return ResNet18Fire(BasicBlock, [2, 2, 2, 2])

def resnet18svd():
    """ return a ResNet 18 SVD object
    """
    return ResNet18SVD(BasicBlock, [2, 2, 2, 2])

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])




