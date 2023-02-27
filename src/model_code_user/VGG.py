'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

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

def model_user(input_network = "VGG11"):

    VGG_list = ["VGG11", "VGG13", "VGG16", "VGG19"]

    if input_network in VGG_list:
        model = VGG(input_network)
    else:
        model = VGG("VGG11")

    input = torch.randn(2, 3, 32, 32)

    return model, input

def test_runtime():
    import time
    model, input = model_user()
    starttime = time.time()
    for i in range(100):
        out = model(input)
    endtime = time.time()
    execution_t = (endtime - starttime) / 100
    print("execution_t: ", execution_t)

# test_runtime()