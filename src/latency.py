# @FunctionName: Model evaluation latency
# @Author: Wangyuzhan
# @Time: 2022/12/25

from thop.vision.basic_hooks import *
from thop.rnn_hooks import *

# 预定义模型
from model_code.AlexNet import model_AlexNet
from model_code.MobileNet import model_MobileNet
from model_code.ResNet import model_ResNet
from model_code.VGG import model_VGG


register_ops = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,
    nn.BatchNorm1d: count_normalization,
    nn.BatchNorm2d: count_normalization,
    nn.BatchNorm3d: count_normalization,
    nn.LayerNorm: count_normalization,
    nn.InstanceNorm1d: count_normalization,
    nn.InstanceNorm2d: count_normalization,
    nn.InstanceNorm3d: count_normalization,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,
    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,
    nn.RNNCell: count_rnn_cell,
    nn.GRUCell: count_gru_cell,
    nn.LSTMCell: count_lstm_cell,
    nn.RNN: count_rnn,
    nn.GRU: count_gru,
    nn.LSTM: count_lstm,
    nn.Sequential: zero_ops,
    nn.PixelShuffle: zero_ops,
}


def HMT_latency(model: nn.Module, inputs):

    prev_training_status = model.training
    model.eval()

    def dfs_lantency(module: nn.Module):
        total_latency = 0
        ret_dict = {}
        for n, m in module.named_children():
            next_dict = {}
            if type(m) in register_ops:
                ###################
                # 查找当前层的延迟 #
                ###################
                m_latency = 1
            else:
                # 不在注册类型内，跳过该层
                continue
            # 求总延迟
            total_latency += m_latency
            # 获得延迟和层对应关系
            ret_dict[n] = (total_latency)

        return total_latency, ret_dict

    total_latency, ret_dict = dfs_lantency(model)

    # reset model to original status
    model.train(prev_training_status)

    return total_latency, ret_dict


def test():

    Model, input = model_AlexNet()

    total_latency, ret_dict = HMT_latency(model=Model, inputs=input)

    print(HMT_latency(model=Model, inputs=input))

test()

# if __name__ == "__main__":

#     HMT_latency()
