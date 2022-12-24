# Model-Performance-Evaluation
Users input the deep learning model and device resources to evaluate the performance of the deep learning model. The measurement includes precision, energy consumption, time delay, computation and storage capacity.

# 模型性能评估

**简介：**

参数量是指模型含有多少参数，直接决定模型文件的大小，也影响模型推断时对内存的占用量；
存储量是根据模型的参数，存储在设备上模型的占用大小，；
计算量是指模型推断时需要多少计算次数，通常是以***MAC(Multiply ACcumulate，乘积累加)***次数来表示；
这三者其实是评估模型时非常重要的参数，一个实际要应用的模型不应当仅仅考虑它在准确率上有多出色的表现，还应该要考虑它的鲁棒性、扩展性以及对资源的依赖程度，但事实上很多论文都不讨论他们模型需要多少计算力，一种可能是他们的定位还是纯学术研究——提出一种新的思路，即使这种思路不便于应用，但未来说不定计算力上来了，或者有什么飞跃性的改进方法来改进这一问题，或者提出自己的思路来启发其他研究者的研究（抛砖引玉）。

**等待完善功能：**
时延、能耗、精度

**主要功能：**

该功能可统计模型的参数量，并且能根据用户给定输入数据的格式计算模型推理过程中的计算量，为深度学习模型在边端设备上的部署方式提供参考。

**参数说明：**
| 参数 | 说明 | 具体参数 |
| --- | --- | --- |
| 必填参数 | 用户自定义模型或者系统与定义模型 | `Predefined_model` |
| | | `User_defined_model` |
| 可选参数 | 网络类型, 评估参数 | `--network` |
| | | `--performance` |
| `network` | 网络类型 | `AlexNet, MobileNet, ResNet, VGG` |
| `performance` | 评估参数 | `Computation, Parameter, Storage, Latency, Energy, Accuracy` |

**使用方法：**
1. 进入目录
`cd src`
2. 执行评估文件
python Model_evaluation.py Predefined_model --network AlexNet


**注意事项：**
目前处于开发阶段，评估参数仅提供`Computation, Parameter, Storage`，剩余功能`Latency, Energy, Accuracy`等待上线