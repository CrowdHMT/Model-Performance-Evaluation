## 1  输入输出

**问**

假设输入数据大小为 $W_1\times H_1\times C_1$，输出数据大小为 $W_2\times H_2\times C_2$，全连接层和卷积层的网络的权重大小分别是什么，如果有偏置，也列出来

**答**

对于全连接层，权重的大小为 $W_1\times H_1\times C_1 \times W_2\times H_2\times C_2$，偏置的大小为 $W_2\times H_2\times C_2$。其中 $W_1\times H_1\times C_1$ 表示输入数据的大小，$W_2\times H_2\times C_2$ 表示输出数据的大小。

对于卷积层，权重的大小为 $k_w\times k_h\times C_1\times C_2$，其中 $k_w$ 和 $k_h$ 分别表示卷积核的宽度和高度，$C_1$ 表示输入数据的通道数，$C_2$ 表示输出数据的通道数。如果有偏置，偏置的大小为 $C_2$。

需要注意的是，这里的权重大小和偏置大小只是给出了参数的数量，不考虑数据类型和存储方式等细节。在实际计算中，这些因素也会对模型的大小和存储空间产生影响。

## 2  单元能耗

**问**

深度学习任务，访存一次CPU，访存一次GPU，访存一次cache，使用CPU完成一次矩阵乘法，使用GPU完成一次矩阵乘法，分别需要多少能量，给出具体数值

**答1**

访问内存和进行算术运算的能耗会因不同的硬件、算法和数据集等因素而有所差异。以下是一些可能的能耗数据，供参考：

对于CPU，一次内存访问的能耗通常在10-200皮焦耳（pJ）之间，一次矩阵乘法的能耗通常在0.1-10毫焦耳（mJ）之间。
对于GPU，一次内存访问的能耗通常在0.01-1毫焦耳（mJ）之间，一次矩阵乘法的能耗通常在0.01-1毫焦耳（mJ）之间。
对于Cache，一次内存访问的能耗通常在0.01-0.1皮焦耳（pJ）之间。
需要注意的是，这些数据是估计值，实际的能耗会受到多种因素的影响，包括硬件和系统实现的具体细节、算法和数据集的特点、系统负载和温度等环境因素。

此外，需要指出的是，使用CPU和GPU完成矩阵乘法的能耗差异较大，因为GPU通常比CPU更适合进行并行计算，能够更有效地利用计算资源。但是，GPU在进行单个操作时的能耗可能会比CPU高，这是因为GPU具有更多的处理器和内存带宽，需要更多的能源来维护。

**答2**

CPU能耗：一些研究人员使用专门的电力仪器对各种CPU进行测试，以确定它们的能耗。例如，有一个名为SPECpower_ssj2008的基准测试套件，用于评估服务器处理器的能效。同时，一些CPU制造商会在其技术文档和官方网站上提供能耗数据，例如Intel的数据表中给出了各种CPU的TDP（Thermal Design Power）值，这是CPU在设计功耗范围内的最大功耗。

GPU能耗：GPU能耗的测试也可以使用专门的电力仪器进行，例如GPU-Z等软件可以显示GPU的功耗。此外，GPU制造商通常会在其官方文档和网站上公布能耗数据，例如NVIDIA的数据表中列出了各种GPU的TDP值。

Cache能耗：一些研究人员使用仿真工具对不同类型的Cache进行建模和测试，以确定它们的能耗。例如，有一个名为Cacti的开源工具，用于模拟各种Cache的能耗和性能。

深度学习任务的能耗：这些数据通常是通过在实际硬件上运行深度学习算法来收集的，例如在各种CPU、GPU和专用加速器上运行标准的深度学习基准测试（例如MNIST、CIFAR-10、ImageNet等）。同时，一些研究人员会使用模拟器和仿真工具来模拟深度学习任务的能耗，以便进行更深入的分析。

### 来源

些来源提供了处理器和内存能耗的测量、模拟或估算结果：

J. Cong等人在2003年发表的论文《High-level power estimation and optimization for DSM platforms》中给出了一些处理器运算的能耗估算数据。
A. González等人在2014年发表的论文《Modeling and analysis of memory power consumption under different memory mapping schemes》中给出了一些内存访问的能耗测量和仿真结果。
SPEC CPU2006基准测试中包含一些能耗测量结果，可以用于比较不同处理器的能耗。
EEMBC（嵌入式微控制器基准委员会）发布了一些嵌入式系统的基准测试，包括能耗测试，这些测试可以用于评估不同嵌入式系统的能耗表现。
一些处理器和内存的制造商，如英特尔、AMD、三星等，会在产品规格书中提供一些能耗数据和参数。如下：
英特尔：英特尔在其产品规格书中提供了许多处理器的功耗和热设计功耗（TDP）信息。此外，英特尔还提供了一个名为“英特尔功耗分析工具”（Intel Power Gadget）的免费软件，可用于实时监测英特尔处理器的功耗。
AMD：AMD在其产品规格书中提供了许多处理器的功耗和热设计功耗（TDP）信息。此外，AMD还提供了一个名为“AMD峰值功耗计算器”（AMD TDP Calculator）的在线工具，可用于估算特定处理器的峰值功耗。
三星：三星在其DRAM和NAND闪存的技术文档中提供了许多能耗信息，例如读取和写入的功耗、自刷新功耗等。
Hynix：Hynix在其DRAM和NAND闪存的技术文档中提供了许多能耗信息，例如读取和写入的功耗、自刷新功耗等。
Micron：Micron在其DRAM和NAND闪存的技术文档中提供了许多能耗信息，例如读取和写入的功耗、自刷新功耗等
以下是一些与CPU、GPU和深度学习任务能耗有关的论文和文档的链接：

"The SPECpower_ssj2008 benchmark: A community effort to develop a standard benchmark for power and performance characterization of server-class computers" (https://ieeexplore.ieee.org/document/4476932)

"Intel 11th Gen Intel Core Desktop Processors Datasheet, Vol. 1" (https://www.intel.com/content/dam/products/hero/foreground/badge-11th-gen-core-i9-desktop-processor-families.jpg/_jcr_content/renditions/heroImg.desktop.hero-large.jpg)

"NVIDIA Ampere Architecture Whitepaper" (https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)

"CACTI 3.0: An Integrated Cache Timing, Power, and Area Model" (https://ieeexplore.ieee.org/document/7053197)

"Measuring Deep Learning Inference Power Consumption on Mobile Devices" (https://arxiv.org/abs/1711.07240)

"Benchmarking State-of-the-Art Deep Learning Software Tools" (https://arxiv.org/abs/2007.04846)

"Efficient Processing of Deep Neural Networks: A Tutorial and Survey" (https://ieeexplore.ieee.org/document/8589241)

这些论文和文档提供了一些有关CPU、GPU和深度学习任务能耗的详细信息和数据。

# next

假设有一个包含两个卷积层和一个全连接层的神经网络，输入数据为 $32\times 32$ 的 RGB 图像（即输入张量的大小为 $32\times 32\times 3$），输出为 10 个类别的概率分布。

假设第一个卷积层使用 $3\times 3$ 的卷积核，输出通道数为 32，第二个卷积层使用 $3\times 3$ 的卷积核，输出通道数为 64，全连接层的输出节点数为 10。这个神经网络使用 float32 数据类型。

计算每一层的内存访问量：

第一个卷积层的输入大小为 $32\times 32\times 3$，输出大小为 $30\times 30\times 32$。输入和输出张量占用的字节数分别为：
$$\text{输入张量占用字节数} = 32\times 32\times 3\times 4 = 12,288\text{ B}$$

$$\text{输出张量占用字节数} = 30\times 30\times 32\times 4 = 28,800\text{ B}$$

第一个卷积层的权重大小为 $3\times 3\times 3\times 32$，占用的字节数为：
$$\text{权重张量占用字节数} = 3\times 3\times 3\times 32\times 4 = 2,304\text{ B}$$

第二个卷积层的输入大小为 $14\times 14\times 32$，输出大小为 $12\times 12\times 64$。输入和输出张量占用的字节数分别为：
$$\text{输入张量占用字节数} = 14\times 14\times 32\times 4 = 7,168\text{ B}$$

$$\text{输出张量占用字节数} = 12\times 12\times 64\times 4 = 29,491\text{ B}$$

第二个卷积层的权重大小为 $3\times 3\times 32\times 64$，占用的字节数为：
$$\text{权重张量占用字节数} = 3\times 3\times 32\times 64\times 4 = 55,296\text{ B}$$

全连接层的输入大小为 10，输出大小为 10。输入和输出张量占用的字节数分别为：
$$\text{输入张量占用字节数} = 10\times 4 = 40\text{ B}$$

$$\text{输出张量占用字节数} = 10\times 4 = 40\text{ B}$$

因此，每一层的内存访问量为：

第一个卷积层的内存访问量为：
$$\text{输入张量占用字节数} + \text{输出张量占用字节数} + \text{权重张量占用字节数} = 12,288\text{ B} + 28,800\text{ B} + 2,304\text{ B} = 43,392\text{ B}$$

第二个卷积层的内存访问量为：
$$\text{输入张量占用字节数} + \text{输出张量占用字节数} + \text{权重张量占用字节数} = 7,168\text{ B} + 29,491\text{ B} + 55,296\text{ B} = 91,955\text{ B}$$

全连接层的内存访问量为：
$$\text{输入张量占用字节数} + \text{输出张量占用字节数} = 40\text{ B} + 40\text{ B} = 80\text{ B}$$

因此，整个神经网络的内存访问量为 $43,392\text{ B} + 91,955\text{ B} + 80\text{ B} = 135,427\text{ B}$。

这里的计算公式可以总结为：

$$\text{内存访问量} = (\text{输入张量大小} + \text{输出张量大小} + \text{权重大小}) \times \text{数据类型字节数}$$

其中，输入张量、输出张量和权重的大小指的是它们占用的内存大小，数据类型字节数指的是输入张量、输出张量、权重和偏置的数据类型所占用的字节数。

第一个卷积层的输入张量大小为 $32\times 32\times 3$，即 $32\times 32$ 的 RGB 彩色图像。该卷积层使用了 32 个大小为 $3\times 3$ 的卷积核，每个卷积核都有三个通道。因此，该卷积层的权重张量大小为 $3\times 3\times 3\times 32$。

具体来说，一个卷积核的大小为 $3\times 3\times 3$，其中 $3\times 3$ 表示卷积核的空间大小，即卷积核的宽和高均为 $3$，$3$ 表示卷积核的深度，即卷积核的通道数。由于该卷积层使用了 32 个卷积核，因此权重张量的大小为 $3\times 3\times 3\times 32$。

卷积层的输出大小取决于卷积核的数量、卷积核的大小、输入数据的大小、步长和填充方式等因素。假设输入数据大小为 $W_1\times H_1\times C_1$，卷积核大小为 $F\times F$，步长为 $S$，填充大小为 $P$，卷积核数量为 $K$，则卷积层的输出大小为 $W_2\times H_2\times C_2$，其中：

$W_2=\frac{W_1-F+2P}{S}+1$

$H_2=\frac{H_1-F+2P}{S}+1$

$C_2=K$

其中，$W_2$ 和 $H_2$ 表示输出矩阵的宽度和高度，$C_2$ 表示输出矩阵的通道数，也就是卷积核的数量。

以题目中的卷积层为例，输入数据的大小为 $96\times96\times1$，卷积核大小为 $5\times5$，卷积核数量为 $256$，则假设步长为 $1$，填充方式为 $0$，则输出大小为：

$W_2=\frac{96-5+2\times0}{1}+1=92$

$H_2=\frac{96-5+2\times0}{1}+1=92$

$C_2=256$

因此，该卷积层的输出大小为 $92\times92\times256$。

## 1

假设输入数据大小为 $W_1\times H_1\times C_1$，其中每个参数代表以下内容：

$W_1$：输入数据的宽度或水平方向上的大小。
$H_1$：输入数据的高度或垂直方向上的大小。
$C_1$：输入数据的通道数或深度，它表示输入数据的每个位置都有 $C_1$ 个值，即每个位置是一个 $C_1$ 维向量。
例如，在图像分类问题中，输入数据通常是一张图像，$W_1$ 表示图像的宽度，$H_1$ 表示图像的高度，$C_1$ 表示图像的通道数（例如，对于 RGB 彩色图像，$C_1=3$）。

在卷积神经网络中，每一层的输入都是一个 $W_1\times H_1\times C_1$ 的张量，其中 $W_1\times H_1$ 表示张量的空间维度，$C_1$ 表示张量的通道维度。

## 2

这个问题的答案取决于所使用的层的类型。不同类型的层可以有不同的权重参数形状和大小。

对于卷积神经网络中的卷积层，权重参数通常是由卷积核的形状和深度所决定的。假设使用的是 $k$ 个大小为 $F\times F\times C_1$ 的卷积核，其中 $F$ 是卷积核的大小（假设卷积核在宽度和高度上的大小相同），$C_1$ 是输入数据的通道数，则该层的权重大小为 $F\times F\times C_1\times k$。如果在卷积层中还包括偏置参数，则需要额外考虑每个卷积核对应一个偏置参数，因此总权重大小为 $(F\times F\times C_1 + 1)\times k$。

对于全连接层，权重参数的大小取决于输入和输出数据的大小。假设输入数据的大小为 $W_1\times H_1\times C_1$，输出数据的大小为 $W_2\times H_2\times C_2$，则该层的权重大小为 $W_1\times H_1\times C_1\times W_2\times H_2\times C_2$。如果在全连接层中还包括偏置参数，则需要额外考虑每个输出单元对应一个偏置参数，因此总权重大小为 $(W_1\times H_1\times C_1 + 1)\times W_2\times H_2\times C_2$。

## 3

需要注意的是，除了权重参数之外，神经网络中还有其他类型的参数，例如偏置参数、批量归一化参数等，这些参数的大小也需要考虑在内。

全连接层：权重形状为 $C_1\times C_2$，其中 $C_1$ 是输入数据的通道数，$C_2$ 是输出数据的通道数。因此，权重的大小是 $C_1\times C_2$。

卷积层：权重形状为 $K\times K\times C_1\times C_2$，其中 $K$ 是卷积核的大小，$C_1$ 是输入数据的通道数，$C_2$ 是输出数据的通道数。因此，权重的大小是 $K\times K\times C_1\times C_2$。
