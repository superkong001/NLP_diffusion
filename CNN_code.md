Keras框架一个简单的CNN架构示例，用于处理64x64像素的彩色图像（3个颜色通道），包含两个卷积层，每个卷积层后面跟着一个最大池化层，最后是两个全连接层进行分类（假设有10个类别）。

```Bash
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

卷积神经网络(CNN)详解与代码实现
> https://blog.csdn.net/shenyuan12/article/details/108200571

# step1 导入需要的包

```Bash
PyTorch中主要的包
  torch.nn ：包含用于构建神经网络的模块和可扩展类的子包。
  torch.autograd ：支持PyTorch中所有的可微张量运算的子包
  torch.nn.functional ：一种功能接口，包含用于构建神经网络的典型操作，如损失函数、激活函数和卷积运算
  torch.optim ：包含标准优化操作（如SGD和Adam）的子包。
  torch.utils ：工具包，包含数据集和数据加载程序等实用程序类的子包，使数据预处理更容易
  torchvision ：一个提供对流行数据集、模型架构和计算机视觉图像转换的访问的软件包

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision # pytorch的一个视觉处理工具包(需单独安装)
```

参考： pyTorch中tensor运算

> https://blog.csdn.net/weixin_43328816/article/details/124056831

# step2 数据预处理

## 将数据转换成tensor

PyTorch中对于数据集的处理有三个非常重要的类：Dataset、Dataloader、Sampler，均是 torch.utils.data 包下的模块（类）

Dataloader是数据的加载类，是对于Dataset和Sampler的进一步包装，用于实际读取数据

参考：PyTorch实现CNN、pytorch中的Variable——反向传播必备

> https://blog.csdn.net/hhr603894090/article/details/122094623
> https://blog.csdn.net/weixin_44912159/article/details/104800020



