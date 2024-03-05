
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



