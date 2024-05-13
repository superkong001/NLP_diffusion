https://datawhaler.feishu.cn/wiki/LxSCw0EyRidru1kFkttc1jNQnnh

参考： 一文读懂各种神经网络层（ Pytorch ）
> https://mp.weixin.qq.com/s/wOEL6Hj2lZ_OJclmnGTnHw

<img width="416" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/265034ad-1c76-4966-8b95-724c51b5bb06">

参考：Transformers技术解析+实战(LLM)，多种Transformers diffusion模型技术图像生成技术+实战
> https://github.com/datawhalechina/sora-tutorial/blob/main/docs/chapter2/chapter2_2.md

# SelfAttention 

## 知识点

![84b043c179aaad94406f9182af81c47b_seq2seq](https://github.com/superkong001/NLP_diffusion/assets/37318654/1f1f31cf-c9e0-43e9-9361-20842ac6576e)

![205241478d09a2981deb7775da0ffb4b_seq2seq2](https://github.com/superkong001/NLP_diffusion/assets/37318654/d0b62426-47e4-4275-9915-ef5a9be5a961)

From: https://github.com/google/seq2seq

参考：https://zhuanlan.zhihu.com/p/106867810

加性Attention，如（Bahdanau attention）：

<img width="172" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/b686288b-d422-4206-98a3-0b32469c5ad7">

乘性Attention，如（Luong attention）：

<img width="309" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/e340d342-1fd1-491a-b315-7cb4f3bca70b">

"Attention is All You Need" 这篇论文提出了Multi-Head Self-Attention，是一种：Scaled Dot-Product Attention。

<img width="253" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/f8d058e2-a535-4d32-bf13-4388e50287f4">

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

<img width="362" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/e9a1725a-b997-418c-a45b-590827ed15c6">

<img width="326" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/7ec088b2-9e4e-49ac-b89f-8f6d032b8818">

<img width="623" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/bb0c06cf-bf25-4566-aa0f-e11078c4d9f7">

<img width="629" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/83dc3359-d7b1-4489-993c-0c77a7c77e3f">

<img width="491" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/0ca207d4-b104-4c0f-bcdc-a842faa9226d">

<img width="585" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/43a67ae9-62e8-415b-a1a6-464bcb71d6cd">

<img width="621" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/1b27e1fd-f820-424f-8952-36bbebdb6239">

<img width="637" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/7a191634-82fb-4d7e-9ba0-41a65220e12b">

参考： Vision Transformers的注意力层概念解释和代码实现
https://mp.weixin.qq.com/s/VkxU5Bm5x4TZZNYQw8Yzvg

[1] Vaswani et al (2017). Attention Is All You Need. https://doi.org/10.48550/arXiv.1706.03762

[2] Dosovitskiy et al (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. https://doi.org/10.48550/arXiv.2010.11929

[3] Yuan et al (2021). Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet. https://doi.org/10.48550/arXiv.2101.11986GitHub code: https://github.com/yitu-opensource/T2T-ViT

## 实践体验

```Bash
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MySelfAttention(nn.Module):
    # 定义了初始化方法，它接受一个config对象作为参数，这个对象包含了模型的配置信息
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 隐藏层的维度(hidden_dim)、Multi-Head的数量(num_heads)
        # 由于在自注意力中，隐藏层的维度需要被等分到每个头上，因此确保隐藏层的维度能被Multi-Head的数量整除。
        # num_heads为self-attention的头数
        assert config.hidden_dim % config.num_heads == 0

        # 创建了三个线性层，用于计算查询（Q）、键（K）和值（V）向量
        # torch.nn.Linear(in_features, # 输入的神经元个数(内容长度)i
        #            out_features, # 输出神经元个数o
        #            bias=True # 是否包含偏置b
        #            )
        # Y(n*o) =X(n*i)W(i*o)+b,n为batch_size
        self.wq = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.wk = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.wv = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # 定义了一个dropout层，用于在训练阶段按某种概率随即将输入的张量元素随机归零，防止网络过拟合
        self.att_dropout = nn.Dropout(config.dropout)

    # 定义了模块的前向传播方法forward,接收输入x和可选的mask参数。从输入x中提取query、key和value
    def forward(self, x, mask=None):
        # 从输入的embeding后数据提取批次大小(数量)、每批序列(内容)长度和每个内容的维度
        batch_size, seq_len, hidden_dim = x.shape

        # 通过线性层变换得到查询q、键k和值向量v
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # query、key 和 value reshape成(batch_size, seq_len, num_heads, head_dim)形状，以便进行多头注意力计算
        q = q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        k = k.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        v = v.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)

        # transpose(1, 2)交换第一维和第二维的位置，得到(batch_size, num_heads, seq_len, head_dim)形状
        # 键向量在点积前被转置，以保证矩阵乘法的维度一致性。
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算query和key点积，以得到注意力得分
        # (b, nh, ql, hd) @ (b, nh, hd, kl) => b, nh, ql, kl
        att = torch.matmul(q, k.transpose(2, 3))
        # 通过query和key的维度的平方根对注意力得分进行缩放。防止点积变得太大，影响梯度的稳定性。
        att /= math.sqrt(self.config.head_dim)

        # 如果提供了 mask，将注意力矩阵中mask对应位置的值置为负无穷
        if mask is not None:
            attn = att.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # 对注意力得分进行softmax归一化，以获取最终的注意力权重。
        # 表示输入序列中每个位置对其他位置的关注程度
        score = F.softmax(att.float(), dim=-1)
        # 进行dropout
        score = self.att_dropout(score)

        # 将注意力矩阵与value做矩阵乘法得到输出。
        # 然后将输出转置并reshape成(batch_size, seq_len, d_model) 形状。
        # 最后通过线性层w_o进行一次变换
        # (b, nh, ql, kl) @ (b, nh, kl, hd) => b, nh, ql, hd
        # 使用注意力权重对值向量进行加权求和，得到每个位置的加权表示
        attv = torch.matmul(score, v)
        # transpose(1, 2)交换第一维和第二维的位置，转换回(batch_size, seq_len, num_heads, head_dim)形状
        # contiguous()方法使得返回的张量在内存中连续存储。
        # 因为转置后的张量可能不是物理上连续的，而contiguous()会重新排列存储以使得每个元素在物理内存中紧密排列，以免引发错误或性能下降。
        attv = attv.transpose(1, 2).contiguous()
        # 将输出重塑为原始输入的形状
        attv = attv.view(batch_size, seq_len, -1)

        # 返回注意力权重和加权求和的结果
        return score, attv
```

```Bash
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from selfattention import SelfAttention

# 模型，只用一个核心的SelfAttention模块（可支持Single-Head或Multi-Head），来学习理解Attention机制。

class AttentionDemoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.attn = MySelfAttention(config)
        self.fc = nn.Linear(config.hidden_dim, config.num_labels)

    def forward(self, x):
        batch_size, seq_len = x.shape
        # x为batch_size, seq_len的张量
        h = self.emb(x)
        # 送入自注意力机制，得到自注意力得分 (attn_score) 和加权的隐藏状态 (h)
        attn_score, h = self.attn(h)
        # h [batch_size, seq_len, hidden_dim]：如[8 batch_size, 13 token, 512 hidden_dim]

        # 对隐藏状态进行平均池化, [8 batch_size, 13 token, 512 hidden_dim] -> [8 batch_size, 512 hidden_dim]
        # Applies a 1D average pooling over an input signal composed of several input planes.
        # 第一个参数是进行池化操作的输入张量，经过 permute 调整维度后的 h。
        # seq_len 是池化窗口的大小。例子中，池化窗口的大小被设置为序列的全长度，意味着对整个序列长度的特征进行平均。
        # 第三个参数 1 表示池化操作的步长。在这里，步长为 1 意味着池化窗口在移动时不会重叠
        # permute 方法用于调整张量的维度，以适应池化层的输入要求
        # 同时操作tensor的若干维度将tensor的维度换位
        # dims (tuple of int) – The desired ordering of dimensions
        # 从[batch_size, seq_len, hidden_dim] 转换为 [batch_size, hidden_dim, seq_len]
        h = F.avg_pool1d(h.permute(0, 2, 1), seq_len, 1)

        # squeeze 方法去除多余的维度,去除最后一个维度
        # 输出张量的形状就变为 [batch_size, hidden_dim, seq_len]
        h = h.squeeze(-1)

        # 经过全连接层得到每个类别的预测分数（logits)
        logits = self.fc(h)

        return attn_score, logits

@dataclass
class Config:
    # 词表大小
    vocab_size: int = 5000
    # 默认
    hidden_dim: int = 512
    # 多少头
    num_heads: int = 16
    head_dim: int = 32
    dropout: float = 0.1
    # 标签数
    num_labels: int = 2
    # 最长句子长度
    max_seq_len: int = 512
    # 训练次数
    num_epochs: int = 10

config = Config(5000, 512, 16, 32, 0.1, 2)
model = Model(config)
x = torch.randint(0, 5000, (3, 30))
print(x.shape)
attn, logits = model(x)
print(attn.shape, logits.shape)
```

<img width="545" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/4161bd45-e74f-4a69-9e59-3ae60005adfe">

# finetune

参考： https://zhuanlan.zhihu.com/p/650197598
大模型参数高效微调技术原理综述（五）-LoRA、AdaLoRA、QLoRA (https://zhuanlan.zhihu.com/p/636215898)

<img width="432" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/e0339909-4500-4ad4-9611-d5b309d78a53">

两个主要研究方向，以减少微调参数的数量，同时保持甚至提高预训练语言模型的性能。

方向一：添加小型网络模块：将小型网络模块添加到PLMs中，保持基础模型保持不变的情况下仅针对每个任务微调这些模块，可以用于所有任务。这样，只需引入和更新少量任务特定的参数，就可以适配下游的任务，大大提高了预训练模型的实用性。如：Adapter tuning、Prefix tuning、Prompt Tuning等，这类方法虽然大大减少了内存消耗。但是这些方法存在一些问题，比如：Adapter tuning引入了推理延时；Prefix tuning或Prompt tuning直接优化Prefix和Prompt是非单调的，比较难收敛，并且消耗了输入的token。

方向二：下游任务增量更新：对预训练权重的增量更新进行建模，而无需修改模型架构，即W=W0+△W。比如：Diff pruning、LoRA等， 此类方法可以达到与完全微调几乎相当的性能，但是也存在一些问题，比如：Diff pruning需要底层实现来加速非结构化稀疏矩阵的计算，不能直接使用现有的框架，训练过程中需要存储完整的∆W矩阵，相比于全量微调并没有降低计算成本。 LoRA则需要预先指定每个增量矩阵的本征秩 r 相同，忽略了在微调预训练模型时，权重矩阵的重要性在不同模块和层之间存在显著差异，并且只训练了Attention，没有训练FFN，事实上FFN更重要。

如果一个大模型是将数据映射到高维空间进行处理，这里假定在处理一个细分的小任务时，是不需要那么复杂的大模型的，可能只需要在某个子空间范围内就可以解决，那么也就不需要对全量参数进行优化了，我们可以定义当对某个子空间参数进行优化时，能够达到全量参数优化的性能的一定水平（如90%精度）时，那么这个子空间参数矩阵的秩就可以称为对应当前待解决问题的本征秩（intrinsic rank）。

预训练模型本身就隐式地降低了本征秩，当针对特定任务进行微调后，模型中权重矩阵其实具有更低的本征秩（intrinsic rank）。同时，越简单的下游任务，对应的本征秩越低。(https://arxiv.org/abs/2012.13255)

## LoRA

（论文：LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS），该方法的核心思想就是通过低秩分解来模拟参数的改变量，从而以极小的参数量来实现大模型的间接训练。

在涉及到矩阵相乘的模块，在原始的PLM旁边增加一个新的通路，通过前后两个矩阵A,B相乘，第一个矩阵A负责降维，第二个矩阵B负责升维，中间层维度为r，从而来模拟所谓的本征秩（intrinsic rank）。

<img width="447" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/df9ab7da-2433-4de1-a4a8-5c045f08dd90">

可训练层维度和预训练模型层维度一致为d，先将维度d通过全连接层降维至r，再从r通过全连接层映射回d维度，其中，r<<d，r是矩阵的秩，这样矩阵计算就从d x d变为d x r + r x d，参数量减少很多。

<img width="400" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/fb1da97f-b17b-4f6c-a41e-ae8552c1f04c">

通过消融实验发现同时调整Wq和Wv会产生最佳结果。

<img width="400" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/20dcfd31-ba3e-4470-a030-64ee9ab45e14">

## AdaLoRA

（论文：ADAPTIVE BUDGET ALLOCATION FOR PARAMETEREFFICIENT FINE-TUNING），是对LoRA的一种改进，它根据重要性评分动态分配参数预算给权重矩阵。具体做法如下：

> 调整增量矩分配。AdaLoRA将关键的增量矩阵分配高秩以捕捉更精细和任务特定的信息，而将较不重要的矩阵的秩降低，以防止过拟合并节省计算预算。

> 以奇异值分解的形式对增量更新进行参数化，并根据重要性指标裁剪掉不重要的奇异值，同时保留奇异向量。由于对一个大矩阵进行精确SVD分解的计算消耗非常大，这种方法通过减少它们的参数预算来加速计算，同时，保留未来恢复的可能性并稳定训练。

<img width="363" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/0cbc84e9-4e47-4045-97fc-6a3356f65471">

## QLoRA

QLoRA（论文： QLORA: Efficient Finetuning of Quantized LLMs），使用一种新颖的高精度技术将预训练模型量化为 4 bit，然后添加一小组可学习的低秩适配器权重，这些权重通过量化权重的反向传播梯度进行微调。QLORA 有一种低精度存储数据类型（4 bit），还有一种计算数据类型（BFloat16）。

<img width="478" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/9aa9e8be-bab8-42c6-9198-6712410c2fb2">

# LLM

LLaMA
- Tokenize
- Decoding
- Transformer Block

![561217a8861fd7afdfcf31f08f54c6f1_68747470733a2f2f716e696d672e6c6f766576697669616e2e636e2f626c6f672d6c6c616d612d617263682e6a7067](https://github.com/superkong001/NLP_diffusion/assets/37318654/8642c075-45a6-4931-aae7-80f59d247f54)

