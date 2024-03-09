参考： 一文读懂各种神经网络层（ Pytorch ）
> https://mp.weixin.qq.com/s/wOEL6Hj2lZ_OJclmnGTnHw

<img width="416" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/265034ad-1c76-4966-8b95-724c51b5bb06">


参考：Transformers技术解析+实战(LLM)，多种Transformers diffusion模型技术图像生成技术+实战
> https://github.com/datawhalechina/sora-tutorial/blob/main/docs/chapter2/chapter2_2.md

# SelfAttention 

## 知识点

![84b043c179aaad94406f9182af81c47b_seq2seq](https://github.com/superkong001/NLP_diffusion/assets/37318654/1f1f31cf-c9e0-43e9-9361-20842ac6576e)

加性Attention，如（Bahdanau attention）：

<img width="172" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/b686288b-d422-4206-98a3-0b32469c5ad7">

乘性Attention，如（Luong attention）：

<img width="309" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/e340d342-1fd1-491a-b315-7cb4f3bca70b">

"Attention is All You Need" 这篇论文提出了Multi-Head Self-Attention，是一种：Scaled Dot-Product Attention。

<img width="253" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/f8d058e2-a535-4d32-bf13-4388e50287f4">

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

<img width="326" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/7ec088b2-9e4e-49ac-b89f-8f6d032b8818">

<img width="623" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/bb0c06cf-bf25-4566-aa0f-e11078c4d9f7">

<img width="629" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/83dc3359-d7b1-4489-993c-0c77a7c77e3f">

<img width="491" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/0ca207d4-b104-4c0f-bcdc-a842faa9226d">

<img width="585" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/43a67ae9-62e8-415b-a1a6-464bcb71d6cd">

<img width="621" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/1b27e1fd-f820-424f-8952-36bbebdb6239">

## 实践体验

```Bash
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
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

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.attn = MySelfAttention(config)
        self.fc = nn.Linear(config.hidden_dim, config.num_labels)

    def forward(self, x):
        batch_size, seq_len = x.shape
        h = self.emb(x)
        # 送入自注意力机制，得到自注意力得分 (attn_score) 和加权的隐藏状态 (h)
        attn_score, h = self.attn(h)
        # h [batch_size, seq_len, -1]

        # 对隐藏状态进行平均池化
        # Applies a 1D average pooling over an input signal composed of several input planes.
        # 第一个参数是进行池化操作的输入张量，经过 permute 调整维度后的 h。
        # seq_len 是池化窗口的大小。例子中，池化窗口的大小被设置为序列的全长度，意味着对整个序列长度的特征进行平均。
        # 第三个参数 1 表示池化操作的步长。在这里，步长为 1 意味着池化窗口在移动时不会重叠
        # permute 方法用于调整张量的维度，以适应池化层的输入要求
        # input (Tensor) – the input tensor;
        # dims (tuple of int) – The desired ordering of dimensions
        # 从[batch_size, seq_len, hidden_dim] 转换为 [batch_size, hidden_dim, seq_len]
        h = F.avg_pool1d(h.permute(0, 2, 1), seq_len, 1)

        # squeeze 方法去除多余的维度,去除最后一个维度
        # 输出张量的形状就变为 [batch_size, hidden_dim, seq_len]
        h = h.squeeze(-1)
        logits = self.fc(h)

        return attn_score, logits

@dataclass
class Config:
    vocab_size: int = 5000
    hidden_dim: int = 512
    num_heads: int = 16
    head_dim: int = 32
    dropout: float = 0.1    
    num_labels: int = 2    
    max_seq_len: int = 512    
    num_epochs: int = 10

config = Config(5000, 512, 16, 32, 0.1, 2)
model = Model(config)
x = torch.randint(0, 5000, (3, 30))
print(x.shape)
attn, logits = model(x)
print(attn.shape, logits.shape)
```

<img width="545" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/4161bd45-e74f-4a69-9e59-3ae60005adfe">


# LLM
