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

## 实践体验

```Bash
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        assert config.hidden_dim % config.num_heads == 0
        
        self.wq = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.wk = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.wv = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.att_dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        k = k.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        v = v.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # (b, nh, ql, hd) @ (b, nh, hd, kl) => b, nh, ql, kl
        att = torch.matmul(q, k.transpose(2, 3))
        att /= math.sqrt(self.config.head_dim)
        score = F.softmax(att.float(), dim=-1)
        score = self.att_dropout(score)
        
        # (b, nh, ql, kl) @ (b, nh, kl, hd) => b, nh, ql, hd
        attv = torch.matmul(score, v)
        attv = attv.view(batch_size, seq_len, -1)
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
        self.attn = SelfAttention(config)
        self.fc = nn.Linear(config.hidden_dim, config.num_labels)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        h = self.emb(x)
        attn_score, h = self.attn(h)
        h = F.avg_pool1d(h.permute(0, 2, 1), seq_len, 1)
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

# LLM
