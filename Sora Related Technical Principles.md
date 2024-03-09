来源：

https://mp.weixin.qq.com/s/LQGwoU6xZJftmMtsQKde_w

厦门大学平潭研究院研究院杨知铮

# Sora 能力：

![1709189570545](https://github.com/superkong001/NLP_diffusion/assets/37318654/ba92e574-e683-487d-a024-c3f7150bdfb0)

# 相关paper

Twitter上广泛传播的论文《Scalable diffusion models with transformers》也被认为是Sora技术背后的重要基础

来自清华大学，人民大学和北京人工智能研究院等机构共同研究的CVPR2023的论文U-ViT《All are Worth Words: A ViT Backbone for Diffusion Models》，这项研究设计了一个简单而通用的基于vit的架构（U-ViT），替换了U-Net中的卷积神经网络（CNN），用于diffusion模型的图像生成任务。

GitHub链接：https://github.com/baofff/U-ViT

论文链接：https://arxiv.org/abs/2209.12152

模型链接：https://modelscope.cn/models/thu-ml/imagenet256_uvit_huge

<img width="850" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/0aa064ef-1f67-4171-97ae-d701aecab693">

Sora技术报告原文： https://openai.com/research/video-generation-models-as-world-simulators

核心摘要：我们探索了利用视频数据对生成模型进行大规模训练。具体来说,我们在不同持续时间、分辨率和纵横比的视频和图像上联合训练了以文本为输入条件的扩散模型。我们引入了一种transformer架构,该架构对视频的时空序列包和图像潜在编码进行操作。我们最顶尖的模型Sora已经能够生成最长一分钟的高保真视频,这标志着我们在视频生成领域取得了重大突破。我们的研究结果表明,通过扩大视频生成模型的规模,我们有望构建出能够模拟物理世界的通用模拟器,这无疑是一条极具前景的发展道。

# 相关技术解读

### 魔搭成晨解读

<img width="800" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/60cfbef1-d94b-4c0c-9533-19760baa3b9f">

## 模型训练流程

<img width="885" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/7f01d478-bb19-419b-959c-debcac07fd54">

<img width="853" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/b590156d-e914-46fa-940e-f837ea8c34a2">

Sora是一个在不同时长、分辨率和宽高比的视频及图像上训练而成的扩散模型，同时采用了Transformer架构

<img width="743" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/8747e9e3-b77e-45b5-82fb-bbc0fd4cccc4">

### 模型训练：扩散模型 DDPM

<img width="883" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/a6f11dd7-a2e3-45c8-a4ac-b614a41f13f3">

<img width="782" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/f1b7f79c-385a-4357-9f51-334b83b7f1bc">

### 模型训练：基于扩散模型的主干 U-Net

<img width="873" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/58142607-3195-4c31-b6ba-1175aef8b615">

<img width="889" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/a6b5ce50-fa2c-4b92-aac3-5baf37f8ddb4">

![image](https://github.com/superkong001/NLP_diffusion/assets/37318654/ad467aca-8d3a-4878-b99a-5b1ef68b14d2)

### Latent Diffusion Model(Stable Diffusion)

<img width="911" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/b587415e-6bd0-4146-9581-97668845ebe3">

# Sora关键技术拆解

## einops

einops是一个用于操作张量的库,它的出现可以替代我们平时使用的reshape、view、transpose和permute等操作。einops支持numpy、pytorch、tensorflow等

<img width="873" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/9e5d9f5d-6a7d-4068-956d-aa0d7117a5ff">

## Vision Transformer(ViT)

Vision Transformer (ViT) 模型由 Alexey Dosovitskiy等人在 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale 中提出。这是第一篇在 ImageNet 上成功训练 Transformer 编码器的论文，与熟悉的卷积架构相比，取得了非常好的结果。论文提出，虽然 Transformer 架构已成为自然语言处理任务事实上的标准，但其在计算机视觉中的应用仍然有限。 在视觉中，attention要么与卷积网络结合应用，要么用于替换卷积网络的某些组件，同时保持其整体结构不变。 ViT证明这种对 CNN 的依赖是不必要的，直接应用于图像块序列（patches）的纯 Transformer 可以在图像分类任务上表现良好。 当对大量数据进行预训练并转移到多个中型或小型图像识别基准（ImageNet、CIFAR-100、VTAB 等）时，Vision Transformer (ViT) 与SOTA的CNN相比取得了优异的结果，同时需要更少的计算资源来训练，Vision Transformer (ViT) 基本上是 Transformers，但应用于图像。

每个图像被分割成一系列不重叠的块（分辨率如 16x16 或 32x32），并线性embedding，接下来，添加position embedding，并通过编码器层发送。 在开头添加 [CLS] 标记以获得整个图像的表示。 可以在hidden states之上添加MLP head以对图像进行分类。

ViT架构：

<img width="889" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/d64549f4-bce7-41ba-b0cd-b81cdb714e9b">

![image](https://github.com/superkong001/NLP_diffusion/assets/37318654/7adfda20-750a-4a3f-a330-d533f42d115c)

Paper: > https://arxiv.org/abs/2010.11929

Official repo (in JAX): > https://github.com/google-research/vision_transformer

## Spacetime latent patches 

ViViT：视频ViT

ViViT基于纯变压器的视频模型分类，借鉴了ViT图像分类中取得的成功。 ViViT从输入视频中提取时空标记，然后由一系列转换器层进行编码。

1) 摊大饼法：从输入视频剪辑中均匀采样 n_t 个帧，使用与ViT相同的方法独立地嵌入每个2D帧(embed each 2D frame independently using the same method as ViT)，并将所有这token连接在一起

<img width="823" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/0442df1a-5cd2-4c40-814b-fc6941ab8a34">

<img width="835" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/341ece0e-a0b7-433d-b0eb-8dd7aede02d3">

将输入的视频划分为若干tuplet,每个tuplet会变成一个token。经过Spatial Temperal Attention进行空间/时间建模获得有效的视频表征token,即上图灰色block

<img width="862" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/be3e03e0-b92e-4692-910b-850cde9bea7b">

paper：https://arxiv.org/abs/2103.15691

<img width="889" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/85a803d7-83cf-4062-807a-5817d93b22a9">


tips:-技术难点：视频压缩网络类比于 Latent Diffusion Model 中的 VAE 但压缩率是多少,Encoder的复杂度、时空交互的range？Scale up？

<img width="914" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/f94a1ef5-1c51-4c05-8596-287b8c86f48c">

## Latte:用于视频生成的潜在扩散变压器

Latte提出了一种新颖的潜在扩散变压器，用于视频生成。Latte 首先从输入视频中提取时空标记，然后采用一系列 Transformer 块对潜在空间中的视频分布进行建模。为了对从视频中提取的大量标记进行建模，从分解输入视频的空间和时间维度的角度引入了四种有效的变体。为了提高生成视频的质量，我们通过严格的实验分析确定了 Latte 的最佳实践，包括视频剪辑补丁嵌入、模型变体、时间步级信息注入、时间位置嵌入和学习策略。我们的综合评估表明，Latte 在四个标准视频生成数据集（即 FaceForensics、SkyTimelapse、UCF101 和 Taichi-HD）上实现了最先进的性能。此外， Latte也 扩展到文本到视频生成 (T2V) 任务，其中 Latte 取得了与最新 T2V 模型相当的结果。我们坚信，Latte 为未来将 Transformer 纳入视频生成扩散模型的研究提供了宝贵的见解。

![image](https://github.com/superkong001/NLP_diffusion/assets/37318654/ce5c5e93-4eb7-4f27-85d7-1ad6b6425ba6)

## 网络结构：Diffusion Transformer，DiT

<img width="871" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/66b237a4-c78b-445e-a2f1-cd825476586e">

Tip-技术难点：

1. 训练数据怎么构建?

OpenAI 使用类似 DALLE3 的Cationining 技术训练了自己的 Video Captioner用以给视频生成详尽的文本描述

2. Transformer Scale up到多大?

3. 从头训练到收敛的trick?

4. 如何实现Long Context(长达1分钟的视频)的支持->切断+性能优化

5. 如何保证视频中实体的高质量和一致性? 

模型层不通过多个 Stage 方式来进行视频预测而是整体预测视频的 Latent在训练过程中引入 Auto Regressive的task帮助模型更好地学习视频特征和帧间关系

## 网络结构： DALLE 2

1. 将文本提示输入文本编码器，该训练过的编码器便将文本提示映射到表示空间；
   
2. 先验模型将文本编码映射到图像编码，图像编码捕获文本编码中的语义信息；
 
3. 图像解码模型随机生成一幅从视觉上表现该语义信息的图像；

<img width="576" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/9da42d3c-d8b2-4d33-acc5-a0e0a537551c">

summary：

1. Scaling Law:模型规模的增大对视频生成质量的提升具有明确意义，从而很好地解决视频一致性、连续性等问题；

2. Data Engine:数据工程很重要，如何设计视频的输入（e.g. 是否截断、长宽比、像素优化等）、patches 的输入方式、文本描述和文本图像对质量；
   
   AI Infra：AI 系统（AI 框架、AI 编译器、AI 芯片、大模型）工程化能力是很大的技术壁垒，决定了 Scaling 的规模;
   
3. LLM：LLM 大语言模型仍然是核心，多模态（文生图、图生文）都需要文本语义去牵引和约束生成的内容，CLIP/BLIP/GLIP 等关联模型会持续提升能力；

# Computation Cost

一分钟长度、每秒30帧的视频，平均每帧包含256个token，总计将产生 460k token。

以 34B 模型（这里只是一个猜测），需要7xA100资源推理。Dit XL 输入 512x512， 训练使用一个 TPU V3-256 Pod， 按照 TFLOPS 换算约等于 105 个 A100。那么 Sora 需要的训练和微调的资源会是多少?

<img width="743" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/74236f24-5689-4dda-a13c-c30c5a31cb39">

