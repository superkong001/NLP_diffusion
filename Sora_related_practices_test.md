https://github.com/superkong001/NLP_diffusion/assets/37318654/7a9ea5cf-2c3f-4cc3-859d-7f4d21082f87

![shanhaijin-1](https://github.com/superkong001/NLP_diffusion/assets/37318654/50ffd4ee-2ce0-4497-9915-8dc94891bc18)


# 原AI制作视频

视频生成大致流程:
1. 先使用ChatGPT，写分镜剧本。
2. 在使用SD，MJ等软件画图。
3. 最后利用SVD，Pika，Runway登软件生成视频。

## 如何搭配使用?

生图类软件之间:   
> Dalle出想法、出草图、配合ChatGpt出提示词。

> Midjourney做美工打手

Midjourney基本咒语:
> 图片链接提示+文字描述提示[场景][主体][情境][风格][光线][镜头]+后缀参数提示。

<img width="480" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/5302f6f2-0a72-44b9-9f3c-734d1b9b1323">

<img width="480" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/dc034549-6fbc-4cdf-853a-7bf695b5e833">

视频生成软件优缺点一览表：

<img width="482" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/92a7b2fc-dec2-4e97-8fce-dda8aebb7ad0">

## 剧本
GPT4, eg:

```Bash
Q: 我想用《山海经》做一个1分钟左右的短片，用那种第一人称讲述的视角，你有什么好想法吗?
A:
使用《山海经》作为短片的主题是一个非常有创意的想法，特别是通过第一人称的视角来呈现。这样的视角可以让观众更加沉浸在故事中，感受到《山海经》的神秘与奇幻。以下是我的构思:
脚本创作从灵感到脚本
短片构思:《山海探秘》
故事背景
·时代背景:古代神话时期。
地点:《山海经》中的一个神秘山脉或海域。
主角设定
一个好奇而勇敢的年轻探险者，热爱探索未知的世界，

故事大纲....

视觉与音效
运用丰富的色彩和细腻的特效来刻画奇幻的山海世界，
配乐神秘悠扬，与《山海经》的主题相契合。
分镜头创作

技术要点
使用稳定的手持相机或稳定器拍摄，以保证第一人称视角的流畅性。
在剪辑时穿插一些插画或动画来增强故事的视觉效果。
这样的短片可以吸引观众的好奇心，同时展现《山海经》的奇幻魅力

Q: 觉得基本的构思可以 但是我更希望是一个老者的视角 感觉老者的那种嗓音更适合《山海经》的风格
A: 将短片的叙述视角改为一位老者，确实会增加一种历史的深度和神秘感，使之更加符合《山海经》的风格。以下是调整后的短片构思:
1.开场(10秒):
·镜头通过迷雾缓缓推进，展现一片神秘莫测的山海景观。
·主角(旁白)简短介绍自己对《山海经》的兴趣和探险目的。2.探险开始(15秒):
得到·AI学习圈
1专场直播
分镜头创作②
·镜头第一人称视角，穿越茂密的森林，攀登险峻的山峰,
插入《山海经》中的一些描述，如奇异的植物和神秘的地貌
3.遇见奇幻生物(20秒):主角偶遇《山海经》中描述的神话生物(如凤凰、九尾狐等)短暂的交互和惊艳的视觉效果，展现这些生物的神秘与美丽。4.意外事件(10秒):
。主角意外发现一条隐藏的路径，或是一些古老的遗迹，
·通过旁白表达对这一发现的好奇和兴奋。
5.高潮与发现(15秒):
主角跟随线索进入一个未知的洞穴或神秘的地下世界。
"发现一些重要的线索或宝藏，这些可能与《山海经》中的传说有关，6.结尾(10秒):
·主角完成探险，回顾整个旅程的精彩瞬间
用一段富有哲理的旁白结束，表达对未知世界的敬畏和对探险的热爱，
```

## 提示词影响

<img width="714" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/67a06e0f-c718-455b-8cc3-022f47b586c1">

<img width="838" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/cf878573-201d-40d8-9aab-50cd39bc262c">

## 相关推荐

SD模型网站： 

C站： https://civitai.com/

## 创作

```Bash
故事设定
开头（10秒）： 画面以动画形式展开，主角（一个穿着现代服饰的年轻人，携带一部智能手机，上面贴着《山海经》的电子贴纸）意外穿越到一个充满二次元风格的《山海经》世界中。主角一脸困惑，环顾四周，这里的一切都与他平时在动漫中看到的奇幻场景如出一辙。

发展（35秒）： 主角开始探索这个世界，遇到各种以二次元风格重新设计的《山海经》生物，例如：萌版九尾狐、战斗力超强的貔貅、潇洒帅气的青龙等。这些生物拥有独特的设计和色彩，符合年轻人的审美。他们与主角交流时，对话框中弹出的文字充满了网络流行语和梗，增添幽默感。

高潮（10秒）： 主角在一位看起来像是高级玩家的NPC（实际上是《山海经》世界的向导）的帮助下，找到了一条通往神秘宫殿的路径。宫殿的门前是一场迷你游戏，玩家需要利用手中的智能手机与《山海经》的知识解决谜题。

结尾（5秒）： 成功解决谜题后，宫殿的大门缓缓开启，主角走入一间充满光芒的宝库，里面藏有返还现实世界的门户。主角回头一笑，走入门户，画面切回现实，他坐在电脑前，屏幕上显示的是他正在玩的一款以《山海经》为背景的二次元风格游戏。

制作建议
视觉风格： 强调二次元动画风格，使用鲜艳的颜色和夸张的表情设计。生物和场景设计应结合现代流行元素，如电子游戏、ACG文化（动画、漫画、游戏）的影响。
音效和配乐： 使用电子音乐或是年轻人喜爱的二次元歌曲作背景音乐，对话和环境音效可借鉴流行游戏和动画。
叙述方式： 故事采用轻松幽默的口吻，对话中融入年轻人的语言风格和网络流行词汇，使其更加亲切和有趣。
通过这样的调整，短片不仅展现了《山海经》的奇幻元素，还巧妙地融入了年轻人喜爱的二次元文化，使其更加符合00后观众的口味和审美。
```

-neg 负面描述

neg lack of detail, erratic fluctuation in motion, noisy, bad quality, distorted, poorly drawn,morphing blurry, grainy, low resolution, oversaturated

# 训练一个 sora 模型

## 数据和模型准备

参考： https://github.com/datawhalechina/sora-tutorial/blob/main/docs/chapter3/chapter3_1/chapter3_1.md

## 声音合成

TTS WIKI 参考 https://github.com/alibaba-damo-academy/KAN-TTS/wiki

### 架构，基于Transformers的Diffusion模型

<img width="816" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/14bba3ce-89fe-4817-abef-7ced1067f5c3">

基于Transformers的架构的Diffusion模型设计了一个简单而通用的基于Vision Transformers（ViT）的架构（U-ViT），替换了latent diffusion model中的U-Net部分中的卷积神经网络（CNN），用于diffusion模型的图像生成任务。

遵循Transformers的设计方法，这类方式将包括时间、条件和噪声图像patches在内的所有输入都视作为token。

在 Sora 的技术报告中，Sora 使用视频压缩网络将各种大小的视频压缩为潜在空间中的时空 patches sequence，然后使用 Diffusion Transformer 进行去噪，最后解码生成视频。

![image](https://github.com/superkong001/NLP_diffusion/assets/37318654/0f4144cb-8ff1-4b33-9f80-8a1b698c64e0)

Open-Sora  在下图中总结了  Sora  可能使用的训练流程： https://hpc-ai.com/blog/open-sora

![image](https://github.com/superkong001/NLP_diffusion/assets/37318654/3b4999b7-aa81-4b73-95f9-45c1ddad9acf)

训练链路：

![image](https://github.com/superkong001/NLP_diffusion/assets/37318654/7f7cbac0-2bdf-4695-ba34-fa9c25bd7479)

### 数据准备

开源数据集：

**VideoInstruct-100K：**

VideoInstruct100K  是使用人工辅助和半自动注释技术生成的高质量视频对话数据集。数据集中的问题答案与以下内容相关：

- 视频摘要
- 基于描述的问题答案（探索空间、时间、关系和推理概念）
- 创意/生成性问题解答

链接：[https://modelscope.cn/datasets/AI-ModelScope/VideoInstruct-100K](https://modelscope.cn/datasets/AI-ModelScope/VideoInstruct-100K/summary)

**panda-70m：**

Panda-70M  是一个包含  70M  高质量视频字幕对的大规模数据集。该存储库分为三个部分：

- 数据集数据加载包括列出  Panda-70M  数据的  csv  文件以及下载数据集的代码。
- 分割包括将长视频分割成多个语义一致的短片的代码。
- 字幕包括在  Panda-70M  上训练的拟议视频字幕模型。

**链接：**[https://modelscope.cn/datasets/AI-ModelScope/panda-70m](https://modelscope.cn/datasets/AI-ModelScope/panda-70m/summary)

**Youku-mPLUG:**

Youku-mPLUG 预训练数据集挖掘自优酷站内海量的优质短视频内容

- 包含千万级别约 36TB 的视频、文本数据。
- 其中视频均为覆盖 10 ～ 120 秒的 UGC 短视频内容，文本为视频对应的描述标题，长度 5 ～ 30 不等。
- 该数据集抽取时品类均衡，内容共包含 45 个大类。

链接：[https://modelscope.cn/datasets/modelscope/Youku-AliceMind](https://modelscope.cn/datasets/modelscope/Youku-AliceMind/summary)

**MSR-VTT：**

MSR-VTT（Microsoft Research Video to Text）是一个开放域视频字幕的大规模数据集。

- 由  20  个类别的  10,000  个视频片段组成，每个视频片段由  Amazon Mechanical Turks  标注了  20  个英文句子。
- 所有标题中约有  29,000  个独特单词。
- 标准分割使用  6,513  个 split 用于训练，497  个 split 用于验证，2,990  个 split 用于测试。

链接：[https://modelscope.cn/datasets/AI-ModelScope/msr-vtt](https://modelscope.cn/datasets/AI-ModelScope/msr-vtt/summary)

**Shot2Story：**

视频文本基准和用于多镜头视频理解的可扩展代码。包含 20k  视频的详细长摘要和  80k  视频镜头的镜头字幕。

链接：[https://modelscope.cn/datasets/AI-ModelScope/Shot2Story](https://modelscope.cn/datasets/AI-ModelScope/Shot2Story/summary)

**InternVid：**

InternVid  是一个以视频为中心的大规模多模态数据集，可以学习强大且可转移的视频文本表示，以实现多模态理解和生成。 InternVid  数据集包含超过  700  万个视频，持续近  76  万小时，产生  2.34  亿个视频剪辑，并附有总共  4.1B  个单词的详细描述。

链接：[https://modelscope.cn/datasets/AI-ModelScope/InternVid](https://modelscope.cn/datasets/AI-ModelScope/InternVid/summary)

**webvid-10M：**

大型文本视频数据集，包含从素材网站抓取的**1000  万个视频文本对。**

链接：[https://modelscope.cn/datasets/AI-ModelScope/webvid-10M](https://modelscope.cn/datasets/AI-ModelScope/webvid-10M/summary)

### 数据预处理

- [FFmpeg](https://github.com/FFmpeg/FFmpeg)：150w 行+源码，大量底层细节

  - pytorchvideo：主要支持[加载](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html)和少量单 video 模态的[tensor transform](https://pytorchvideo.readthedocs.io/en/latest/api/transforms/transforms.html#module-pytorchvideo.transforms.functional)（翻转、扰动、采样等）

- SORA 官方仅模糊提及使用了 DALLE3 来生成 caption，细粒度的"caption --> spacetime patch"建模比较关键
- 从 SORA 模型效果看，数据需要有变化的时长、分辨率和宽高比

[Data-Juicer](https://github.com/alibaba/data-juicer/docs/DJ_SORA_ZH.md)  扩展了对多模态数据的支持，已实现上百个专用的视频、图像、音频、文本等多模态数据处理算子及工具，帮助用户分析、清洗及生成大规模高质量数据。

开源链接：https://github.com/alibaba/data-juicer/docs/DJ_SORA_ZH.md

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/9554c372-caba-4c36-94c0-46499400da04.png)


### 模型选型和训练

参考： https://github.com/modelscope/modelscope-classroom/tree/main/AIGC-tutorial

