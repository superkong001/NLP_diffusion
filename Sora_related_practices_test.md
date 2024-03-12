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

# 基于Transformers的Diffusion模型

<img width="816" alt="image" src="https://github.com/superkong001/NLP_diffusion/assets/37318654/14bba3ce-89fe-4817-abef-7ced1067f5c3">

基于Transformers的架构的Diffusion模型设计了一个简单而通用的基于Vision Transformers（ViT）的架构（U-ViT），替换了latent diffusion model中的U-Net部分中的卷积神经网络（CNN），用于diffusion模型的图像生成任务。

遵循Transformers的设计方法，这类方式将包括时间、条件和噪声图像patches在内的所有输入都视作为token。

# 相关推荐

SD模型网站： 

C站： https://civitai.com/
