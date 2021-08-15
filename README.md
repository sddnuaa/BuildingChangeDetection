# **基于PaddleSeg的遥感建筑变化检测**
本文利用PaddleSeg和LEVIR-CD建筑变化数据集试着完成基于U-Net 3+的遥感建筑变化检测。

# 一、项目背景
* 传统遥感建筑物变化检测主要通过纯人工或者软件加人工的方式进，人工成本高，效率较低。
* 目前政府部门的变化检测频率较高，传统方式跟不上政府的节奏。
* 随着人工智能技术不断成熟和新模型的不断涌现，采用人工智能的方式进行建筑物变化检测已成趋势。

# 二、数据集简介

> LEVIR-CD 由 637 个超高分辨率（VHR，0.5m/像素）谷歌地球（GE）图像块对组成，大小为 1024 × 1024 像素。这些时间跨度为 5 到 14 年的双时态图像具有显着的土地利用变化，尤其是建筑增长。LEVIR-CD涵盖别墅住宅、高层公寓、小型车库、大型仓库等各类建筑。在这里，我们关注与建筑相关的变化，包括建筑增长（从土壤/草地/硬化地面或在建建筑到新的建筑区域的变化）和建筑衰退。这些双时态图像由遥感图像解释专家使用二进制标签（1 表示变化，0 表示不变）进行注释。我们数据集中的每个样本都由一个注释者进行注释，然后由另一个进行双重检查以生成高质量的注释。

> ![](https://ai-studio-static-online.cdn.bcebos.com/fb96cc34fbaa4286b87799e2852d6d3cafdbf6ec5f6247f3ab55658ed0c1d35b)

数据来源：[https://justchenhao.github.io/LEVIR/](https://justchenhao.github.io/LEVIR/)

论文地址：[https://www.mdpi.com/2072-4292/12/10/1662](https://www.mdpi.com/2072-4292/12/10/1662)

## 2.1 数据加载和预处理

### 2.1.1 数据集的解压


```python
# 数据集解压
! mkdir -p datasets
! mkdir -p datasets/train
! mkdir -p datasets/val
! mkdir -p datasets/test
! unzip -q /home/aistudio/data/data104390/train.zip -d datasets/train
! unzip -q /home/aistudio/data/data104390/val.zip -d datasets/val
! unzip -q /home/aistudio/data/data104390/test.zip -d datasets/test
```

### 2.1.2 安装PaddleSeg


```python
# 安装paddleseg
! pip -q install paddleseg
```

### 2.1.3 生成数据列表
在LEVIR-CD中已经分为了train、test和val三个部分，每个部分有A、B和label三个文件夹分别保存时段一、时段二的图像和对应的建筑变化标签（三个文件夹中对应文件的文件名都相同，这样就可以使用replace只从A中就获取到B和label中的路径），list的每一行由空格隔开


```python
# 生成数据列表
import os

def creat_data_list(dataset_path, mode='train'):
    with open(os.path.join(dataset_path, (mode + '_list.txt')), 'w') as f:
        A_path = os.path.join(os.path.join(dataset_path, mode), 'A')
        A_imgs_name = os.listdir(A_path)  # 获取文件夹下的所有文件名
        A_imgs_name.sort()
        for A_img_name in A_imgs_name:
            A_img = os.path.join(A_path, A_img_name)
            B_img = os.path.join(A_path.replace('A', 'B'), A_img_name)
            label_img = os.path.join(A_path.replace('A', 'label'), A_img_name)
            f.write(A_img + ' ' + B_img + ' ' + label_img + '\n')  # 写入list.txt
    print(mode + '_data_list generated: {} instances'.format(len(A_imgs_name)))

dataset_path = 'datasets'  # data的文件夹
# 分别创建三个list.txt
creat_data_list(dataset_path, mode='train')
creat_data_list(dataset_path, mode='test')
creat_data_list(dataset_path, mode='val')
```

    train_data_list generated: 445 instances
    test_data_list generated: 128 instances
    val_data_list generated: 64 instances


### 2.1.4 构建数据集
这里主要是自定义数据Dataset，这里是选择继承paadle.io中的Dataset，而没有使用PaddleSeg中的Dataset，有以下几个问题需要注意
- 在初始化中transforms、num_classes和ignore_index需要，避免后面PaddleSeg在Eval时报错
- 这里使用transforms，因为通道数不为3通道，直接使用PaddleSeg的不行Compose会报错，需要将to_rgb设置为False
- 如果使用Normalize，需要将参数乘6，例如`mean=[0.5]*6`
- label二值图像中的值为0和255，需要变为0和1，否则啥也学不到，而且计算出来的Kappa系数还为负数
- 注意图像的组织方式——将两时段的图像在通道层concat起来（22-24行代码）


```python
# 构建数据集
import os
import cv2
import numpy as np
import paddle
from paddle.io import Dataset
from paddleseg.transforms import Compose, Resize
import paddleseg.transforms as T

class ChangeDataset(Dataset):
    # 这里的transforms、num_classes和ignore_index需要，避免PaddleSeg在Eval时报错
    def __init__(self, dataset_path, mode, transforms=[], num_classes=2, ignore_index=255):
        list_path = os.path.join(dataset_path, (mode + '_list.txt'))
        self.data_list = self.__get_list(list_path)
        self.mode = mode
        self.data_num = len(self.data_list)
        self.transforms = Compose(transforms, to_rgb=False)  # 一定要设置to_rgb为False，否则这里有6个通道会报错
        self.is_aug = False if len(transforms) == 0 else True
        self.num_classes = num_classes  # 分类数
        self.ignore_index = ignore_index  # 忽视的像素值
    def __getitem__(self, index):
        A_path, B_path, lab_path = self.data_list[index]
        A_img = cv2.cvtColor(cv2.imread(A_path), cv2.COLOR_BGR2RGB)
        B_img = cv2.cvtColor(cv2.imread(B_path), cv2.COLOR_BGR2RGB)
        image = np.concatenate((A_img, B_img), axis=-1)  # 将两个时段的数据concat在通道层
        label = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        if self.is_aug:
            image, label = self.transforms(im=image, label=label)
            image = paddle.to_tensor(image).astype('float32')
        else:
            image = paddle.to_tensor(image.transpose(2, 0, 1)).astype('float32')
        label = label.clip(max=1)  # 这里把0-255变为0-1，否则啥也学不到，计算出来的Kappa系数还为负数
        label = paddle.to_tensor(label[np.newaxis, :]).astype('int64')
        if self.mode == 'test':
            return image, label, A_img, B_img
        else:
            return image, label
    def __len__(self):
        return self.data_num
    # 这个用于把list.txt读取并转为list
    def __get_list(self, list_path):
        data_list = []
        with open(list_path, 'r') as f:
            data = f.readlines()
            for d in data:
                data_list.append(d.replace('\n', '').split(' '))
        return data_list

dataset_path = 'datasets'
# 完成三个数据的创建
# transforms = [Resize([512, 512])]

train_transforms = [  
    T.RandomHorizontalFlip(),  # 水平翻转
    T.RandomVerticalFlip(),  # 垂直翻转
    # T.RandomScaleAspect(),  # 随机缩放
    # T.RandomRotation(),  # 随机旋转
    T.Resize(target_size=(512, 512)), 
    T.Normalize(mean=[0.5]*6, std=[0.5]*6)   # 归一化
]

# 构建验证集
val_transforms = [
    T.RandomHorizontalFlip(),  # 水平翻转
    T.RandomVerticalFlip(),  # 垂直翻转
    # T.RandomScaleAspect(),  # 随机缩放
    # T.RandomRotation(),  # 随机旋转
    T.Resize(target_size=(512, 512)),
    T.Normalize(mean=[0.5]*6, std=[0.5]*6)   # 归一化
]

# 构建验证集
test_transforms = [
    T.Resize(target_size=(512, 512)),
    T.Normalize(mean=[0.5]*6, std=[0.5]*6)   # 归一化
]

train_data = ChangeDataset(dataset_path, 'train', train_transforms)
val_data = ChangeDataset(dataset_path, 'val', val_transforms)
test_data = ChangeDataset(dataset_path, 'test', test_transforms)
```

## 2.2 数据集查看


```python
%matplotlib inline
import matplotlib.pyplot as plt

for idx, (img, lab, A_img, B_img) in enumerate(test_data):  # 从test_data来读取数据
    if idx == 50:
        # m_img = img.reshape((1, 6, 512, 512))
        # s_img = img.reshape((6, 512, 512)).numpy().transpose(1, 2, 0)
        # # 拆分6通道为两个3通道的不同时段图像
        # s_A_img = s_img[:,:,0:3]
        # s_B_img = s_img[:,:,3:6]
        lab_img = lab.reshape((512, 512)).numpy()
        plt.figure(figsize=(10, 10))
        plt.subplot(1,3,1);plt.imshow(A_img.astype('int64'));plt.title('Time 1')
        plt.subplot(1,3,2);plt.imshow(B_img.astype('int64'));plt.title('Time 2')
        plt.subplot(1,3,3);plt.imshow(lab_img);plt.title('Label')
        plt.show()
        break
```

![](https://ai-studio-static-online.cdn.bcebos.com/27f1959967ff4560b83eb36179344ddfbdfaf759fcd7486b91bf49d6800b9818)


# 三、模型选择和开发



## 3.1 模型简介
U-Net由Olaf Ronneberger等人在在2015年MICCAI上提出。U-Net在神经元结构分割方面取得了巨大的成功，由于功能在层之间传播，因此其框架是突破性的。后续在U-Net的基础上涌现了许多优秀的架构如：U-Net++，Attention U-Net，U2-Net等，今天我们将介绍新的U-Net结构：U-Net 3+。

UNet，UNet++，UNet 3+ 结构图：

![](https://ai-studio-static-online.cdn.bcebos.com/ac32ceb9da5246fc9cec3060e5d413dd4e9ffd65864749208f26e500384b2319)

<center><b><p>左：UNet，中UNet++，右：UNet 3+</p></b></center>

通过增强U-Net架构，在多个数据集上U-NET 3+性能优于Attention UNET，PSPNet，DeepLabV2，DeepLabV3和DeepLabv3 +。UNet++使用嵌套和密集跳过连接，但它没有从全尺度探索足够的信息。在 UNet 3+ 中，使用了全面的跳过连接和深度监督：
  * 全尺度跳跃连接：将来自不同尺度特征图的低级细节与高级语义结合起来。
  
  ![](https://ai-studio-static-online.cdn.bcebos.com/1ab8b98834a64832b1138a66f3dd39060cc8a05ba4834a4bb086f72b2929dc4f)

  * 全尺度的深度监督：从全尺度聚合特征图中学习分层表示。
  
  ![](https://ai-studio-static-online.cdn.bcebos.com/5b629ab519114d1ea74686d6cf5955cecf83bb55925946358b680ef799a551c7)
   
  
  * 进一步提出了混合损失函数和分类引导模块（CGM）
  
  ![](https://ai-studio-static-online.cdn.bcebos.com/76a5be244b97416d939452577639c348dcab68aa58cd46518abf4810b3d8a599)
  
  
UNet 3+提供更少的参数，但可以产生更准确的位置感知和边界增强的分割图。

论文：UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation

论文链接：[https://arxiv.org/abs/2004.08790](https://arxiv.org/abs/2004.08790)

代码链接：[https://github.com/ZJUGiveLab/UNet-Version](https://github.com/ZJUGiveLab/UNet-Version)

## 3.2 检测原理
将两时相的图像叠起来形成RGBRGB格式的6通道图像，然后使用语义分割二分割的方法进行变化检测的学习。

## 3.3 模型训练
* 模型：UNet 3+
* 损失函数：BCELoss + LovaszSoftmaxLoss（通过MixedLoss类选择训练时的损失函数， 通过coef参数对不同loss进行权重配比，从而灵活地进行训练调参）
* 优化器：Adam
* 学习率变化：CosineAnnealingDecay

这里需要注意的就是模型的输入，需要6个通道。



```python
import paddle
from paddleseg.models import UNet3Plus
from paddleseg.models.losses import BCELoss
from paddleseg.models.losses import MixedLoss
from paddleseg.models.losses import LovaszSoftmaxLoss

from paddleseg.core import train

# 参数、优化器及损失
epochs = 60
batch_size = 4
iters = epochs * 400 // batch_size
base_lr = 2e-4
# losses = {}
# losses['types'] = [BCELoss()]
# losses['coef'] = [1]

losses = {}
losses['types'] = [MixedLoss([BCELoss(), LovaszSoftmaxLoss()], [0.7, 0.3])]
losses['coef'] = [1]

model = UNet3Plus(in_channels=6, num_classes=2)
# paddle.summary(model, (1, 6, 512, 512))  # 可查看网络结构
lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)  # 余弦衰减
optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters())  # Adam优化器
# 训练
train(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    optimizer=optimizer,
    save_dir='output1',
    iters=iters,
    batch_size=batch_size,
    # resume_model='output1/iter_3000',
    save_interval=300,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)
```

## 3.4 模型评估测试

> 模型评估（output），评估结果不错，在测试集表现一般，可能有些过拟合
>
> >  ```
> >  train_transforms = [ 
> >     T.RandomHorizontalFlip(),  # 水平翻转
> >     T.RandomVerticalFlip(),  # 垂直翻转
> >     T.RandomRotation(),  # 随机旋转
> >     T.RandomScaleAspect(),  # 随机缩放
> >     T.Resize(target_size=(512, 512)), 
> >     T.Normalize(mean=[0.5]*6, std=[0.5]*6)   # 归一化
>  > ]
> >  ```
>
> >  ```python
> >  2021-08-15 14:37:03 [INFO]	Start evaluating (total_samples: 64, total_iters: 64)...
> >  57/64 [=========================>....] - ETA: 0s - batch_cost: 0.1101 - reader cost: 0.08
> >  64/64 [==============================] - 7s 111ms/step - batch_cost: 0.1092 - reader cost: 0.08
> >  2021-08-15 14:37:10 [INFO]	[EVAL] #Images: 64 mIoU: 0.9431 Acc: 0.9868 Kappa: 0.9405 
> >  2021-08-15 14:37:10 [INFO]	[EVAL] Class IoU: 
> >  [0.985  0.9012]
> >  2021-08-15 14:37:10 [INFO]	[EVAL] Class Acc: 
> >  [0.9912 0.9564]
> >  2021-08-15 14:37:11 [INFO]	[EVAL] The model with the best validation mIoU (0.9438) was saved at iter 2700.
> >  ```
>
> >  ![](https://ai-studio-static-online.cdn.bcebos.com/7a142f2c8306495eb4b83b430e7528193bda3f06531f4272b4a13d0d97690c9d)

> 模型评估（output1），测试集表现还不错
> 
> > ```
> > train_transforms = [  
> >     T.RandomHorizontalFlip(),  # 水平翻转
> >     T.RandomVerticalFlip(),  # 垂直翻转
> >     T.Resize(target_size=(512, 512)), 
> >     T.Normalize(mean=[0.5]*6, std=[0.5]*6)   # 归一化
> > ]
> > ```
> 
> > ```
> > 2021-08-15 19:47:58 [INFO]	Start evaluating (total_samples: 64, total_iters: 64)...
> > /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:239: UserWarning: The dtype of left and right variables are not the same, left dtype is 
> > 64/64 [==============================] - 7s 106ms/step - batch_cost: 0.1048 - reader cost: 0.07
> > 2021-08-15 19:48:05 [INFO]	[EVAL] #Images: 64 mIoU: 0.8807 Acc: 0.9894 Kappa: 0.8660 
> > 2021-08-15 19:48:05 [INFO]	[EVAL] Class IoU: 
> > [0.9891 0.7723]
> > 2021-08-15 19:48:05 [INFO]	[EVAL] Class Acc: 
> > [0.9935 0.8914]
> > 2021-08-15 19:48:06 [INFO]	[EVAL] The model with the best validation mIoU (0.8807) was saved at iter 3300.
> > ```
> 
> > ![](https://ai-studio-static-online.cdn.bcebos.com/c8afbdc9b9d642ac84dc613705c2c5b74ff3db8210544f90a71202c74cf69db5)

> 模型评估（output2），loss下降很慢，效果不好,提前停止训练
> 
> > ```
> > train_transforms = [  
> >     T.RandomHorizontalFlip(),  # 水平翻转
> >     T.RandomVerticalFlip(),  # 垂直翻转
> >     T.RandomScaleAspect(),  # 随机缩放
> >     T.Resize(target_size=(512, 512)), 
> >     T.Normalize(mean=[0.5]*6, std=[0.5]*6)   # 归一化
> > ]
> > ```
> 
> > ```
> > 2021-08-15 18:01:15 [INFO]	Start evaluating (total_samples: 64, total_iters: 64)...
> > 64/64 [==============================] - 7s 108ms/step - batch_cost: 0.1061 - reader cost: 0.08
> > 2021-08-15 18:01:22 [INFO]	[EVAL] #Images: 64 mIoU: 0.8175 Acc: 0.9830 Kappa: 0.7810 
> > 2021-08-15 18:01:22 [INFO]	[EVAL] Class IoU: 
> > [0.9824 0.6527]
> > 2021-08-15 18:01:22 [INFO]	[EVAL] Class Acc: 
> > [0.9905 0.8023]
> > 2021-08-15 18:01:23 [INFO]	[EVAL] The model with the best validation mIoU (0.8175) was saved at iter 600.
> > ```
> 
> > ![](https://ai-studio-static-online.cdn.bcebos.com/b7a0d47360704a93bbed2b72c2f97f46d16b899c308b4d6c9db3a97e4a8e2827)

> 模型评估（output3），去掉随机缩放，loss下降很快，评估结果不错，测试集上显示伪变化过多。
>
> > ```
> > train_transforms = [  
> > T.RandomHorizontalFlip(),  # 水平翻转
> > T.RandomVerticalFlip(),  # 垂直翻转
> > T.RandomRotation(),  # 随机旋转
> > T.Resize(target_size=(512, 512)), 
> > T.Normalize(mean=[0.5]*6, std=[0.5]*6)   # 归一化
> > ]
> > ```
>
>
> > ```
> > 2021-08-15 21:35:48 [INFO]	Start evaluating (total_samples: 64, total_iters: 64)...
> > 64/64 [==============================] - 7s 108ms/step - batch_cost: 0.1063 - reader cost: 0.084
> > 2021-08-15 21:35:55 [INFO]	[EVAL] #Images: 64 mIoU: 0.9694 Acc: 0.9891 Kappa: 0.9688 
> > 2021-08-15 21:35:55 [INFO]	[EVAL] Class IoU: 
> > [0.986  0.9528]
> > 2021-08-15 21:35:55 [INFO]	[EVAL] Class Acc: 
> > [0.9917 0.9801]
> > 2021-08-15 21:35:56 [INFO]	[EVAL] The model with the best validation mIoU (0.9694) was saved at iter 2100.
> > ```
>
>>![](https://ai-studio-static-online.cdn.bcebos.com/b6c14f5d155445aa95c9adff8186f328259c772e50c346dda5bbb1bcf56e3d49)

> 训练过程中，数据增强策略进行了4次调整，output1和output3效果都不错，output1总体效果不错，output3把微小的变化都检测出来了。

# 四、效果展示
从test_data中测试一组数据，这里需要注意的就是
- 数据拆分为[0:3]和[3:6]分别表示两个时段的图像，类型需要转换为int64才能正常显示
- 加载模型只用加载模型参数，即.pdparams


```python
%matplotlib inline
import paddle
from paddleseg.models import UNet3Plus
import matplotlib.pyplot as plt

model_path = 'output1/best_model/model.pdparams'  # 加载得到的最好参数
model = UNet3Plus(in_channels=6, num_classes=2)
para_state_dict = paddle.load(model_path)
model.set_dict(para_state_dict)

for idx, (img, lab, A_img, B_img) in enumerate(test_data):  # 从test_data来读取数据
    if idx == 1:  # 查看第二个
        m_img = img.reshape((1, 6, 512, 512))
        m_pre = model(m_img)
        s_img = img.reshape((6, 512, 512)).numpy().transpose(1, 2, 0)
        # 拆分6通道为两个3通道的不同时段图像
        s_A_img = s_img[:,:,0:3]
        s_B_img = s_img[:,:,3:6]
        lab_img = lab.reshape((512, 512)).numpy()
        # pre_img = paddle.argmax(m_pre[0], axis=1).reshape((1024, 1024)).numpy()
        pre_img = paddle.argmax(m_pre[0], axis=1).reshape((512, 512)).numpy()
        plt.figure(figsize=(10, 10))
        # plt.subplot(2,2,1);plt.imshow(s_A_img.astype('int64'));plt.title('Time 1')
        # plt.subplot(2,2,2);plt.imshow(s_B_img.astype('int64'));plt.title('Time 2')
        plt.subplot(2,2,1);plt.imshow(A_img.astype('int64'));plt.title('Time 1')
        plt.subplot(2,2,2);plt.imshow(B_img.astype('int64'));plt.title('Time 2')
        plt.subplot(2,2,3);plt.imshow(lab_img);plt.title('Label')
        plt.subplot(2,2,4);plt.imshow(pre_img);plt.title('Change Detection')
        plt.show()
        break  # 只看一个结果就够了
```

![](https://ai-studio-static-online.cdn.bcebos.com/23090e187858400bb216a6e144b4c4eb0a1fbf76d1404b5d91c7a2e29d8735a2)


# 五、总结与升华
* UNet 3+ 使用了全面的跳过连接和深度监督，提供更少的参数，但可以产生更准确的位置感知和边界增强的分割图。
* 本项目采用UNet 3+ ，并且在数据增强、losss设计进行了优化，经过30个epoch训练，预测结果还是不错的，大概的建筑变化都检测出来了。


# 六、展望
* 进一步调参，提升精度，减少伪变化
* 优化小样本，提升精度
* 在EdgeBoard或者Jetson Nano上部署

# 作者简介
> 百度飞桨开发者AI学习新星

> 常用开发语言：C、Python、Luat、Go

> 常用操作系统：RTOS(RT-Thread、uC/OS)、CentOS、Raspberry Pi OS、LuatOS

> 工作领域：定位与导航、物联网通信（NB-IoT、CAT1）、遥感与地理信息、智慧城市、人工智能

> 工作经验10年+

> 现居城市：苏州

我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ 

[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/845928](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/845928)

