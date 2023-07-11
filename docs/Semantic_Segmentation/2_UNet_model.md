# UNet 模型

进行中……

## 1 模型框架

论文地址：[arxiv.org/pdf/1505.04597.pdf](https://arxiv.org/pdf/1505.04597.pdf)

该论文模型 U-Net 在2015年被提出

我基于 `Pytorch` 模型实现（进行中...）：[jermainn/UNet: 基于 Pytorch 框架实现 UNet 网络 (github.com)](https://github.com/jermainn/UNet)

### 1.1 研究背景

2012年，Krizhevsky等人提出了 AlexNet 网络模型，可谓是深度学习的开山之作。其将 CNN 的基本原理应用到了很深的网络之中，首次在大规模图像数据集实现了深层卷积神经网络结果，点燃了深度学习的火焰。

之后，深度卷积神经网络在许多视觉识别任务中都变现突出，又提出了R-CNN、VGGNet等。

### 1.2 研究问题

卷积神经网络主要应用与图像的分类问题，输出单类别标签。然而，在医学影像处理过程中，还需要获得定位信息，也就是说，需要像素级的类别标签。并且生物医学任务中也没有大量的数据集可供使用。

> [!NOTE]
>
> (1) 像素级的预测
>
> (2) 数据量少

Ciresan等人提出的DNN网络有着可局部化和图块训练数据远大于图像数量的优点，但是，也存在着两个明显的缺陷：训练时间长；需要在局部准确度和获取整体上下文信息之间取舍。

### 1.3 方法总结

#### 网络结构

![UNet](https://cdn.jsdelivr.net/gh/jermainn/imgpic@master/note_img/UNet.webp)

UNet 网络延续了全卷积FCN网络的思想，其结构如上图所示，其中深蓝色的箭头表示卷积过程，采用 `3x3` 的卷积核，且不使用 `padding` ，红色的箭头表示 `max_pooling` ，绿色的箭头表示转置卷积（反卷积），浅蓝色的箭头表示 `1x1` 的卷积操作，灰色箭头表示将左侧特征层进行中心裁剪后与右侧特征层进行拼接。

该网络呈现U形的对称结构，该网页也因此的命名。该网络由两段路径组成， the contractiong path 和 the expansive path，可以看作是一个编解码过程（Encoder-Deconder）的过程。

在前一段中，采用了 `3x3` 的卷积核（不使用 `padding` ）进行卷积操作，采用 `ReLU` 激活函数，通过 `2x2` 的 `max_pooling（stride 2）`  实现下采样，并在下采样会后通过卷积将特征层数加倍。

在后一段，通过转置卷积进行上采样，在上采样过程中减半特征层数，同时将左侧的特征进行中心裁剪后与右侧特征层进行拼接，每层上采样之后，再进行两次的 `3x3` 的卷积，采用 `ReLU` 激活函数。在输出层采用了 `1x1` 的卷积。

#### 模型训练

论文中采用了随机梯度下降法（SGD）进行训练。论文中更加倾向于使用较大的图像切片，直至减小批量数到1，采用大动量（0.99）以使大量之前的样本来决定当前的优化步骤的更新。

损失函数是逐像素的 `softmax` 函数和交叉熵函数的结合

`softmax` 函数：
$$
p_k(x) = \frac{e^{a_k(x)}}{\sum^{K}_{k^{'}}{e^{a_{k^{'}}(x)}}}
$$

> 其中 $ a_k(x)$ 表示在第 $ k $ 层的特征图上 $x$ 位置处的激活值，$K$ 表示总类别数，也是总层数，$p_k(x)$ 是似然函数

交叉熵函数：
$$
E = \sum_{x \in \Omega}{w(x) \log{(p_{{\scr{l}}_{(x)}}(x))}}
$$

> 其中 $\scr{l}$ 是真实标签，$w$ 是对像素级别引入的权重值

权重图通过标注图来计算，以补偿训练图像中某类像素的不同频率，同时迫使网络更好地学习到同一类别接触面之间的小边界。权重计算公式：
$$
w(x) = w_c(x) + e^{- \frac{(d_1(x)+d_2(x))^2}{2 {\sigma}^2}}
$$

> 其中 $w_c$ 是权重图以平衡类间像素的频率，$d_1$ 表示离最近边界的距离，$d_2$ 表示离第二近边界的距离

> [!ATTENTION]
>
> 这里还要再理解一下

#### 初始化权重

论文中通过从具有标准偏差 $\sqrt{2/N}$的高斯分布中取得初始权重，其中 $N$ 为每个神经元的输入节点数量。例如 $3\times 3$ 的 $64$ 通道的卷积层 $N=3\times 3\times 64=576$ 

> [!NOTE]

#### 数据增强

只有少量数据的情况下获得较好的鲁棒性，对于显微图像，需要考虑平移和旋转不变性以及形变和灰度变化的鲁棒性。论文中主要使用了平滑形变。

> [!ATTENTION]
>
> 上面部分还需思考

##### overlap-tile 策略

另外由于论文中UNet模型在卷积的过程中没有使用 `padding` ，因此输出额特征图的大小也不断减小。而作者使用了 overlap-tile 策略，在输入图像之前将原图像进行镜像的边缘填充。

overlap-tile-strategy 可以搭配图像分块 patch 进行使用，当内存资源有效无法对一整幅大图进行训练的时候，可以对图像进行分块。而相邻两块之间使用该策略，使其具有一定的重叠，以此获得图像上下文信息，从而实现图像的无缝分割。

### 1.4 创新点

1. 延续了全卷积的思想，同时在上采样的过程中调整为多层特征层以更好地传递上下文信息，
2. 特征融合，浅层的特征图拼接到后续层，呈现一个U形，且取得较好的结果
3. 网络结构简单高效，使用简单的数据增强，可以仅通过少量数据获得良好成效
4. 引入权重图，更有效学习同一类别接触面间的分割边界

### 1.5 实验结论

在仅有的少量数据集中，通过数据增强、特征融合、引入权重图等方法，在 EM segmentation challenge 和 the cell tracking challenge(PhC-U373 和 DIC-HeLa) 任务中取得优秀的表现。

---



## 2 模型构建

模型的实现使用 `Pytorch` 框架；

在我的模型的实现过程中，在论文中所述模型的基础上，在卷积的过程中引入了 `padding` ，而没有使用 `overlap-tile` 策略；

另外，提供参数 `bilinear` ，可以选择是否使用双线性插值的上采样 `nn.Upsample` 来替换原论文汇中的转置卷积 `nn.ConvTranspose2d` ；

```python
import torch.nn as nn
import torch.nn.functional as F
import torch

class DoubleConv(nn.Module):
    """
    每次 maxpool 之后进行两次卷积
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)

        )

    def forward(self, x):
        return self.maxpool(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSampling, self).__init__()
        if bilinear:
            # 双线性插值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 保证 x1 与 x2 同大小，可以拼接
        # [left, right, top, bottom] 填充行列数
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # [N, C, H, W] 在 channel 上拼接
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Sequential):
    """
    这里父级采用了，nn.Sequential
    """
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class UNet(nn.Module):
    def __init__(self,
                 n_channels,
                 num_classes,
                 bilinear=False,
                 base_c=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # the contracting path
        self.in_conv = DoubleConv(self.n_channels, base_c)
        self.down1 = DownSampling(base_c, base_c*2)
        self.down2 = DownSampling(base_c*2, base_c*4)
        self.down3 = DownSampling(base_c*4, base_c*8)
        # 转置卷积 和 双线性插值 在每块的最后一层不太一样
        factor = 2 if bilinear else 1
        self.down4 = DownSampling(base_c*8, base_c*16//factor)
        # the expansive path
        self.up1 = UpSampling(base_c*16, base_c*8//factor, bilinear)
        self.up2 = UpSampling(base_c*8, base_c*4//factor, bilinear)
        self.up3 = UpSampling(base_c*4, base_c*2//factor, bilinear)
        self.up4 = UpSampling(base_c*2, base_c, bilinear)
        # out
        self.outc = OutConv(base_c, num_classes)

    def forward(self, x):
        # the contracting path
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        # the expansive path
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up1(x, x1)

        return self.outc(x)

if __name__ == '__main__':
    net = UNet(n_channels=1, num_classes=1)
    print(net)
```

模型结构

```python
UNet(
  (in_conv): DoubleConv(
    (double_conv): Sequential(
      (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
  (down1): DownSampling(
    (maxpool): Sequential(
      (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): DoubleConv(
        (double_conv): Sequential(
          (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
        )
      )
    )
  )
  (down2): DownSampling(
    (maxpool): Sequential(
      (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): DoubleConv(
        (double_conv): Sequential(
          (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
        )
      )
    )
  )
  (down3): DownSampling(
    (maxpool): Sequential(
      (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): DoubleConv(
        (double_conv): Sequential(
          (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
        )
      )
    )
  )
  (down4): DownSampling(
    (maxpool): Sequential(
      (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): DoubleConv(
        (double_conv): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
        )
      )
    )
  )
  (up1): UpSampling(
    (up): ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))
    (conv): DoubleConv(
      (double_conv): Sequential(
        (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
  (up2): UpSampling(
    (up): ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
    (conv): DoubleConv(
      (double_conv): Sequential(
        (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
  (up3): UpSampling(
    (up): ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
    (conv): DoubleConv(
      (double_conv): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
  (up4): UpSampling(
    (up): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
    (conv): DoubleConv(
      (double_conv): Sequential(
        (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
    )
  )
  (outc): OutConv(
    (0): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

---



## 3 数据读取

下载了论文中进行 EM segmentation challenge 中使用的数据集

其中，还有已经标注的 30 张 $512 \times 512$ 像素大小的图片和标签

```python
from torch.utils import data
import glob
import os
from PIL import Image
import random
import numpy as np

class isbiDataSet(data.Dataset):
    def __init__(self, root):
        """
        类初始化函数：
        根据指定的路径读取所有图片数据
        """
        self.root = root
        # # 返回不带后缀的文件名
        self.img_ids = [file.split('.')[0]
                        for file in os.listdir(os.path.join(root, "train/image"))]
        # print(self.img_ids, len(self.img_ids))

        # # 返回所有满足要求的文件路径列表
        # self.imgs_path = glob.glob(os.path.join(data_root, "train/image/*.png"))
        # print(self.imgs_path)

    def __len__(self):
        """
        返回数据量多少
        """
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        数据获取函数：
        数据的读取与预处理
        """
        name = self.img_ids[index]
        # 图片和标签
        image = Image.open(
            os.path.join(self.root, "train/image/%s.png" % name))
        label = Image.open(os.path.join(self.root, "train/label/%s.png" % name))
        # 读入的单通道图像 否则用 .convert('L')
        # resize
        image = image.resize(image.size, Image.BICUBIC)
        label = label.resize(image.size, Image.BICUBIC)

        # 随机翻转
        flipCode = random.randint(0, 7)
        if flipCode:
            image = image.transpose(flipCode-1)
            label = label.transpose(flipCode-1)

        image = np.asarray(image)
        label = np.asarray(label)

        if label.max() > 1:
            label = label / 255      # 将标签 255 变为 1
        return image.copy(), label.copy()

if __name__ == "__main__":
    root = "./../dataset"
    dataset = isbiDataSet(root)
    print("数据的个数：", len(dataset))
    train_loader = data.DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    for image, label in train_loader:
        print(image.shape, image.size())
```

---



## 4 模型训练

进行中……

