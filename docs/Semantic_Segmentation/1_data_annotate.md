# 数据标注

## 0 标注工具

常见的工具有：

- `labelme` 
- `labelimg` 
- `PhotoShop` 
- `Matlab` 

下面主要介绍使用 `labelme` 和 `Matlab` 工具进行标注的方法

---

## 1 `labelme` 安装与使用

`labelme` 是麻省理工（MIT）计算机科学与人工智能实验室（CSAIL）研发的图像标注工具。其源代码是由 `JavaScript` 语言开发，作为在线的图像标注工具

在线使用：[LabelMe. The Open annotation tool (mit.edu)](http://labelme.csail.mit.edu/Release3.0/)

开源源码：[CSAILVision/LabelMeAnnotationTool: Source code for the LabelMe annotation tool. (github.com)](https://github.com/CSAILVision/LabelMeAnnotationTool)

由该设计灵感进行设计，通过 `Python` 进行编写，结合 `QT（PyQT）` ，得到下列图像标注软件

项目地址：[wkentaro/labelme: Image Polygonal Annotation with Python (polygon, rectangle, circle, line, point and image-level flag annotation). (github.com)](https://github.com/wkentaro/labelme)

> [!NOTE]
>
> 该项目介绍中也包含了安装方法，下面主要使用了该软件进行图像的标注，在 `Anaconda` 环境中安装

### 安装

首先进入 `conda` 创建的环境，通过 `pip` 命令进行安装

```cmd
pip install labelme
```

等待安装结束

### 使用

通过 `-h` 查看 `labelme` 参数含义

```
labelme -h

# 参数
--flags FLAGS		多分类标签，用逗号分隔；或者文件
--labels LABELS		分割标签，用逗号分隔；或者列表文件
--nosortlabels		是否对标签进行排序
```

> [!NOTE]
>
> 提前建立 `label.txt` 标签列表

打开软件界面

```cmd
labelme
```

<img src="https://cdn.jsdelivr.net/gh/jermainn/imgpic@master/note_img/labelUI.png" alt="labelUI" style="zoom:67%;" />

点击保存后，会生成 `.json` 文件

使用下述命令进行查看已经标注的 `.json` 文件

```cmd
labelme_draw_json 文件名.json
```

<img src="https://cdn.jsdelivr.net/gh/jermainn/imgpic@master/note_img/jsonLabeled.webp" alt="jsonLabeled"  />

将标注的 `.json` 文件转化为 `.png` 图片文件（单个）

```cmd
labelme_json_to_dataset 文件.json --output 输出文件夹
```

**批量统一转化 `.json` 文件** 

-  使用 `labelme2voc.py` 文件：[labelme/examples/semantic_segmentation at main · wkentaro/labelme · GitHub](https://github.com/wkentaro/labelme/tree/main/examples/semantic_segmentation)

  > 这是作者给出的语义分割图像标注的例子

- 首先创建 `label.txt` 标签列表

  ```
  -- labels.txt --
  
  __ignore__
  _background_
  person
  bottle
  chair
  sofa
  bus
  car
  ```

  > 标签列表可以使转换过程中统一各个标签的编码，其中， `__ignore__` 和 `_background_` 需要有
  >
  > 其中 `__imgnor__` 转换为 255，在 `npy` 文件中为 -1， `_background_` 转换为 0，下面按顺序分配

命令行中

```cmd
python 目录/labelme2voc.py 目录1 目录2 --labels 目录3/labels.txt

# 目录1 是图片和json文件所在目录
# 目录2 是输入文件夹目录
# 目录2 是标注列表文件所在目录

# 例如：
python labelme2voc.py data_annotated data_annotated_voc --labels labels.txt
```

生成 `VOC` 的目录结构

```
data_annotated_voc
├── JPEGImages							# 原图片
├── SegmentationClass					# npy 文件
├── SegmentationClassPNG				# png 格式标注文件
├── SegmentationClassVisualization		# 标注与原图掩膜图片
└── class_names.txt						# 转换过程中用到的所有类别
```

![dataVOC](https://cdn.jsdelivr.net/gh/jermainn/imgpic@master/note_img/dataVOC.webp)

> 可以使用 `labelme_draw_label_png 图片.png` 查看单个标注的 `.png` 图片

---



## 2 `Matlab` 工具的使用

需要用到 `Matlab` 的 `imageLabeler` 工具进行标注

该工具位于 `Computer Vision Toolbox` (计算机视觉工具箱) 中

### 使用

直接在命令窗口输入 `imageLabeler` ，或者在 matlab 的菜单栏点击 `app` 下打开

<img src="https://cdn.jsdelivr.net/gh/jermainn/imgpic@master/note_img/imageLabeler.webp" alt="imageLabeler" style="zoom: 50%;" />

工具界面

<img src="https://cdn.jsdelivr.net/gh/jermainn/imgpic@master/note_img/imageLabelerUI1.webp" alt="imageLabelerUI1" style="zoom:70%;" />

图像标注

<img src="https://cdn.jsdelivr.net/gh/jermainn/imgpic@master/note_img/imageLabelerUI2.webp" alt="imageLabelerUI2" style="zoom:70%;" />

> [!TIP]
>
> 在工具栏中，有智能标注、笔刷等选择

文件保存

<img src="https://cdn.jsdelivr.net/gh/jermainn/imgpic@master/note_img/imageLabelerUI3.webp" alt="imageLabelerUI3" style="zoom:70%;" />

> 保存标注结果，或者保存(save) 会话(session)
>
> 标注结果为 png 图像

---



## 3 数据读取

```python
import numpy as np
from PIL import Image

img = Image.open("./data_annotated_voc/JPEGImages/03.jpg").convert('RGB')
lbl = Image.open("./data_annotated_voc/SegmentationClassPNG/03.png")

# resize
img = img.resize([500, 375], Image.BICUBIC)
lbl = img.resize([500, 375], Image.BICUBIC)

# 
img = np.asarray(img, np.float32)
lbl = np.asarray(lbl, np.float32)
```

---



## 4 继续学习

- 标注数据集的格式：voc、coco、yolo 等
- 由 `.json` 文件到 `.png` 文件的转换实现
- 复杂数据集的批量数据读取
