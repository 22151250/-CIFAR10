[//]: # (Image References)

[image1]: ./images/tsne.png "t-SNE"

# Keras Transfer Learning on CIFAR-10

在Jupyter notebook中, 我先计算bottleneck features，针对数据集[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

这些特征通过可视化工具 t-SNE展示。[Barnes-Hut implementation of t-SNE](http://lvdmaaten.github.io/tsne/)

对于 Python >= 3.5,安装以下工具

```
pip install git+https://github.com/alexisbcook/tsne.git
```
 利用bottleneck features和CNN进行图像分类， 模型准确率82.68%.

![t-SNE][image1]

## 报告
1.数据集 使用 cifar-10数据集，由几万张微小的彩色图像组成，每个图像描绘了来自十个不同类别之一的对象。

2.数据集的加载 
from keras.datasets import cifar10 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

3.构造模型 
可以利用Keras访问训练好的CNN模型，因此，没有从头开始构建CNN。
使用迁移学习，利用在分类任务中表现比较良好性能的CNN，我选择使用InceptionV3模型，

from keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=True)

训练好的InceptionV3存储在变量base_model中。
原来网络最后一层是全连接层，用于区分ImageNet数据库中的1000个不同对象类别。删除最后一层，并将生成的网络保存在一个新模型中。

model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output) 

新模型将不再返回预测的图像类别；然而，现在存储的CNNmodel提供了一种从图像中提取特征的方法。通过将每张 CIFAR-10 图像传递给该模型，可以将每张图像从其 32x32x3 的原始图像像素阵列转换为具有2048个维度的目的向量。

4.可视化结果 为了可视化瓶颈特征，使用了t-SNE（又名 t-Distributed Stochastic Neighbor Embedding）的降维技术。t-SNE 降低了每个点的维数，其方式是低维空间中的点保留与原始高维空间的逐点距离。
Scikit-learn有一个t-SNE的实现，但它不能很好地扩展到大型数据集。 
在github上找到一个工具可以很好解决降维问题；利用终端安装。 可视化生成的二维点会产生结果图，其中的点是根据图像所对应的对象类别进行颜色编码。

