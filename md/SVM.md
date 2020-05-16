# SVM学习笔记

## 前言

SVM主要的思想就是寻找一个超平面，对样本能够进行有效的划分，达到这里需要补充一些向量与空间的知识（以前学的差不多已经还给老师了。）

### 平面的向量表示

这里有一个知乎的链接，总结得比较到位：[直线与平面的向量表示](https://zhuanlan.zhihu.com/p/73397884)

3维空间中的平面由4个参数确定，通常的表现形式为：$Ax+By+Cz+D=0$,确定平面的过程就是在确定四个参数的过程。当然这么说不是很准确，因为这是一个线性齐次方程组，所谓而基础解系就是平面方程的法向量+D参数，所以ABCD是不是唯一解，基础解系乘以一个非0常数仍然是这个平面方程。通常说三点不共线就可以去确定一个平面，其实就是三个点不共线，那么，系数矩阵的秩为3，基础解系就一个向量，也就是法向量，如果共线，系数矩阵的秩为2，基础解系就有2个向量，这两个向量线性组成的向量空间，向量并不平行，就不是平面的法向量了。

用matplotlib画平面：

```python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import numpy as np

fig = plt.figure()
ax = Axes3D(fig)

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y= np.meshgrid(X, Y)
w1=1
w2=2
w3=2
b1=10
b2=0
# w1x+w2y+w3z=b
Z = (b1-(w1*X+w2*Y))/w3
Z1 = (b2-(w1*X+w2*Y))/w3
# Plot the surface.
surf = ax.plot_surface(X, Y, Z,
                       linewidth=0, antialiased=False)
surf1 = ax.plot_surface(X, Y, Z1,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r'$X$', fontsize=20, color='blue')
ax.set_ylabel(r'$Y$', fontsize=20, color='blue')
ax.set_zlabel(r'$Z$', fontsize=20, color='blue')

plt.show()
```

对于超平面，就是4维及其以上的平面。那么公式表示为：$w^{T} x+b=0, w=(A, B, C \ldots .)$ 法向量



### 点到平面的距离

同样摆上大佬的知乎链接：[点到直线与面的距离公式](https://zhuanlan.zhihu.com/p/63499708)

首先放上一个比较经典的图：

![](https://pic1.zhimg.com/80/v2-00a47c670dc37811f1c9d0f9df764e14_720w.jpg)

点到平面的距离实际上就是A点与平面上任一点B的距离在法向量上的投影，



$$
{|\mathrm{AC}|=\left|\overrightarrow{\mathrm{AB}} \cdot \frac{\vec{n}}{|\vec{n}|}\right|=\left|\left(x_{1}-x_{0}, y_{1}-y_{0}, z_{1}-z_{0}\right) \cdot \frac{(a, b, c)}{\sqrt{a^{2}+b^{2}+c^{2}}}\right|}\\{=\frac{1}{\sqrt{a^{2}+b^{2}+c^{2}}}\left|a\left(x_{1}-x_{0}\right)+b\left(y_{1}-y_{0}\right)+c\left(z_{1}-z_{0}\right)\right|}\\=\frac{\left|a x_{0}+b y_{0}+c z_{0}+d\right|}{\sqrt{a^{2}+b^{2}+c^{2}}}
$$

同理，多维空间进行类似推广。

## SVM的基本型

对于一个样本集合。

```python
import numpy as np
import matplotlib.pyplot as plt

X0=np.random.normal(-2,1,50)#标准正态分布
X1=np.random.normal(2,1,50)#标准正态分布

Y0=np.random.normal(2,1,50)#标准正态分布
Y1=np.random.normal(-2,1,50)#标准正态分布
fig=plt.Figure()
plt.scatter(X0,Y0)
plt.scatter(X1,Y1)
# plt.scatter(X1,Y1)
plt.show()
```

