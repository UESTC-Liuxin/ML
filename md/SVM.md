# SVM学习笔记

## 前言

SVM主要的思想就是寻找一个超平面，对样本能够进行有效的划分，达到这里需要补充一些向量与空间的知识（以前学的差不多已经还给老师了。）

### 平面的向量表示

这里有一个知乎的链接，总结得比较到位：[直线与平面的向量表示](https://zhuanlan.zhihu.com/p/73397884)

3维空间中的平面由4个参数确定，通常的表现形式为：$Ax+By+Cz+D=0$,确定平面的过程就是在确定四个参数的过程。当然这么说不是很准确，因为这是一个线性齐次方程组，所谓而基础解系就是平面方程的法向量+D参数，所以ABCD是不是唯一解，基础解系乘以一个非0常数仍然是这个平面方程。通常说三点不共线就可以去确定一个平面，其实就是三个点不共线，那么，系数矩阵的秩为3，基础解系就一个向量，也就是法向量，如果共线，系数矩阵的秩为2，基础解系就有2个向量，这两个向量线性组成的向量空间，向量并不平行，就不是平面的法向量了。

**用matplotlib画平面：**

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

### 拉格朗日乘子法与二次规划

对于一个带有约束条件的求极值问题，一个比较常用的方法就是拉格朗日乘子（数）法。[拉格朗日乘数法通俗理解](https://blog.csdn.net/THmen/article/details/87366904)

$F(x, y, \lambda)=f(x, y)+\lambda g(x, y)$这是一个典型的拉格朗日乘子法的标准形式，$f(x, y)$为目标函数，$g(x, y)$为限制条件，通常为0等式的表达式或0不等式的表达式。拉格朗日乘数法的核心思想就是，在约束条件下，目标函数的梯度方向与限制条件的梯度方向是平行的，$▽f=λ*▽g $ 

| **原问题：**              | **对偶问题：**                            |
| ------------------------- | ----------------------------------------- |
| $min f(x,y) ；s.t.g(x)=0$ | 由$▽f=λ*▽g得： fx=λ*gx， fy=λ*gy， g(x)=0$. |
| 约束优化问题              | 无约束方程组问题                          |

因此，求解这个对偶问题的方程组就行了。

然而这只是针对于约束条件为等式的情况下，但是，对于不等式情况下，还需要引入一个知识点，叫做**KKT条件**。


$$
\begin{aligned}
&\min _{x, y} f(x, y)\\
&\text {s.t.} \quad g_{i}(x, y) \leq 0, i=1,2, \cdots, N\\
&h_{i}(x, y)=0, i=1,2, \cdots, M
\end{aligned}
$$

这是一个约束条件为不等式的优化条件，按照之前的做法，无法带入一个方程组。[如何理解拉格朗日乘子法和KKT条件？](https://www.matongxue.com/madocs/987/)

具体的推导看上面的式子吧，我觉得我不可能讲得有这么好了。KKT条件如下所示，最优解必须满足：

$$
\left\{\begin{array}{l}\nabla f+\sum_{i}^{n} \lambda_{i} \nabla g_{i}+\sum_{j}^{m} \mu_{j} \nabla h_{j}=0 \\ h_{i} = 0, i=1,2, \cdots, m\\ g_{j} \leq 0, j=1,2, \cdots, n \\ \mu_{j} \geq 0 \\ \mu_{j} g_{j}=0\end{array}\right.
$$

大概意思就是，如果原$f(x,y)$本身的最优解满足$g(x,y)$，那么$\mu$就不起作用，就等于0；如果不满足，那么最优指一定是在$g(x,y)=0$上，这时一定有，f的法线方向与g的发现方向相反，必须有$\mu>0$才能满足$\nabla f+\mu \nabla g=0$。因此只需要求解以上方程组，就可得到最优值。

关于以上方程的求解，就涉及到了支持向量与二次规划的问题了。。。。未完待续 。。。

## SVM的基本型

对于一个样本集合。

<img src="https://pic3.zhimg.com/v2-197913c461c1953c30b804b4a7eddfcc_1200x500.jpg" style="zoom:67%;" />

需要寻找一个平面来对它进行划分，如果本身样本就是可划分的，那么一定存在一个超平面$WX+b=0$使得正负两个样本集使得 

$$
\left\{\begin{array}{ll}\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b >0 ,在超平面正法向量一侧\\ \boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b<0,在负法向量一侧\end{array}\right.
$$

由于可以通过w,b的伸缩，将变成1，设定标签$y_i=+-1$:

$$
\left\{\begin{array}{ll}\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \geqslant+1, & y_{i}=+1 \\ \boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \leqslant-1, & y_{i}=-1\end{array}\right.
$$

最终w，b的约束条件为：$y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m$.

为了找到两个样本集中的支持向量间的最大间隔，$\gamma=2r=r=2\frac{\left|\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\right|}{\|\boldsymbol{w}\|}=2\frac{1}{\|\boldsymbol{w}\|}$ ，问题就变成了：
$$
\begin{array}{cl}
\min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2} \\
\text { s.t. } & y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m
\end{array}
$$
这就是SVM的基本型。也就是SVM的分类依据。

## SVM优化的对偶问题

为了找到合适的w,b，我们需要进行最优化操作。这是一个凸二次规划问题，但利用拉格朗日乘子法，能够得到更高的效率。

根据拉格朗日乘子式：
$$
L(\boldsymbol{w}, b, \boldsymbol{\alpha})=\frac{1}{2}\|\boldsymbol{w}\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)
$$
在满足KKT条件的情况下，变成它的对偶问题：
$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}=\max _{\alpha_{i} \geq 0}\min _{\boldsymbol{w}, b}  L(\boldsymbol{w}, b, \boldsymbol{\alpha})
$$
关于这个对偶问题是如何形成的，这里还有一个写的很好帖子：[支持向量机（SVM）——原理篇](https://zhuanlan.zhihu.com/p/31886934)

为了得到求解对偶问题的具体形式，令 ![[公式]](https://www.zhihu.com/equation?tex=+L%5Cleft%28+%5Cboldsymbol%7Bw%2C%7Db%2C%5Cboldsymbol%7B%5Calpha+%7D+%5Cright%29+) 对 ![[公式]](https://www.zhihu.com/equation?tex=+%5Cboldsymbol%7Bw%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=+b+) 的偏导为0，可得

$$\begin{array}{l}
\boldsymbol{w}=\sum_{i=1}^{N} \alpha_{i} y_{i} \boldsymbol{x}_{i} \\
\sum_{i=1}^{N} \alpha_{i} y_{i}=0
\end{array}$$

将以上两个等式带入拉格朗日目标函数，消去 ![[公式]](https://www.zhihu.com/equation?tex=+%5Cboldsymbol%7Bw%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=+b+) ， 得
$$
\begin{array}{l}
\qquad \begin{aligned}
L(\boldsymbol{w}, b, \boldsymbol{\alpha})=& \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}\right)-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(\left(\sum_{j=1}^{N} \alpha_{j} y_{j} \boldsymbol{x}_{j}\right) \cdot \boldsymbol{x}_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i} \\
=&-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
\end{aligned} \\
即，
\min _{\boldsymbol{w}, b} L(\boldsymbol{w}, b, \boldsymbol{\alpha})=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
\end{array}\\\begin{aligned}
&\text {s.t.} \quad \sum_{i=1}^{N} \alpha_{i} y_{i}=0\\
&\alpha_{i} \geq 0, i=1,2, \ldots, N
\end{aligned}
$$
把目标式子加一个负号，将求解极大转换为求解极小
$$
\begin{array}{l}
\min _{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\
\text { s.t. } \quad \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
\quad \alpha_{i} \geq 0, i=1,2, \ldots, N
\end{array}
$$

## SMO算法

说实在的，我觉得这个算法的数学推导有点子复杂，我确实没看懂。[机器学习算法实践-SVM中的SMO算法](https://zhuanlan.zhihu.com/p/29212107)