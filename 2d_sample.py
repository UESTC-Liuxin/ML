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
