import numpy as np
import matplotlib.pyplot as plt

X0=np.random.normal(-3,1,(50,2))#标准正态分布
X1=np.random.normal(3,1,(50,2))#标准正态分布


fig=plt.Figure()
plt.scatter(X0[:,:1],X0[:,1:])
plt.scatter(X1[:,:1],X1[:,1:])
plt.plot(np.arange(-5,5),-1*np.arange(-5,5),'r-')
# plt.scatter(X1,Y1)
plt.show()
