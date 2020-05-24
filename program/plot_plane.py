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
ax.set_xlabel(r'$x_1$', fontsize=20, color='blue')
ax.set_ylabel(r'$x_2$', fontsize=20, color='blue')
ax.set_zlabel(r'$x_3$', fontsize=20, color='blue')

plt.show()