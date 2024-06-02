'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 - y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


x = y = np.arange(-10.0, 10.0, .1)
X, Y = np.meshgrid(x, y)

Z = f(X,Y) / 3

ax.plot_surface(X, Y, Z,color='gray',alpha=.5)

ax.set_zlim3d(-100, 100)
#To plot the surface at 100, use your same grid but make all your numbers zero

plt.axis('off')
plt.savefig('check.pdf')
plt.show()