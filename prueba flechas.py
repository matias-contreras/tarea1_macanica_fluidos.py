import numpy as np
import matplotlib.pyplot as plt

V = np.array([[1,1], [-2,2], [4,-7]])
origin = np.array([[1, 2, 0],[1, 2, 0]]) # origin point
x = np.linspace(-4, 4, 50)
y = x + 1

plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=10)
plt.plot(x, y)
plt.show()