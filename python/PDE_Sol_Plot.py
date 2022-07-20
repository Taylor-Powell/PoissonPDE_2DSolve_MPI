"""


Author: T. Powell
Date:   May 16, 2022


"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np

color_map = cm.get_cmap('bone')

# *** Manually change these values ***
Nx = 51
Ny = 51
file = open('Poisson_Sol_Jacobi_Nx_51_Ny_51.dat', "rb")

# Import data and put into pandas dataframe
data = (np.array(np.fromfile(file, dtype=np.double, offset=0))).reshape(Ny*Nx,3)
df = pd.DataFrame(data, columns = ['x', 'y', 'z'])

# Create a 2D mesh from unique x and y values
x1 = np.linspace(df['x'].min(),df['x'].max(), len(df['x'].unique()))
y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
x2, y2 = np.meshgrid(x1, y1)
z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='cubic')


# Plotting
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=color_map,
                        linewidth=0, antialiased=False)
ax.set_zlim(0.0, 0.5)
ax.zaxis.set_major_locator(LinearLocator(6))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#fig.colorbar(surf, shrink=0.6, aspect=7.5)
plt.title(r'Solution of Poisson Eqn w/ Dirichlet BCs')

plt.show()