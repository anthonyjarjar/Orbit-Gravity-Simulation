import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

G = 6.67430e-11  
M = 1.989e30     
c = 299792458    

def sMetric(G, M, r, c):
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero or NaNs
        r = np.maximum(r, 1e6)
        metric = (-(G * M) / (r**2)) * (1 - (2 * G * M) / (c**2 * r))**(-1/2)
    return metric

x, y = np.meshgrid(np.linspace(-1e7, 1e7, 30),  
                   np.linspace(-1e7, 1e7, 30))

# Convert Cartesian coordinates to polar coordinates
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Avoid the singularity at the origin by excluding values near the center
r[r < 2e6] = np.nan  # Exclude the region close to the origin 

# Directional vectors in polar coordinates (radial only)
g_r = sMetric(G, M, r, c)

# Convert polar vectors to Cartesian vectors
gx = g_r * (x / r) 
gy = g_r * (y / r) 

# remove any nan
gx[np.isnan(gx)] = 0
gy[np.isnan(gy)] = 0

magnitude = np.sqrt(gx**2 + gy**2)

plt.quiver(x, y, gx, gy, magnitude, cmap='plasma',linewidth=2, headwidth=5, headlength=7, headaxislength=5)  

plt.title('Relativistic Gravitational Field (Excluding Center)')
plt.colorbar(label='Vector Magnitude')

plt.grid()
plt.show()
