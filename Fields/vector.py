import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

G = 6.67430e-11  
M = 1.989e30     
c = 299792458    
L = 1e10  
mu = 1 

def CorrectedNewtonianMetric(G, M, r, c, L, mu):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.maximum(r, 1e6)  
        V_newtonian = -(G * M) / r
        V_centrifugal = (L**2) / (2 * mu * r**2)
        V_relativistic = -(G * (M + mu) * L**2) / (c**2 * mu * r**3)
        metric = V_newtonian + V_centrifugal + V_relativistic
    return metric

x, y = np.meshgrid(np.linspace(-1e7, 1e7, 25),  
                   np.linspace(-1e7, 1e7, 25))

r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

r[r < 2e6] = np.nan  

g_r = CorrectedNewtonianMetric(G, M, r, c, L, mu)

gx = g_r * (x / r) 
gy = g_r * (y / r) 

gx[np.isnan(gx)] = 0
gy[np.isnan(gy)] = 0

magnitude = np.sqrt(gx**2 + gy**2)

plt.quiver(x, y, gx, gy, magnitude, cmap='plasma', linewidth=2, 
           headwidth=5, headlength=7, headaxislength=5)  

plt.title('Relativistic Gravitational Field (Post-Newtonian Correction)')
plt.colorbar(label='Vector Magnitude')

plt.grid()
plt.show()
