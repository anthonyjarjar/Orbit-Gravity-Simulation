import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

G = 6.67430e-11  
M = 1.989e4     
c = 299792458
G = 6.67430e-11  
M = 1.989e30     
c = 299792458    
L = 1e10  
mu = 1     

# Define the relativistic gravitational acceleration function
def CorrectedNewtonianMetric(G, M, r, c, L, mu):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.maximum(r, 1e6)  
        V_newtonian = -(G * M) / r
        V_centrifugal = (L**2) / (2 * mu * r**2)
        V_relativistic = -(G * (M + mu) * L**2) / (c**2 * mu * r**3)
        metric = V_newtonian + V_centrifugal + V_relativistic
    return metric


# Meshgrid for Cartesian coordinates
x, y = np.meshgrid(np.linspace(-1e7, 1e7, 200),  
                   np.linspace(-1e7, 1e7, 200))

# Convert Cartesian coordinates to polar coordinates
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Avoid the singularity at the origin by setting a minimum radius
r[r == 0] = 1e3 

# Directional vectors in polar coordinates
g1 = CorrectedNewtonianMetric(G, M, r, c, L, mu) * np.cos(theta)
g2 = CorrectedNewtonianMetric(G, M, r, c, L, mu) * np.sin(theta)


magnitude = np.sqrt(g1**2 + g2**2)


# Plotting the magnitude as a heatmap with logarithmic scaling
plt.imshow(magnitude, extent=[-1e7, 1e7, -1e7, 1e7], origin='lower', cmap='plasma', 
           norm=LogNorm(vmin=np.min(magnitude[magnitude > 0]), vmax=np.max(magnitude)))

plt.title('Relativistic Gravitational Field with Logarithmic Scaling')
plt.colorbar(label='Vector Magnitude (log scale)')

plt.grid()
plt.show()
