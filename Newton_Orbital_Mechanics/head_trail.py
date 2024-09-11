import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 6.67430e-11
dt = 500 
num_steps = 40000  
tolerance = 1e-6
axis_crossing_count = 0

mass_earth, mass_moon = 1.9891e30, 5.972e24 
pos_earth, pos_moon = np.array([0.0, 0.0]), np.array([150750000000.0, 0.0]) 
vel_earth, vel_moon = np.array([0.0, 0.0]), np.array([0.0, 30000.0])


def compute_gravitational_force(pos_earth, pos_moon, mass_earth, mass_moon):
    r = pos_moon - pos_earth
    distance = np.linalg.norm(r)
    force_magnitude = G * mass_earth * mass_moon / distance**2
    force_direction = r / distance
    return force_magnitude * force_direction

def rk4_step(pos_moon, vel_moon, pos_earth, vel_earth, dt):
    def acceleration(pos_1, pos_2, mass_other):
        force = compute_gravitational_force(pos_1, pos_2, mass_earth, mass_moon)
        return force / mass_other

    k1_v_moon = acceleration(pos_moon, pos_earth, mass_moon) * dt
    k1_x_moon = vel_moon * dt

    k2_v_moon = acceleration(pos_moon + k1_x_moon / 2, pos_earth, mass_moon) * dt
    k2_x_moon = (vel_moon + k1_v_moon / 2) * dt

    k3_v_moon = acceleration(pos_moon + k2_x_moon / 2, pos_earth, mass_moon) * dt
    k3_x_moon = (vel_moon + k2_v_moon / 2) * dt

    k4_v_moon = acceleration(pos_moon + k3_x_moon, pos_earth, mass_moon) * dt
    k4_x_moon = (vel_moon + k3_v_moon) * dt

    vel_moon_next = vel_moon + (k1_v_moon + 2*k2_v_moon + 2*k3_v_moon + k4_v_moon) / 6
    pos_moon_next = pos_moon + (k1_x_moon + 2*k2_x_moon + 2*k3_x_moon + k4_x_moon) / 6

    k1_v_earth = -acceleration(pos_earth, pos_moon, mass_earth) * dt
    k1_x_earth = vel_earth * dt

    k2_v_earth = -acceleration(pos_earth + k1_x_earth / 2, pos_moon, mass_earth) * dt
    k2_x_earth = (vel_earth + k1_v_earth / 2) * dt

    k3_v_earth = -acceleration(pos_earth + k2_x_earth / 2, pos_moon, mass_earth) * dt
    k3_x_earth = (vel_earth + k2_v_earth / 2) * dt

    k4_v_earth = -acceleration(pos_earth + k3_x_earth, pos_moon, mass_earth) * dt
    k4_x_earth = (vel_earth + k3_v_earth) * dt

    vel_earth_next = vel_earth + (k1_v_earth + 2*k2_v_earth + 2*k3_v_earth + k4_v_earth) / 6
    pos_earth_next = pos_earth + (k1_x_earth + 2*k2_x_earth + 2*k3_x_earth + k4_x_earth) / 6

    return pos_moon_next, vel_moon_next, pos_earth_next, vel_earth_next

positions_earth = np.zeros((num_steps, 2))
positions_moon = np.zeros((num_steps, 2))

positions_earth[0] = pos_earth
positions_moon[0] = pos_moon
x_axis_crossings = []

for step in range(1, num_steps):
    pos_moon, vel_moon, pos_earth, vel_earth = rk4_step(pos_moon, vel_moon, pos_earth, vel_earth, dt)

    positions_earth[step] = pos_earth
    positions_moon[step] = pos_moon

    if (positions_moon[step-1, 1] * pos_moon[1] <= 0): 
        x_axis_crossings.append([np.linalg.norm(pos_moon), np.linalg.norm(pos_earth)])
        axis_crossing_count += 1

    if axis_crossing_count == 2:
        x0, y0 = x_axis_crossings[0]
        x1, y1 = x_axis_crossings[1]

        r_a = max(abs(x0), abs(x1))
        r_p = min(abs(x0), abs(x1))

        eccentricity = (r_a - r_p) / (r_a + r_p)
        print(f"Eccentricity: {eccentricity}")

        axis_crossing_count = 0
        x_axis_crossings = []

def update_frame(i):
    ph.set_data(positions_earth[:i, 0], positions_earth[:i, 1])
    pm.set_data(positions_moon[:i, 0], positions_moon[:i, 1])

    head_earth.set_data(positions_earth[i, 0], positions_earth[i, 1])
    head_moon.set_data(positions_moon[i, 0], positions_moon[i, 1])

    return ph, pm, head_earth, head_moon


fig, ax = plt.subplots(figsize=(8, 8))
ph, = ax.plot(positions_earth[:, 0], positions_earth[:, 1], 'b-', alpha=0.15, lw=2, label='Earth Tail')
pm, = ax.plot(positions_moon[:, 0], positions_moon[:, 1], 'r-', alpha=0.15, lw=2, label='Moon Tail')

head_earth, = ax.plot(positions_earth[:, 0], positions_earth[:, 1], 'bo', markersize=3, label='Earth Head')
head_moon, = ax.plot(positions_moon[:, 0], positions_moon[:, 1], 'ro', markersize=2, label='Moon Head')

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.title('Earth-Moon Orbit Simulation')
plt.grid(True)
plt.axis('equal')  

ani = FuncAnimation(fig, update_frame, frames=num_steps, interval=0.01, blit=False)

ani.save("orbit_simulation.mp4", writer='ffmpeg', fps=30) 

plt.show()

np.savetxt('positions_earth.txt', positions_earth, header='x y', fmt='%.6f')
np.savetxt('positions_moon.txt', positions_moon, header='x y', fmt='%.6f')