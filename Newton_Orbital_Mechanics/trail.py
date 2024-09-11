import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 6.67430e-11  
dt = 500  
num_steps = 40000  

class CelestialBody:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.positions = np.zeros((num_steps, 2))
        self.positions[0] = self.position

    def update_position(self, position, step):
        self.position = np.array(position)
        self.positions[step] = self.position
    
    def update_velocity(self, velocity):
        self.velocity = np.array(velocity)

class OrbitSimulation:
    def __init__(self, body1, body2):
        self.body1 = body1  
        self.body2 = body2  
        self.axis_crossing_count = 0
        self.x_axis_crossings = []
        self.eccentricity = None

    def compute_gravitational_force(self, pos1, pos2, mass1, mass2):
        r = pos2 - pos1
        distance = np.linalg.norm(r)
        force_magnitude = G * mass1 * mass2 / distance**2
        force_direction = r / distance
        return force_magnitude * force_direction

    def acceleration(self, pos1, pos2, mass_other):
        force = self.compute_gravitational_force(pos1, pos2, self.body1.mass, self.body2.mass)
        return force / mass_other

    def rk4_step(self):
        b1, b2 = self.body1, self.body2

        k1_v_b1 = -self.acceleration(b1.position, b2.position, b1.mass) * dt
        k1_x_b1 = b1.velocity * dt

        k2_v_b1 = -self.acceleration(b1.position + k1_x_b1 / 2, b2.position, b1.mass) * dt
        k2_x_b1 = (b1.velocity + k1_v_b1 / 2) * dt

        k3_v_b1 = -self.acceleration(b1.position + k2_x_b1 / 2, b2.position, b1.mass) * dt
        k3_x_b1 = (b1.velocity + k2_v_b1 / 2) * dt

        k4_v_b1 = -self.acceleration(b1.position + k3_x_b1, b2.position, b1.mass) * dt
        k4_x_b1 = (b1.velocity + k3_v_b1) * dt

        vel_b1_next = b1.velocity + (k1_v_b1 + 2*k2_v_b1 + 2*k3_v_b1 + k4_v_b1) / 6
        pos_b1_next = b1.position + (k1_x_b1 + 2*k2_x_b1 + 2*k3_x_b1 + k4_x_b1) / 6

        k1_v_b2 = self.acceleration(b2.position, b1.position, b2.mass) * dt
        k1_x_b2 = b2.velocity * dt

        k2_v_b2 = self.acceleration(b2.position + k1_x_b2 / 2, b1.position, b2.mass) * dt
        k2_x_b2 = (b2.velocity + k1_v_b2 / 2) * dt

        k3_v_b2 = self.acceleration(b2.position + k2_x_b2 / 2, b1.position, b2.mass) * dt
        k3_x_b2 = (b2.velocity + k2_v_b2 / 2) * dt

        k4_v_b2 = self.acceleration(b2.position + k3_x_b2, b1.position, b2.mass) * dt
        k4_x_b2 = (b2.velocity + k3_v_b2) * dt

        vel_b2_next = b2.velocity + (k1_v_b2 + 2*k2_v_b2 + 2*k3_v_b2 + k4_v_b2) / 6
        pos_b2_next = b2.position + (k1_x_b2 + 2*k2_x_b2 + 2*k3_x_b2 + k4_x_b2) / 6

        return pos_b1_next, vel_b1_next, pos_b2_next, vel_b2_next

    def simulate(self):
        for step in range(1, num_steps):
            pos_earth, vel_earth, pos_moon, vel_moon = self.rk4_step()

            #! FIXED EARTH'S POS
            #* self.body1.update_position(pos_earth, step) 
            #* self.body1.update_velocity(vel_earth)
            self.body2.update_position(pos_moon, step)
            self.body2.update_velocity(vel_moon)

            self.check_axis_crossing(step)

    def check_axis_crossing(self, step):
        if self.body2.positions[step-1, 1] * self.body2.position[1] <= 0:
            self.x_axis_crossings.append([np.linalg.norm(self.body2.position), np.linalg.norm(self.body1.position)])
            self.axis_crossing_count += 1

        if self.axis_crossing_count == 2:
            self.calculate_eccentricity()

    def calculate_eccentricity(self):
        x0, y0 = self.x_axis_crossings[0]
        x1, y1 = self.x_axis_crossings[1]

        r_a = max(abs(x0), abs(x1))
        r_p = min(abs(x0), abs(x1))

        self.eccentricity = (r_a - r_p) / (r_a + r_p)
        print(f"Eccentricity: {self.eccentricity}")

        self.axis_crossing_count = 0
        self.x_axis_crossings = []

class AnimationPlot: 
    def __init__(self, body1, body2):
        self.body1 = body1
        self.body2 = body2
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ph, = self.ax.plot([], [], label='Earth')
        self.pm, = self.ax.plot([], [], label='Moon')
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_xlim(-2e11, 2e11)
        self.ax.set_ylim(-2e11, 2e11)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
        plt.title('Earth-Moon Orbit Simulation')
        plt.grid(True)
        plt.axis('equal')

    def update_frame(self, i):
        self.ph.set_data(self.body1.positions[:i, 0], self.body1.positions[:i, 1])
        self.pm.set_data(self.body2.positions[:i, 0], self.body2.positions[:i, 1])
        return self.ph, self.pm

    def animate(self):
        ani = FuncAnimation(self.fig, self.update_frame, frames=num_steps, interval=0.001, blit=True)
        plt.show()

    def save_positions(self):
        np.savetxt('Newton_Orbital_Mechanics/positions_earth.txt', self.body1.positions, header='x y', fmt='%.6f')
        np.savetxt('Newton_Orbital_Mechanics/positions_moon.txt', self.body2.positions, header='x y', fmt='%.6f')


earth = CelestialBody(mass=1.9891e30, position=[0.0, 0.0], velocity=[0.0, 0.0])
moon = CelestialBody(mass=5.972e24, position=[150750000000.0, 0.0], velocity=[0.0, 30000.0])

simulation = OrbitSimulation(earth, moon)

simulation.simulate()

animation = AnimationPlot(earth, moon)
animation.animate()

animation.save_positions()
