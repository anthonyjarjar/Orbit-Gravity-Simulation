#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

const double G = 6.67430e-11;  
const double TIMESTEP = 1000; 
const double SOFTENING = 1e9; 
const int LOG_INTERVAL = 1000; 


const double AU = 1.496e11;  
const double YEAR = 3.154e7;

struct Body {
    double x, y;  
    double vx, vy; 
    double mass;   
    
    Body(double x, double y, double vx, double vy, double mass)
        : x(x), y(y), vx(vx), vy(vy), mass(mass) {}
};

std::pair<double, double> computeAcceleration(const Body &b1, const Body &b2) {
    double dx = b2.x - b1.x;
    double dy = b2.y - b1.y;
    double dist = std::sqrt(dx * dx + dy * dy + SOFTENING * SOFTENING);
    double force = (G * b1.mass * b2.mass) / (dist * dist);
    double ax = force * dx / dist / b1.mass;
    double ay = force * dy / dist / b1.mass;
    return {ax, ay};
}

void rungeKuttaUpdate(Body &body, const std::vector<Body> &bodies, bool isSun) {
    if (isSun) return;

    double k1_vx = body.vx;
    double k1_vy = body.vy;
    double k1_ax = 0, k1_ay = 0;

    for (const auto &other : bodies) {
        if (&body != &other) {
            auto [ax, ay] = computeAcceleration(body, other);
            k1_ax += ax;
            k1_ay += ay;
        }
    }

    double k2_vx = body.vx + k1_ax * (TIMESTEP / 2);
    double k2_vy = body.vy + k1_ay * (TIMESTEP / 2);
    double k2_ax = 0, k2_ay = 0;
    for (const auto &other : bodies) {
        if (&body != &other) {
            Body temp(body.x + k1_vx * (TIMESTEP / 2), body.y + k1_vy * (TIMESTEP / 2), k2_vx, k2_vy, body.mass);
            auto [ax, ay] = computeAcceleration(temp, other);
            k2_ax += ax;
            k2_ay += ay;
        }
    }

    double k3_vx = body.vx + k2_ax * (TIMESTEP / 2);
    double k3_vy = body.vy + k2_ay * (TIMESTEP / 2);
    double k3_ax = 0, k3_ay = 0;
    for (const auto &other : bodies) {
        if (&body != &other) {
            Body temp(body.x + k2_vx * (TIMESTEP / 2), body.y + k2_vy * (TIMESTEP / 2), k3_vx, k3_vy, body.mass);
            auto [ax, ay] = computeAcceleration(temp, other);
            k3_ax += ax;
            k3_ay += ay;
        }
    }

    double k4_vx = body.vx + k3_ax * TIMESTEP;
    double k4_vy = body.vy + k3_ay * TIMESTEP;
    double k4_ax = 0, k4_ay = 0;
    for (const auto &other : bodies) {
        if (&body != &other) {
            Body temp(body.x + k3_vx * TIMESTEP, body.y + k3_vy * TIMESTEP, k4_vx, k4_vy, body.mass);
            auto [ax, ay] = computeAcceleration(temp, other);
            k4_ax += ax;
            k4_ay += ay;
        }
    }

    body.vx += (TIMESTEP / 6) * (k1_ax + 2 * k2_ax + 2 * k3_ax + k4_ax);
    body.vy += (TIMESTEP / 6) * (k1_ay + 2 * k2_ay + 2 * k3_ay + k4_ay);
    body.x += (TIMESTEP / 6) * (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx);
    body.y += (TIMESTEP / 6) * (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy);
}

int main() {
    std::vector<Body> bodies = {
        Body(0, 0, 0, 0, 1.989e30),

        Body(0.72 * AU, 0, 0, 35.02e3, 4.867e24),
        
        Body(1 * AU, 0, 0, 29.78e3, 5.972e24), 
        
        Body(5.2 * AU, 0, 0, 13.07e3, 1.898e27),
        
        Body(19.22 * AU, 0, 0, 6.80e3, 8.681e25)
    };

    std::ofstream file("solar_system_data.csv");  
    file << "Step,Body,PosX,PosY,VelX,VelY\n";  

    int total_steps = int(100 * YEAR / TIMESTEP);
    for (int step = 0; step < total_steps; ++step) {
        for (size_t i = 0; i < bodies.size(); ++i) {
            bool isSun = (i == 0);
            rungeKuttaUpdate(bodies[i], bodies, isSun);
        }

        if (step % LOG_INTERVAL == 0) {
            for (size_t i = 0; i < bodies.size(); ++i) {
                file << step << "," << i << "," << bodies[i].x / AU << "," << bodies[i].y / AU << "," 
                     << bodies[i].vx / 1e3 << "," << bodies[i].vy / 1e3 << "\n"; 
            }
        }
    }

    file.close();
    return 0;
}