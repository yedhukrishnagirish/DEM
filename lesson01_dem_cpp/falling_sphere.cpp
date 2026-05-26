#include <fstream>
#include <iostream>

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator*(double s, Vec3 v) {
    return {s * v.x, s * v.y, s * v.z};
}

Vec3 operator/(Vec3 v, double s) {
    return {v.x / s, v.y / s, v.z / s};
}

struct Sphere {
    Vec3 position;
    Vec3 velocity;
    Vec3 force;
    double radius = 0.1;
    double mass = 1.0;
};

struct ContactParams {
    double normal_stiffness = 20000.0;
    double normal_damping = 30.0;
};

void clear_force(Sphere& sphere) {
    sphere.force = {0.0, 0.0, 0.0};
}

void apply_gravity(Sphere& sphere, double gravity) {
    sphere.force.y += -sphere.mass * gravity;
}

void apply_floor_contact(Sphere& sphere, ContactParams params) {
    const double floor_y = 0.0;
    const double overlap = floor_y + sphere.radius - sphere.position.y;

    if (overlap <= 0.0) {
        return;
    }

    const double normal_velocity = sphere.velocity.y;
    const double normal_force =
        params.normal_stiffness * overlap -
        params.normal_damping * normal_velocity;

    if (normal_force > 0.0) {
        sphere.force.y += normal_force;
    }
}

void integrate_symplectic_euler(Sphere& sphere, double dt) {
    const Vec3 acceleration = sphere.force / sphere.mass;
    sphere.velocity = sphere.velocity + dt * acceleration;
    sphere.position = sphere.position + dt * sphere.velocity;
}

int main() {
    Sphere sphere;
    sphere.position = {0.0, 1.0, 0.0};

    ContactParams contact;

    const double gravity = 9.81;
    const double dt = 1.0e-4;
    const double end_time = 2.0;
    const int steps = static_cast<int>(end_time / dt);

    std::ofstream csv("falling_sphere.csv");
    csv << "time,y,vy,force_y\n";

    for (int step = 0; step <= steps; ++step) {
        const double time = step * dt;

        clear_force(sphere);
        apply_gravity(sphere, gravity);
        apply_floor_contact(sphere, contact);

        if (step % 100 == 0) {
            csv << time << ","
                << sphere.position.y << ","
                << sphere.velocity.y << ","
                << sphere.force.y << "\n";
        }

        integrate_symplectic_euler(sphere, dt);
    }

    std::cout << "Final y: " << sphere.position.y << "\n";
    std::cout << "Final vy: " << sphere.velocity.y << "\n";
    std::cout << "Wrote falling_sphere.csv\n";

    return 0;
}