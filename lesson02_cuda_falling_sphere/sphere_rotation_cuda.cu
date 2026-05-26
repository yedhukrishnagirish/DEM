#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

__host__ __device__ Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ Vec3 operator*(double s, Vec3 v) {
    return {s * v.x, s * v.y, s * v.z};
}

__host__ __device__ Vec3 operator/(Vec3 v, double s) {
    return {v.x / s, v.y / s, v.z / s};
}

__host__ __device__ Vec3& operator+=(Vec3& a, Vec3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ double length(Vec3 v) {
    return sqrt(dot(v, v));
}

struct Sphere {
    Vec3 position;
    Vec3 velocity;
    Vec3 force;

    // Rotation-related values
    Vec3 angular_velocity;   // omega, rad/s
    Vec3 torque;             // tau
    Vec3 orientation;        // angle tracker, radians

    double radius = 0.1;
    double mass = 1.0;

    // Moment of inertia
    double inertia = 1.0;
};

struct ContactParams {
    double normal_stiffness = 20000.0;
    double normal_damping = 30.0;

    double tangential_damping = 8.0;
    double friction_coefficient = 0.4;
};

void check_cuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(result) << "\n";
        std::exit(1);
    }
}

__device__ void clear_force_and_torque(Sphere& sphere) {
    sphere.force = {0.0, 0.0, 0.0};
    sphere.torque = {0.0, 0.0, 0.0};
}

__device__ void apply_gravity(Sphere& sphere, double gravity) {
    sphere.force.y += -sphere.mass * gravity;
}

__device__ void apply_floor_contact(Sphere& sphere, ContactParams params) {
    const double floor_y = 0.0;
    const Vec3 normal = {0.0, 1.0, 0.0};

    // Sphere-floor overlap
    const double overlap = floor_y + sphere.radius - sphere.position.y;

    if (overlap <= 0.0) {
        return;
    }

    // Contact point is at bottom of sphere
    const Vec3 contact_arm = {0.0, -sphere.radius, 0.0};

    // Velocity at contact point:
    // v_contact = v_linear + omega cross r
    const Vec3 contact_velocity =
        sphere.velocity + cross(sphere.angular_velocity, contact_arm);

    const double normal_velocity = dot(contact_velocity, normal);

    double normal_force_magnitude =
        params.normal_stiffness * overlap -
        params.normal_damping * normal_velocity;

    if (normal_force_magnitude <= 0.0) {
        return;
    }

    const Vec3 normal_force = normal_force_magnitude * normal;

    // Tangential velocity
    const Vec3 tangential_velocity =
        contact_velocity - normal_velocity * normal;

    Vec3 tangential_force = {0.0, 0.0, 0.0};

    const double tangential_speed = length(tangential_velocity);

    if (tangential_speed > 1.0e-12) {
        // Basic tangential damping/friction
        Vec3 raw_friction =
            -params.tangential_damping * tangential_velocity;

        // Coulomb limit: |Ft| <= mu * Fn
        const double max_friction =
            params.friction_coefficient * normal_force_magnitude;

        const double raw_size = length(raw_friction);

        if (raw_size > max_friction) {
            raw_friction = (max_friction / raw_size) * raw_friction;
        }

        tangential_force = raw_friction;
    }

    sphere.force += normal_force;
    sphere.force += tangential_force;

    // Torque = r cross F
    sphere.torque += cross(contact_arm, tangential_force);
}

__device__ void integrate_translation(Sphere& sphere, double dt) {
    const Vec3 acceleration = sphere.force / sphere.mass;

    sphere.velocity = sphere.velocity + dt * acceleration;
    sphere.position = sphere.position + dt * sphere.velocity;
}

__device__ void integrate_rotation(Sphere& sphere, double dt) {
    const Vec3 angular_acceleration = sphere.torque / sphere.inertia;

    sphere.angular_velocity =
        sphere.angular_velocity + dt * angular_acceleration;

    sphere.orientation =
        sphere.orientation + dt * sphere.angular_velocity;
}

__global__ void step_kernel(Sphere* spheres,
                            int sphere_count,
                            ContactParams contact,
                            double gravity,
                            double dt) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= sphere_count) {
        return;
    }

    Sphere& sphere = spheres[index];

    clear_force_and_torque(sphere);

    apply_gravity(sphere, gravity);

    apply_floor_contact(sphere, contact);

    integrate_translation(sphere, dt);

    integrate_rotation(sphere, dt);
}

int main() {
    constexpr int sphere_count = 10;

    Sphere host_spheres[sphere_count];

    for (int i = 0; i < sphere_count; ++i) {
        host_spheres[i].position = {-0.9 + 0.2 * i, 1.0 + 0.08 * i, 0.0};

        // IMPORTANT:
        // vy = 0.0 means it will NOT go upward first.
        // Gravity will pull it downward.
        host_spheres[i].velocity = {1.5, 0.0, 0.0};

        // Initial spin around z-axis
        host_spheres[i].angular_velocity = {0.0, 0.0, 15.0};

        host_spheres[i].force = {0.0, 0.0, 0.0};
        host_spheres[i].torque = {0.0, 0.0, 0.0};
        host_spheres[i].orientation = {0.0, 0.0, 0.0};

        host_spheres[i].radius = 0.08;
        host_spheres[i].mass = 1.0;

        // Solid sphere inertia:
        // I = 2/5 m r^2
        host_spheres[i].inertia =
            0.4 *
            host_spheres[i].mass *
            host_spheres[i].radius *
            host_spheres[i].radius;
    }

    Sphere* device_spheres = nullptr;

    check_cuda(
        cudaMalloc(&device_spheres, sizeof(host_spheres)),
        "cudaMalloc failed"
    );

    check_cuda(
        cudaMemcpy(
            device_spheres,
            host_spheres,
            sizeof(host_spheres),
            cudaMemcpyHostToDevice
        ),
        "copy CPU to GPU failed"
    );

    ContactParams contact;

    const double gravity = 9.81;
    const double dt = 1.0e-4;
    const double end_time = 2.0;
    const int steps = static_cast<int>(end_time / dt);

    std::ofstream csv("sphere_rotation_cuda.csv");

    csv << "time,id,"
        << "x,y,z,"
        << "vx,vy,vz,"
        << "omega_x,omega_y,omega_z,"
        << "angle_x,angle_y,angle_z,"
        << "force_x,force_y,force_z,"
        << "torque_x,torque_y,torque_z\n";

    const int threads_per_block = 128;
    const int blocks =
        (sphere_count + threads_per_block - 1) / threads_per_block;

    for (int step = 0; step <= steps; ++step) {
        if (step % 100 == 0) {
            check_cuda(
                cudaMemcpy(
                    host_spheres,
                    device_spheres,
                    sizeof(host_spheres),
                    cudaMemcpyDeviceToHost
                ),
                "copy GPU to CPU failed"
            );

            const double time = step * dt;

            for (int i = 0; i < sphere_count; ++i) {
                const Sphere& sphere = host_spheres[i];

                csv << time << ","
                    << i << ","

                    << sphere.position.x << ","
                    << sphere.position.y << ","
                    << sphere.position.z << ","

                    << sphere.velocity.x << ","
                    << sphere.velocity.y << ","
                    << sphere.velocity.z << ","

                    << sphere.angular_velocity.x << ","
                    << sphere.angular_velocity.y << ","
                    << sphere.angular_velocity.z << ","

                    << sphere.orientation.x << ","
                    << sphere.orientation.y << ","
                    << sphere.orientation.z << ","

                    << sphere.force.x << ","
                    << sphere.force.y << ","
                    << sphere.force.z << ","

                    << sphere.torque.x << ","
                    << sphere.torque.y << ","
                    << sphere.torque.z << "\n";
            }
        }

        step_kernel<<<blocks, threads_per_block>>>(
            device_spheres,
            sphere_count,
            contact,
            gravity,
            dt
        );

        check_cuda(cudaGetLastError(), "step_kernel failed");
    }

    check_cuda(cudaDeviceSynchronize(), "CUDA sync failed");

    check_cuda(
        cudaMemcpy(
            host_spheres,
            device_spheres,
            sizeof(host_spheres),
            cudaMemcpyDeviceToHost
        ),
        "final copy GPU to CPU failed"
    );

    std::cout << "Final sphere 0 y: "
              << host_spheres[0].position.y << "\n";

    std::cout << "Final sphere 0 vy: "
              << host_spheres[0].velocity.y << "\n";

    std::cout << "Final sphere 0 omega_z: "
              << host_spheres[0].angular_velocity.z << "\n";

    std::cout << "Final sphere 0 angle_z: "
              << host_spheres[0].orientation.z << "\n";

    std::cout << "Simulated spheres: "
              << sphere_count << "\n";

    std::cout << "Wrote sphere_rotation_cuda.csv\n";

    check_cuda(cudaFree(device_spheres), "cudaFree failed");

    return 0;
}