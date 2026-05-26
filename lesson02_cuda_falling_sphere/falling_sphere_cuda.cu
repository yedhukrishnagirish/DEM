#include <cuda_runtime.h>
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
__host__ __device__ Vec3 operator*(double s, Vec3 v) {
    return {s * v.x, s * v.y, s * v.z};
}
__host__ __device__ Vec3 operator/(Vec3 v, double s) {
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
void check_cuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(result) << "\n";
        std::exit(1);
    }
}
__device__ void clear_force(Sphere& sphere) {
    sphere.force = {0.0, 0.0, 0.0};
}
__device__ void apply_gravity(Sphere& sphere, double gravity) {
    sphere.force.y += -sphere.mass * gravity;
}
__device__ void apply_floor_contact(Sphere& sphere, ContactParams params) {
    const double floor_y = 0.0;
    const double overlap = floor_y + sphere.radius - sphere.position.y;
    if (overlap <= 0.0) {
        return;
    }
    const double normal_velocity = sphere.velocity.y;
    const double normal_force = params.normal_stiffness * overlap -
                                params.normal_damping * normal_velocity;
    if (normal_force > 0.0) {
        sphere.force.y += normal_force;
    }
}
__device__ void integrate_symplectic_euler(Sphere& sphere, double dt) {
    const Vec3 acceleration = sphere.force / sphere.mass;
    sphere.velocity = sphere.velocity + dt * acceleration;
    sphere.position = sphere.position + dt * sphere.velocity;
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
    clear_force(sphere);
    apply_gravity(sphere, gravity);
    apply_floor_contact(sphere, contact);
    integrate_symplectic_euler(sphere, dt);
}
int main() {
    constexpr int sphere_count = 1;

    Sphere host_spheres[sphere_count];
    host_spheres[0].position = {0.0, 1.0, 0.0};
    host_spheres[0].velocity = {0.0, 0.0, 0.0};
    host_spheres[0].radius = 0.1;
    host_spheres[0].mass = 1.0;

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

    std::ofstream csv("falling_sphere_cuda.csv");
    csv << "time,y,vy,force_y\n";

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
            const Sphere& sphere = host_spheres[0];

            csv << time << ","
                << sphere.position.y << ","
                << sphere.velocity.y << ","
                << sphere.force.y << "\n";
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

    std::cout << "Final y: " << host_spheres[0].position.y << "\n";
    std::cout << "Final vy: " << host_spheres[0].velocity.y << "\n";
    std::cout << "Wrote falling_sphere_cuda.csv\n";

    cudaFree(device_spheres);

    return 0;
}