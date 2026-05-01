#include "trace_kernel_utils.h"

// Quaternionic arithmetic functions.

// Calculate the Hamilton (quaternionic) product of two quaternions.
__device__ void quatProduct(float u[4], float v[4], float result[4])
{
    result[0] = u[0]*v[0] - (u[1]*v[1] + u[2]*v[2] + u[3]*v[3]);
    // Cross product of the vector components of u and v is needed.
    float cross[3];
    cross[0] = u[2]*v[3] - u[3]*v[2];
    cross[1] = u[3]*v[1] - u[1]*v[3];
    cross[2] = u[1]*v[2] - u[2]*v[1];
    #pragma unroll
    for (int i { 1 }; i < 4; i++)
    {
        result[i] = u[0]*v[i] + v[0]*u[i] + cross[i-1];
    }
}

// Rotates a 3D Cartesian vector, vec (a pure quaternion), by rotation_quat.
// result will be the rotated vector represented as a pure quaternion.
__device__ void rotateVecByQuat(float vec[4], float rotation_quat[4], float result[4])
{
    // Assume that rotation_quat is normalised; checking isn't worth the cost.
    float rotation_quat_inverse[4];
    rotation_quat_inverse[0] = rotation_quat[0];
    rotation_quat_inverse[1] = -rotation_quat[1];
    rotation_quat_inverse[2] = -rotation_quat[2];
    rotation_quat_inverse[3] = -rotation_quat[3];
    float intermediate_result[4];
    quatProduct(vec, rotation_quat_inverse, intermediate_result);
    quatProduct(rotation_quat, intermediate_result, result);
}

// Function definitions for Metric and its derived classes.

// Stores the calculated independent components in g.
// Flat/Minkowski spacetime; same everywhere.
__device__ void Metric::calculateMetric(float r[4], float g[4][4])
{
    g[0][0] = -1.; g[0][1] = 0.; g[0][2] = 0.; g[0][3] = 0.;
    g[1][0] = 0.; g[1][1] = 1.; g[1][2] = 0.; g[0][3] = 0.;
    g[2][0] = 0.; g[2][1] = 0.; g[2][2] = 1.; g[2][3] = 0.;
    g[3][0] = 0.; g[3][1] = 0.; g[3][2] = 0.; g[3][3] = 1.;
}

// Returns whether to terminate a photon passing through a point in this metric.
__device__ bool Metric::terminateRay(float r[4]){
    // Flat spacetime has no obvious termination condition.
    // Currently just measures whether the ray is beyond some radius.
    float radius { norm3df(r[1], r[2], r[3]) };
    return radius < 10.;
}

// Calculates the start velocity of a photon at pixel (x, y), where (0, 0) is the top-left corner of the camera.
// Overwrites result into v. Assumes Minkowski coordinates.
__device__ void Metric::calculateStartV(float x,
                                        float y,
                                        float g[4][4],
                                        float v[4],
                                        unsigned int d_cam_pixels[2],
                                        float d_cam_quat[4],
                                        float d_cam_fov_conv_factor)
{
    // Local phi and theta coordinates in the camera's reference frame.
    // Negative in phi because phi increases anticlockwise around the local z-axis.
    float phi { -((x - 0.5f*d_cam_pixels[0]) * (d_cam_fov_conv_factor)) };
    float theta { (y - 0.5f*d_cam_pixels[1]) * (d_cam_fov_conv_factor) + 0.5f*pi_device };
    // Minkowski/Cartesian coordinates.
    float unrotated_v[4];
    unrotated_v[0] = 0.;
    unrotated_v[1] = sin(theta)*cos(phi);
    unrotated_v[2] = sin(theta)*sin(phi);
    unrotated_v[3] = cos(theta);
    // Rotate to align with the camera's orientation in the global frame.
    rotateVecByQuat(unrotated_v, d_cam_quat, v);
    // Modify the t-component to make the velocity null.
    makeVNull(v, g);
}

// Make a velocity vector null (assuming Minkowski coordinates).
__device__ void Metric::makeVNull(float v[4], float g[4][4])
{
    float a { g[0][0] };
    float b { 0. };
    float c { 0. };

    #pragma unroll
    for (int i { 1 }; i < 4; i++)
    {
        b += g[0][i]*v[i];
    }
    b *= 2.;

    // Calculate c.
    for (int i { 1 }; i < 4; i++)
    {
        float contraction { 0. };
        #pragma unroll
        for (int j { 1 }; j < 4; j++)
        {
            contraction += g[i][j]*v[j];
        }
        c += contraction*v[i];
    }

    // Take the positive root solution. a = g_00 is usually negative, so this normally makes v[0] negative
    // in order to evolve the photon backwards from the camera. Makes no difference for static metrics.
    v[0] = (-b + sqrt(b*b - 4.*a*c)) / (2.*a);
}

// Calculates the scalar product of a velocity in some metric.
// Tries to use as little memory as possible; the goal
// is to minimize register occupancy, not computation.
__host__ __device__ float scalarProduct(float v[4], float g[4][4])
{
    float result { 0. };
    for (int i { 0 }; i < 4; i++)
    {
        float intermediate { 0. };
        // One component of the matrix product of g with v.
        #pragma unroll
        for (int j { 0 }; j < 4; j++)
        {
            intermediate += g[i][j]*v[j];
        }
        result += v[i]*intermediate;
    }

    return result;
}

// Advances with a step of RKF45.
__device__ void advanceRay(float x[4], float v[4], float g[4][4])
{

}

// CUDA kernels.

// Spacetime raytracing kernel. Should be called from a Tracer object.
// Uses RKF45 (Runge-Kutta-Fehlberg adaptive step).
__global__ void traceImage(Metric* metric,
                           unsigned int d_cam_pixels[2],
                           unsigned char* d_cam_pixel_array,
                           float* d_cam_fov_conv_factor,
                           float d_cam_coords[8],
                           float* d_d_phi,
                           float* d_d_theta)
{
    // Currently intended for 8x4 thread blocks.
    // Big thread blocks are more likely to need different numbers of steps (akin to thread divergence)
    // and require more iteration over the shared array pixel_done.

    // 32 bytes each.
    __shared__ bool pixel_valid[8][4];
    __shared__ bool pixel_done[8][4];
    // Metric tensor. Should be okay to keep this in registers (64 bytes).
    float g[4][4];
    // Keep the Christoffel symbols in shared memory for safety. These can probably be stored
    // safely in registers (256 bytes per core, 8 KB per block), but it might be bad on older GPUs.
    __shared__ float c_symbols[8][4][4][4][4];

    unsigned int pixel_x { blockIdx.x*blockDim.x + threadIdx.x };
    unsigned int pixel_y { blockIdx.y*blockDim.y + threadIdx.y };

    // If false, then the pixel is outside the image; ignore it.
    pixel_valid[threadIdx.x][threadIdx.y] = (pixel_x < d_cam_pixels[0]) && (pixel_y < d_cam_pixels[1]);
    // Count any invalid pixels as complete (stops the raytracer from moving their rays).
    pixel_done[threadIdx.x][threadIdx.y] = !pixel_valid[threadIdx.x][threadIdx.y];

    int num_valid_pixels { 0 };
    for (int i { 0 }; i < 8; i++)
    {
        for (int j { 0 }; j < 4; j++)
        {
            num_valid_pixels += 1*pixel_valid[i][j];
        }
    }

    // Store coordinates and velocity together.
    // First 4 numbers are the 4-position, last 4 are the 4-velocity.
    float xv[8];
    #pragma unroll
    for (int i { 0 }; i < 4; i++)
    {
        xv[i] = d_cam_coords[i];
    }
    // Initial metric tensor at the camera coordinates. Same for all rays.
    metric->calculateMetric(&xv[0], g);
    // Calculate ray starting velocity.
    metric->calculateStartV(static_cast<float>(pixel_x), static_cast<float>(pixel_y), g, &xv[4],
        d_cam_pixels, &d_cam_coords[4], *d_cam_fov_conv_factor);

    // Test; this probably isn't necessary.
    __syncthreads();
}
