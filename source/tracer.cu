#include "tracer.h"
#include "geometry/metric.h"
#define STB_IMAGE_IMPLEMENTATION
#include "STB_IO/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "STB_IO/stb_image_write.h"

#include <iostream>
#include <string>
#include <stdexcept>

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

// Stops this code snippet being spammed everywhere.
void checkCudaError(cudaError_t err, std::string error_msg)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(error_msg);
    }
}

Tracer::Tracer(float initial_pos[4], float initial_quat[4], unsigned int cam_pixels[2], float cam_fov, char skymap_file[])
{
    // Allocate all the device arrays/variables. Freed later by ~Tracer().
    cudaError_t err { cudaSuccess };
    err = cudaMalloc((void**)&d_d_phi, sizeof(float));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_phi on device."));
    err = cudaMalloc((void**)&d_d_theta, sizeof(float));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_theta on device."));
    err = cudaMalloc((void**)&d_cam_coords, sizeof(float)*8);
    checkCudaError(err, std::string("Error: failed to allocate memory for d_cam_coords on device."));
    err = cudaMalloc((void**)&d_sky_pixels, 2*sizeof(int));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_sky_pixels on device."));
    err = cudaMalloc((void**)&d_sky_pixels_f, 2*sizeof(float));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_sky_pixels_f on device."));
    err = cudaMalloc((void**)&d_cam_pixels, 2*sizeof(int));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_cam_pixels on device."));
    err = cudaMalloc((void**)&d_cam_fov_conv_factor, sizeof(float));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_cam_fov_conv_factor on device."));

    setCameraCoords(initial_pos, initial_quat);
    setCameraResFOV(cam_pixels, cam_fov);
    importSkyMap(skymap_file);
}

// Free allocated arrays on host and device.
Tracer::~Tracer()
{
    cudaFree(d_sky_pixels);
    cudaFree(d_sky_pixels_f);
    cudaFree(d_sky_map);
    cudaFree(d_d_phi);
    cudaFree(d_d_theta);
    cudaFree(d_cam_coords);
    cudaFree(d_cam_pixels);
    cudaFree(d_cam_pixel_array);
    cudaFree(d_cam_fov_conv_factor);
}

void Tracer::setCameraCoords(float camera_pos[4], float camera_quat[4])
{
    for (int i { 0 }; i < 4; i++)
    {
        h_cam_coords[i] = camera_pos[i];
        h_cam_coords[4+i] = camera_quat[i];
    }
    // Copy to device.
    cudaError_t err { cudaSuccess };
    err = cudaMemcpy(d_cam_coords, &h_cam_coords[0], 8*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, std::string("Error: failed to copy camera coordinates to device."));
}

void Tracer::setCameraResFOV(unsigned int cam_pixels[2], float fov_width)
{
    cudaError_t err { cudaSuccess };
    err = cudaMemcpy(d_cam_pixels, &cam_pixels[0], 2*sizeof(unsigned int), cudaMemcpyHostToDevice);
    checkCudaError(err, std::string("Error: failed to copy camera pixel dimensions to device."));
    // Allocate memory for the camera pixel array on the device.
    cudaFree(d_cam_pixel_array);
    // 24-bit RGB image.
    err = cudaMalloc((void**)&d_cam_pixel_array, sizeof(unsigned char)*cam_pixels[0]*cam_pixels[1]*3);
    checkCudaError(err, std::string("Error: failed to allocate memory for camera pixel array on device."));

    // Set camera FOV conversion factor.
    float fov_rad { fov_width * (pi_host/180.f) };
    float conv_factor { fov_rad / static_cast<float>(cam_pixels[0]) };
    err = cudaMemcpy(d_cam_fov_conv_factor, &conv_factor, sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, std::string("Error: failed to copy camera FOV to device."));
}

void Tracer::importSkyMap(char skymap_file[])
{
    h_sky_map = stbi_load(skymap_file, &h_sky_pixels[0], &h_sky_pixels[1], &byte_depth, 3);
    h_sky_pixels_f[0] = static_cast<float>(h_sky_pixels[0]);
    h_sky_pixels_f[1] = static_cast<float>(h_sky_pixels[1]);
    h_d_phi = (2.*pi_host) / h_sky_pixels_f[0];
    h_d_theta = pi_host / h_sky_pixels_f[1];
    // Transfer to device memory.
    // Free existing map (if it exists).
    cudaFree(d_sky_map);
    size_t map_size { sizeof(unsigned char)*h_sky_pixels[0]*h_sky_pixels[1]*byte_depth };
    cudaError_t err { cudaSuccess };
    err = cudaMalloc((void**)&d_sky_map, map_size);
    checkCudaError(err, std::string("Error: failed to allocate memory for sky map on device."));
    err = cudaMemcpy(d_sky_map, h_sky_map, map_size, cudaMemcpyHostToDevice);
    checkCudaError(err, std::string("Error: failed to copy sky map from host to device."));
    err = cudaMemcpy(d_d_phi, &h_d_phi, sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, std::string("Error: failed to copy phi interval from host to device."));
    err = cudaMemcpy(d_d_theta, &h_d_theta, sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, std::string("Error: failed to copy theta interval from host to device."));
}

// Calculates the start velocity of a photon at pixel (x, y), where (0, 0) is the top-left corner of the camera.
// Overwrites result into v. Assumes Minkowski coordinates.
__device__ void Tracer::calculateStartV(float x, float y, float g[4][4], float v[4])
{
    // Local phi and theta coordinates in the camera's reference frame.
    // Negative in phi because phi increases anticlockwise around the local z-axis.
    float phi { -((x - 0.5f*d_cam_pixels[0]) * (*d_cam_fov_conv_factor)) };
    float theta { (y - 0.5f*d_cam_pixels[1]) * (*d_cam_fov_conv_factor) + 0.5f*pi_device };
    // Minkowski/Cartesian coordinates.
    float unrotated_v[4];
    unrotated_v[0] = 0.;
    unrotated_v[1] = sin(theta)*cos(phi);
    unrotated_v[2] = sin(theta)*sin(phi);
    unrotated_v[3] = cos(theta);
    float* d_quat { &d_cam_coords[4] };
    // Rotate to align with the camera's orientation in the global frame.
    rotateVecByQuat(unrotated_v, d_quat, v);
    // Modify the t-component to make the velocity null.
    makeVNull(v, g);
}

// Make a velocity vector null (assuming Minkowski coordinates).
__device__ void Tracer::makeVNull(float v[4], float g[4][4])
{
    float a { g[0][0] };
    float b { 0. };
    float c { 0. };

    // Calculate b and c.
    for (int i { 1 }; i < 4; i++)
    {
        float contraction { 0. };
        #pragma unroll
        for (int j { 1 }; j < 4; j++)
        {
            contraction += g[i][j]*v[j];
        }
        b += g[0][i]*v[i];
        c += contraction*v[i];
    }
    b *= 2.;

    // Take the positive root solution. a = g_00 is usually negative, so this normally makes v[0] negative
    // in order to evolve the photon backwards from the camera. Makes no difference for static metrics.
    v[0] = (-b + sqrt(b*b - 4.*a*c)) / (2.*a);
}

// CUDA kernels.
__global__ void traceImage(Tracer &tracer,
                           Metric &metric,
                           unsigned int d_cam_pixels[2],
                           unsigned char* d_cam_pixel_array,
                           float d_cam_coords[8],
                           float d_d_phi,
                           float d_d_theta)
{
    // Currently intended for 8x8 thread blocks.
    // Big thread blocks are more likely to encounter warp divergence.

    __shared__ bool pixel_valid[8][8];
    // Metric tensors for all threads in this block. Risk of register spilling;
    // store in shared memory. Slower than registers; try those later.
    __shared__ float g[8][8][4][4];

    unsigned int pixel_x { blockIdx.x*blockDim.x + threadIdx.x };
    unsigned int pixel_y { blockIdx.y*blockDim.y + threadIdx.y };

    // If false, then the pixel is outside the image; ignore it.
    pixel_valid[threadIdx.x][threadIdx.y] = (pixel_x < d_cam_pixels[0]) && (pixel_y < d_cam_pixels[1]);

    if (pixel_valid[threadIdx.x][threadIdx.y])
    {
        // Calculate ray starting velocity.
        // Store coordinates and velocity together.
        float xv[8];
        for (int i { 0 }; i < 4; i++)
        {
            xv[i] = d_cam_coords[i];
        }
        // Initial metric tensor at the camera coordinates. Same for all rays.
        metric.calculateMetric(&xv[0], &g[threadIdx.x][threadIdx.y][0]);
        tracer.calculateStartV(static_cast<float>(pixel_x), static_cast<float>(pixel_y), &g[threadIdx.x][threadIdx.y][0], &xv[4]);
    }
}
