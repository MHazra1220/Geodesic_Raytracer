#include "tracer.h"
#include "quaternion.h"
#define STB_IMAGE_IMPLEMENTATION
#include "STB_IO/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "STB_IO/stb_image_write.h"

#include <iostream>
#include <string>
#include <stdexcept>

// Stops this code snippet being spammed everywhere.
void checkCudaError(cudaError_t err, std::string error_msg)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(error_msg);
    }
}

Tracer::Tracer(float initial_pos[4], float initial_quat[4], int cam_pixels[2], float cam_fov, char skymap_file[])
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

void Tracer::setCameraResFOV(int cam_pixels[2], float fov_width)
{
    cudaError_t err { cudaSuccess };
    err = cudaMemcpy(d_cam_pixels, &cam_pixels[0], 2*sizeof(int), cudaMemcpyHostToDevice);
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
    unrotated_v[0] = d_cam_coords[0];
    unrotated_v[1] = sin(theta)*cos(phi);
    unrotated_v[2] = sin(theta)*sin(phi);
    unrotated_v[3] = cos(theta);
    float* d_quat { &d_cam_coords[4] };
    // Rotate to align with the camera's orientation in the global frame.
    rotateVecByQuat(unrotated_v, d_quat, v);
}
