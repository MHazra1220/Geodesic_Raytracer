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

Tracer::Tracer(float initial_pos[4], float initial_quat[4], char skymap_file[])
{
    setCameraCoords(initial_pos, initial_quat);

    cudaError_t err { cudaSuccess };
    err = cudaMalloc((void**)&d_d_phi, sizeof(float));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_phi on device."));
    err = cudaMalloc((void**)&d_d_theta, sizeof(float));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_theta on device."));
    err = cudaMalloc((void**)&d_sky_pixels, 2*sizeof(int));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_sky_pixels on device."));
    err = cudaMalloc((void**)&d_sky_pixels_f, 2*sizeof(float));
    checkCudaError(err, std::string("Error: failed to allocate memory for d_sky_pixels_f on device."));

    importSkyMap(skymap_file);
}

// Frees allocated arrays on host and device.
Tracer::~Tracer()
{
    cudaFree(d_sky_pixels);
    cudaFree(d_sky_pixels_f);
    cudaFree(d_sky_map);
    cudaFree(d_d_phi);
    cudaFree(d_d_theta);
}

void Tracer::setCameraCoords(float camera_pos[4], float camera_quat[4])
{
    for (int i { 0 }; i < 4; i++)
    {
        pos[i] = camera_pos[i];
        quat[i] = camera_quat[i];
    }
}

void Tracer::importSkyMap(char skymap_file[])
{
    h_sky_map = stbi_load(skymap_file, &sky_pixels[0], &sky_pixels[1], &byte_depth, 3);
    sky_pixels_f[0] = static_cast<float>(sky_pixels[0]);
    sky_pixels_f[1] = static_cast<float>(sky_pixels[1]);
    h_d_phi = (2.*pi_host) / sky_pixels_f[0];
    h_d_theta = pi_host / sky_pixels_f[1];
    // Transfer to device memory.
    // Free existing map (if it exists).
    cudaFree(d_sky_map);
    size_t map_size { sizeof(unsigned char)*sky_pixels[0]*sky_pixels[1]*byte_depth };
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
