#include "tracer.h"
#include "trace_kernel_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "STB_IO/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "STB_IO/stb_image_write.h"

// DEBUG MODULES
// #include "cuda_profiler_api.h"

#include <iostream>
#include <string>
#include <stdexcept>

// Stops this code snippet being spammed everywhere.
void
checkCudaError(cudaError_t err, std::string error_msg)
{
    if (err != cudaSuccess) throw std::runtime_error(error_msg);
}

Tracer::Tracer(float initial_pos[4], float initial_quat[4], unsigned int cam_pixels[2], float cam_fov, char skymap_file[])
{
    importSkyMap(skymap_file);
    setCameraCoords(initial_pos, initial_quat);
    setCameraResFOV(cam_pixels, cam_fov);
}

// Free allocated arrays on host and device.
Tracer::~Tracer()
{
    stbi_image_free(sky_map);
    free(cam_pixel_array);
}

// Intended for a 360-degree panoramic image
void
Tracer::importSkyMap(char skymap_file[])
{
    stbi_image_free(sky_map);
    sky_map = stbi_load(skymap_file, &sky_pixels[0], &sky_pixels[1], &byte_depth, 3);
    if (sky_map == nullptr) throw std::runtime_error("Error: cannot load skymap file.");

    d_phi = (2. * pi_host) / static_cast<float>(sky_pixels[0]);
    d_theta = pi_host / static_cast<float>(sky_pixels[1]);
}

void
Tracer::setCameraCoords(float camera_pos[4], float camera_quat[4])
{
    #pragma unroll
    for (int i { 0 }; i < 4; i++)
    {
        cam_pos[i] = camera_pos[i];
        cam_quat[i] = camera_quat[i];
    }
}

void
Tracer::setCameraResFOV(unsigned int input_cam_pixels[2], float fov_width)
{
    cam_pixels[0] = input_cam_pixels[0];
    cam_pixels[1] = input_cam_pixels[1];

    // Allocate memory for the camera pixel array on the host and device.
    free(cam_pixel_array);
    // 24-bit RGB image.
    image_mem_size = sizeof(unsigned char) * cam_pixels[0] * cam_pixels[1] * byte_depth;
    cam_pixel_array = (unsigned char*)malloc(image_mem_size);

    // Set camera FOV conversion factor.
    float fov_rad { fov_width * (pi_host / 180.f) };
    cam_fov_conv_factor = fov_rad / static_cast<float>(cam_pixels[0]);
}

void
Tracer::traceImage()
{
    traceImageRKF45(
        &metric,
        cam_pixels,
        cam_pixel_array,
        cam_fov_conv_factor,
        cam_pos,
        cam_quat,
        d_phi,
        d_theta,
        sky_pixels,
        sky_map
    );
}

// Save traced image. Must be copied to host first.
void
Tracer::saveTracedImage(char output_path[])
{
    stbi_write_jpg(output_path, cam_pixels[0], cam_pixels[1], byte_depth, cam_pixel_array, 100);
}
