#include "utilities/float_defn.h"
#include "tracer.h"
#include "trace_kernel_utils.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "STB_IO/stb_image.h"

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "STB_IO/stb_image_write.h"

// DEBUG MODULES
// #include "cuda_profiler_api.h"

#include <stdexcept>

// Stops this code snippet being spammed everywhere.
void
checkCudaError(cudaError_t err, std::string error_msg)
{
    if (err != cudaSuccess) throw std::runtime_error(error_msg);
}

Tracer::Tracer(
    Real initial_pos[4],
    Real initial_quat[4],
    unsigned int cam_pixels[2],
    Real cam_fov,
    char skymap_file[]
)
{
    cudaError_t err { cudaSuccess };
    err = cudaMalloc((void **)&d_cam_coords, 8 * sizeof(Real));
    checkCudaError(err, "Error: failed to allocate memory for d_cam_coords.");
    err = cudaMalloc((void **)&d_cam_fov_conv_factor, sizeof(Real));
    checkCudaError(err, "Error: failed to allocate memory for d_cam_fov_conv_factor.");
    err = cudaMalloc((void **)&d_cam_pixels, 2 * sizeof(unsigned int));
    checkCudaError(err, "Error: failed to allocate memory for d_cam_pixels.");
    err = cudaMalloc((void **)&d_d_phi, sizeof(Real));
    checkCudaError(err, "Error: failed to allocate memory for d_d_phi.");
    err = cudaMalloc((void **)&d_d_theta, sizeof(Real));
    checkCudaError(err, "Error: failed to allocate memory for d_d_theta.");
    err = cudaMalloc((void **)&d_sky_pixels, 2 * sizeof(int));
    checkCudaError(err, "Error: failed to allocate memory for d_sky_pixels.");

    importSkyMap(skymap_file);
    setCameraCoords(initial_pos, initial_quat);
    setCameraResFOV(cam_pixels, cam_fov);
}

// Free allocated arrays.
Tracer::~Tracer()
{
    stbi_image_free(sky_map);
    cudaFree(d_sky_map);
    cudaFree(d_sky_pixels);
    cudaFree(d_d_phi);
    cudaFree(d_d_theta);
    cudaFree(d_cam_coords);
    free(cam_pixel_array);
    cudaFree(d_cam_pixel_array);
    cudaFree(d_cam_pixels);
    cudaFree(d_cam_fov_conv_factor);
}

// Intended for a 360-degree panoramic image
void
Tracer::importSkyMap(char skymap_file[])
{
    stbi_image_free(sky_map);
    sky_map = stbi_load(skymap_file, &sky_pixels[0], &sky_pixels[1], &byte_depth, 3);
    if (sky_map == nullptr) throw std::runtime_error("Error: cannot load skymap file.");

    const Real pi_host { 3.141592653589793 };
    d_phi = (2. * pi_host) / static_cast<Real>(sky_pixels[0]);
    d_theta = pi_host / static_cast<Real>(sky_pixels[1]);

    cudaError_t err { cudaSuccess };
    // Transfer variables to device.
    cudaFree(d_sky_map);
    size_t map_size { sizeof(unsigned char) * sky_pixels[0] * sky_pixels[1] * byte_depth };
    err = cudaMalloc((void **)&d_sky_map, map_size);
    checkCudaError(err, "Error: failed to allocate memory for d_sky_map.");
    err = cudaMemcpy(d_sky_pixels, sky_pixels, 2 * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err, "Error: failed to copy sky pixels x and y from host to device.");
    err = cudaMemcpy(d_sky_map, sky_map, map_size, cudaMemcpyHostToDevice);
    checkCudaError(err, "Error: failed to copy sky map from host to device.");
    err = cudaMemcpy(d_d_phi, &d_phi, sizeof(Real), cudaMemcpyHostToDevice);
    checkCudaError(err, "Error: failed to copy d_phi from host to device.");
    err = cudaMemcpy(d_d_theta, &d_theta, sizeof(Real), cudaMemcpyHostToDevice);
    checkCudaError(err, "Error: failed to copy d_theta from host to device.");
}

void
Tracer::setCameraCoords(Real const camera_pos[4], Real const camera_quat[4])
{
    #pragma unroll
    for (int i { 0 }; i < 4; i++)
    {
        cam_coords[i] = camera_pos[i];
        cam_coords[4 + i] = camera_quat[i];
    }

    cudaError_t err { cudaSuccess };
    // Transfer to device.
    err = cudaMemcpy(d_cam_coords, cam_coords, 8 * sizeof(Real), cudaMemcpyHostToDevice);
    checkCudaError(err, "Error: failed to copy cam coords to device.");
}

void
Tracer::setCameraResFOV(unsigned int input_cam_pixels[2], Real fov_width)
{
    cam_pixels[0] = input_cam_pixels[0];
    cam_pixels[1] = input_cam_pixels[1];

    // Allocate memory for the camera pixel array on the host and device.
    free(cam_pixel_array);
    // 24-bit RGB image.
    image_mem_size = sizeof(unsigned char) * cam_pixels[0] * cam_pixels[1] * byte_depth;
    cam_pixel_array = (unsigned char*)malloc(image_mem_size);

    // Set camera FOV conversion factor.
    Real const pi_host { 3.141592653589793 };
    Real const fov_rad { fov_width * (pi_host / 180.) };
    cam_fov_conv_factor = fov_rad / static_cast<Real>(cam_pixels[0]);

    cudaError_t err { cudaSuccess };
    // Copy to device.
    err = cudaMemcpy(d_cam_pixels, cam_pixels, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    checkCudaError(err, "Error: failed to copy cam pixels from host to device.");
    cudaFree(d_cam_pixel_array);
    err = cudaMalloc((void **)&d_cam_pixel_array, image_mem_size);
    checkCudaError(err, "Error: failed to allocate memory for d_cam_pixel_array.");
    err = cudaMemcpy(d_cam_fov_conv_factor, &cam_fov_conv_factor, sizeof(Real), cudaMemcpyHostToDevice);
    checkCudaError(err, "Error: failed to copy camera FoV conversion factor from host to device.");
}

void
Tracer::traceImageSchwarzschild()
{
    dim3 threadsPerBlock(8, 4);
    auto num_blocks_x { cam_pixels[0] / 8 };
    auto num_blocks_y { cam_pixels[1] / 4 };
    if (cam_pixels[0] & 8 != 0)
    {
        num_blocks_x++;
    }
    if (cam_pixels[1] & 4 != 0)
    {
        num_blocks_y++;
    }
    dim3 numBlocks(num_blocks_x, num_blocks_y);

    traceImageSchwarzschildKernel<<<numBlocks, threadsPerBlock>>>
    (
        d_cam_pixels,
        d_cam_pixel_array,
        d_cam_fov_conv_factor,
        d_cam_coords,
        d_d_phi,
        d_d_theta,
        d_sky_pixels,
        d_sky_map
    );

    cudaError_t err { cudaSuccess };
    err = cudaMemcpy(cam_pixel_array, d_cam_pixel_array, image_mem_size, cudaMemcpyDeviceToHost);
    checkCudaError(err, "Error: failed to copy camera pixel array from device to host.");
    err = cudaMemcpy(cam_coords, d_cam_coords, 8 * sizeof(Real), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Error: failed to copy cam coords from device to host.");
}

// Save traced image. Must be copied to host first.
void
Tracer::saveTracedImage(char output_path[])
{
    stbi_write_jpg(output_path, cam_pixels[0], cam_pixels[1], byte_depth, cam_pixel_array, 100);
}
