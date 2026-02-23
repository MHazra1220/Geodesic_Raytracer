#include "tracer.h"
#include "quaternion.h"
#define STB_IMAGE_IMPLEMENTATION
#include "STB_IO/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "STB_IO/stb_image_write.h"

// Some variables used by the GPU during tracing. Defined outside Tracer
// to stop the GPU needing a copy or reference of the Tracer object.
namespace DevTraceTools
{
    // Device variables and functions.
    // Pointer in host memory; assigned with cudaMemcpy because otherwise it can't be deallocated by the host
    // with cudaFree(). This points to the sky map in device memory.
    unsigned char *dev_sky_map { nullptr };
    // Points to the image array for the camera in device memory.
    unsigned char *dev_camera_pixel_array { nullptr };
    // Camera coordinates and quaternion.
    __device__ float dev_pos[4];
    __device__ float dev_quat[4];
    // Camera dimensions (width, height).
    __device__ int dev_cam_pixels[2];
    __device__ float dev_cam_pixels_f[2];
    __device__ float fov_conversion_factor;
    // Sky map dimensions and phi/theta intervals.
    __device__ int dev_sky_pixels[2];
    __device__ float dev_sky_pixels_f[2];
    __device__ float phi_interval;
    __device__ float theta_interval;
};
