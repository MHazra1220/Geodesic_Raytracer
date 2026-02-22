#include "tracer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "STB_IO/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "STB_IO/stb_image_write.h"

namespace DevTraceTools
{
    // Device variables and functions.
    // Pointer in host memory; assigned with cudaMemcpy because otherwise it can't be deallocated by the host
    // with cudaFree(). This points to the sky map in device memory.
    unsigned char *device_sky_map { nullptr };
    // Points to the image array for the camera in device memory.
    unsigned char *device_camera_pixel_array { nullptr };
};
