#ifndef TRACER
#define TRACER

#include "trace_kernel_utils.h"

/*
 *  Metrics are currently defined in coordinates of (ct, x, y, z)
 *  with the assumption that c = 1 is set, so the coordinates are
 *  in just (t, x, y, z). The mathematical functions can handle
 *  arbitrary coordinates, but some checks rely on Cartesian coordinates
 *  for now (e.g. checking if a photon crosses the photon sphere
 *  in the Schwarzschild metric).
*/

// h_ indicates a host-bound variable/pointer.
// d_ indicates a device-bound variable/pointer.

class Tracer
{
    public:
        Tracer(float initial_pos[4], float initial_quat[4], unsigned int cam_pixels[2], float cam_fov, char skymap_file[]);
        ~Tracer();

        // Setup functions.
        void setCameraCoords(float camera_pos[4], float camera_quat[4]);
        void setCameraResFOV(unsigned int cam_pixels[2], float fov_width);
        void importSkyMap(char skymap_file[]);

        // Calls the raytracing kernel.
        void callTraceKernel();

        // Copy traced image from device back to host and save.
        void transferImageToHost();
        void saveTracedImage(char output_path[]);

    private:
        // Should be 3 for 24-bit RGB images.
        int byte_depth;
        // Dimensions of the sky map in pixels (width, height).
        int h_sky_pixels[2];
        int *d_sky_pixels { nullptr };
        // Sky map is stored on the host and the device.
        // Unsigned char to represent unsigned 8-bit integers.
        unsigned char *h_sky_map { nullptr };
        unsigned char *d_sky_map { nullptr };
        // Intervals between azimuthal and polar angles in radians.
        // These have to be pointers, even though they only store a single number each.
        float *d_d_phi { nullptr };
        float *d_d_theta { nullptr };
        // Camera location and orientation stored together for faster transfer.
        // For a "real-time" view later on, these need to be copied constantly to the device.
        float h_cam_coords[8];
        float *d_cam_coords { nullptr };
        // Camera dimensions.
        unsigned int h_cam_pixels[2];
        unsigned int *d_cam_pixels { nullptr };
        unsigned char *h_cam_pixel_array { nullptr };
        unsigned char *d_cam_pixel_array { nullptr };
        float *d_cam_fov_conv_factor { nullptr };

        // Metric to trace in.
        Metric h_metric;
        Metric *d_metric;
};

#endif // TRACER
