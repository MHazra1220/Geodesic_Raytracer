#ifndef TRACER
#define TRACER

// pi is needed by the host when importing sky maps and used by the device when getting pixels.
const float pi_host = 3.141592653589793;
__device__ __constant__ float pi_device = 3.141592653589793;

/*
 *  Metrics are currently defined in coordinates of (ct, x, y, z)
 *  with the assumption that c = 1 is set, so the coordinates are
 *  in just (t, x, y, z). The mathematical functions can handle
 *  arbitrary coordinates just fine, but some checks rely on Cartesian coordinates
 *  for now (e.g. checking if a photon crosses the photon sphere
 *  in the Schwarzschild metric).
*/

// h_ indicates a host-bound variable.
// d_ indicates a device-bound variable.

class Tracer
{
    private:
        // Should be 3 for 24-bit RGB images.
        int byte_depth;
        // Dimensions of the sky map in pixels (width, height).
        int h_sky_pixels[2];
        // Use float versions when sampling from the sky map.
        float h_sky_pixels_f[2];
        int* d_sky_pixels { nullptr };
        float* d_sky_pixels_f { nullptr };
        // Sky map is stored on the host and the device.
        // Unsigned char to represent unsigned 8-bit integers.
        unsigned char *h_sky_map { nullptr };
        unsigned char *d_sky_map { nullptr };
        // Intervals between azimuthal and polar angles in radians.
        float h_d_phi;
        float h_d_theta;
        // These have to be pointers, even though they only store a single number each.
        float* d_d_phi { nullptr };
        float* d_d_theta { nullptr };
        // Camera location and orientation.
        float h_pos[4];
        float h_quat[4];
        float* d_pos;
        float* d_quat;
        // Camera dimensions.
        int* d_cam_pixels { nullptr };
        unsigned char* d_cam_pixel_array { nullptr };
        float* d_cam_fov_conv_factor { nullptr };

    public:
        Tracer(float initial_pos[4], float initial_quat[4], int cam_pixels[2], float cam_fov, char skymap_file[]);
        ~Tracer();

        // Setup functions.
        void setCameraCoords(float camera_pos[4], float camera_quat[4]);
        void setCameraResFOV(int cam_pixels[2], float fov_width);
        void importSkyMap(char skymap_file[]);

        // Calculates the start velocity of a photon at pixel (x, y), where (0, 0) is the top-left corner of the camera.
        // Overwrites result into v. Assumes Minkowski coordinates.
        __device__ void calculateStartV(float x, float y, float g[4][4], float v[4]);
};

#endif
