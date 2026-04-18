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
        int sky_pixels[2];
        // Use float versions when sampling from the sky map.
        float sky_pixels_f[2];
        int* d_sky_pixels { nullptr };
        float* d_sky_pixels_f { nullptr };
        // Camera location and orientation.
        float pos[4];
        float quat[4];
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

    public:
        Tracer(float initial_pos[4], float initial_quat[4], char skymap_file[]);
        ~Tracer();

        void setCameraCoords(float camera_pos[4], float camera_quat[4]);
        void importSkyMap(char skymap_file[]);
};

#endif
