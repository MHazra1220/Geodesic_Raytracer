#ifndef TRACER
#define TRACER

#include "utilities/float_defn.h"
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
        Tracer(Real initial_pos[4], Real initial_quat[4], unsigned int cam_pixels[2], Real cam_fov, char skymap_file[]);
        ~Tracer();

        // Setup functions.
        void importSkyMap(char skymap_file[]);
        void setCameraCoords(Real camera_pos[4], Real camera_quat[4]);
        void setCameraResFOV(unsigned int input_cam_pixels[2], Real fov_width);

        void traceImage();
        void saveTracedImage(char output_path[]);

    private:
        // Should be 3 for 24-bit RGB images.
        int byte_depth;
        // Dimensions of the sky map in pixels (width, height).
        int sky_pixels[2];
        // Sky map is stored on the host and the device.
        // Unsigned char to represent unsigned 8-bit integers.
        unsigned char *sky_map { nullptr };
        // Intervals between azimuthal and polar angles in radians.
        Real d_phi;
        Real d_theta;
        // Camera location and orientation.
        Real cam_pos[4];
        Real cam_quat[4];
        // Camera dimensions.
        unsigned int cam_pixels[2];
        unsigned char *cam_pixel_array { nullptr };
        Real cam_fov_conv_factor;

        Schwarzschild metric;
};

#endif // TRACER
