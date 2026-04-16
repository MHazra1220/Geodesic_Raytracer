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

class Tracer
{
    private:
        // Dimensions of the sky map in pixels (width, height).
        int sky_pixels[2];
        // Use float versions when sampling from the sky map.
        float sky_pixels_f[2];

        // Private class functions.
};

#endif
