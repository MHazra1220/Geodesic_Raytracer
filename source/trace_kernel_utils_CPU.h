#ifndef TRACE_KERNEL_UTILS_CPU
#define TRACE_KERNEL_UTILS_CPU

#include "utilities/float_defn.h"
#include "utilities/math_functions.h"

// Declarations for CPU raytracing classes and functions.

// pi is needed by the host when importing sky maps and used by the device when getting pixels.
Real const pi_host = 3.141592653589793;

class Schwarzschild
{
public:
    // Schwarzschild spacetime.
    // Overwrite functions specific to this metric.
    void calculateMetric(Real const r[4], Real g[4][4]);
    void calculateStartV(
        Real const x,
        Real const y,
        Real const g[4][4],
        Real v[4],
        unsigned int const cam_pixels[2],
        Real cam_quat[4],
        Real const &cam_fov_conv_factor
    );
    void calculateCentralAccel(Real const r[3], Real const &h_squared, Real accel[3]);
    bool terminateRay(Real const r[4]);
    bool setToBlack(Real const r[4]);

    Real schwarzschildRadius() const;

private:
    void makeVNull(Real v[4], Real const g[4][4]);

    // Black hole radius (Schwarzschild radius).
    // Assumed fixed for now.
    Real const s_radius { 1. };
    // Simulation terminates if a ray gets within inner_limit.
    // TODO: The use of the Newtonian "magic" potential allows us to trace
    // across the event horizon. We should be able to get images from inside
    // a black hole, in which case a different halting condition is needed.
    Real const inner_limit { 1.5 * s_radius };
    Real const inner_limit_squared { inner_limit * inner_limit };
    Real const outer_limit_squared { (40. * s_radius) * (40. * s_radius) };
};

// Advances with a step of RKF45.
void advanceRayRKF45(
    Schwarzschild *metric,
    Real x[4],
    Real v[4],
    Real const &e,
    Real const &h_squared,
    Real &dl,
    Real const &tolerance
);

void traceImageRKF45(
    Schwarzschild *metric,
    unsigned int cam_pixels[2],
    unsigned char *cam_pixel_array,
    Real const &cam_fov_conv_factor,
    Real cam_pos[4],
    Real cam_quat[4],
    Real const &d_phi,
    Real const &d_theta,
    int sky_pixels[2],
    unsigned char *sky_map
);

#endif // TRACE_KERNEL_UTILS_CPU
