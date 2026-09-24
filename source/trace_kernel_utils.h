#ifndef TRACE_KERNEL_UTILS
#define TRACE_KERNEL_UTILS

#include "utilities/float_defn.h"
#include "utilities/math_functions.h"

// I hate having so many things here, but CUDA doesn't like modularised code; it's easier
// to put everything the raytracing kernel uses in a single place.
// This contains both host and device functions for metrics and raytracing.
// Host versions of the metrics are designed as derived classes; this is a bit of a pain with CUDA,
// so that just uses namespaced functions and constants, instead.

// pi is needed by the host when importing sky maps and used by the device when getting pixels.
__device__ __constant__ Real pi_device = 3.141592653589793;

__device__ Real rSquaredDev(Real const r[3]);

// Passing classes into CUDA kernels is a bit of a pain; hence defined with namespaces.
namespace SchwarzschildDevice
{
    // Constants and functions for CUDA kernels to evolve in Schwarzschild spacetime.
    // Black hole radius (Schwarzschild radius).
    // Assumed fixed for now.
    __device__ __constant__ Real s_radius { 1. };
    // Simulation terminates if a ray gets within inner_limit.
    // TODO: The use of the Newtonian "magic" potential allows us to trace
    // across the event horizon. We should be able to get images from inside
    // a black hole, in which case a different halting condition is needed.
    __device__ __constant__ Real inner_limit { 1.5 };
    // Set at the photon sphere.
    __device__ __constant__ Real inner_limit_squared { 2.25 };
    // Set at 40x Schwarzschild radius.
    __device__ __constant__ Real outer_limit_squared { 1600. };

    __device__ void calculateMetric(Real const r[4], Real g[4][4]);
    __device__ void calculateStartV(
        Real const x,
        Real const y,
        Real const g[4][4],
        Real v[4],
        unsigned int const cam_pixels[2],
        Real cam_quat[4],
        Real const &cam_fov_conv_factor
    );
    __device__ void calculateCentralAccel(Real const r[3], Real const &h_squared, Real accel[3]);
    __device__ bool terminateRay(Real const r[4]);
    __device__ bool setToBlack(Real const r[4]);
    __device__ void makeVNull(Real v[4], Real const g[4][4]);

    // Advances with a step of RKF45.
    __device__ void advanceRayRKF45(
        Real x[4],
        Real v[4],
        Real k_all[6][8],
        Real const &e,
        Real const &h_squared,
        Real &dl,
        Real const &tolerance,
        bool &stop_advance
    );
};

// CUDA kernels.
__global__ void traceImage(
    unsigned int const d_cam_pixels[2],
    unsigned char *d_cam_pixel_array,
    Real const &d_cam_fov_conv_factor,
    Real d_cam_coords[8],
    Real const &d_d_phi,
    Real const &d_d_theta,
    int const d_sky_pixels[2],
    unsigned char *d_sky_map
);

#endif // TRACE_KERNEL_UTILS
