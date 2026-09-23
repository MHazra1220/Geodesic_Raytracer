#ifndef TRACE_KERNEL_UTILS
#define TRACE_KERNEL_UTILS

#include "utilities/float_defn.h"

// I hate having so many things here, but CUDA doesn't like modularised code; it's easier
// to put everything the raytracing kernel uses in a single place.

// pi is needed by the host when importing sky maps and used by the device when getting pixels.
Real const pi_host = 3.141592653589793;
// __device__ __constant__ Real pi_device = 3.141592653589793;

// Quaternionic arithmetic functions.

__host__ __device__ void crossProduct(Real const u[3], Real const v[3], Real cross[3]);
// Calculate the Hamilton (quaternionic) product of two quaternions.
void quatProduct(Real const u[4], Real const v[4], Real result[4]);
// Rotates a 3D Cartesian vector, vec (a pure quaternion), by rotation_quat.
// result will be the rotated vector represented as a pure quaternion.
void rotateVecByQuat(Real vec[4], Real rotation_quat[4], Real result[4]);

class Schwarzschild
{
    public:
        // Schwarzschild spacetime.
        // Overwrite functions specific to this metric.
        void calculateMetric(Real r[4], Real g[4][4]);
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

        Real schwarzschildRadius();

    private:
        void makeVNull(Real v[4], Real const g[4][4]);

        // Black hole radius (Schwarzschild radius).
        // Assumed fixed for now.
        Real const s_radius { 1. };
        // Simulation terminates if a ray gets within this radius.
        Real const inner_limit { 1.5 * s_radius };
        Real const inner_limit_squared { inner_limit * inner_limit };
        Real const outer_limit_squared { (40. * s_radius) * (40. * s_radius) };
};

// Calculates the scalar product of a velocity with in some metric.
Real scalarProduct(Real const v[4], Real const g[4][4]);
void invertSymmetric4Metric(Real const m[4][4], Real m_inv[4][4]);

// Advances with a step of RKF45.
void advanceRayRKF45(
    Schwarzschild *metric,
    Real x[4],
    Real v[4],
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

// CUDA kernels.
/*__global__ void traceImage(
    unsigned int d_cam_pixels[2],
    unsigned char *d_cam_pixel_array,
    Real *d_cam_fov_conv_factor,
    Real d_cam_coords[8],
    Real *d_d_phi,
    Real *d_d_theta,
    int d_sky_pixels[2],
    unsigned char *d_sky_map
);*/

#endif // TRACE_KERNEL_UTILS
