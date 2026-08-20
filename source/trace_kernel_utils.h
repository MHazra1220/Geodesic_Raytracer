#ifndef TRACE_KERNEL_UTILS
#define TRACE_KERNEL_UTILS

#include "utilities/float_defn.h"

// I hate having so many things here, but CUDA doesn't like modularised code; it's easier
// to put everything the raytracing kernel uses in a single place.

// pi is needed by the host when importing sky maps and used by the device when getting pixels.
const Real pi_host = 3.141592653589793;
// __device__ __constant__ Real pi_device = 3.141592653589793;

// Quaternionic arithmetic functions.

// Calculate the Hamilton (quaternionic) product of two quaternions.
void quatProduct(Real u[4], Real v[4], Real result[4]);
// Rotates a 3D Cartesian vector, vec (a pure quaternion), by rotation_quat.
// result will be the rotated vector represented as a pure quaternion.
void rotateVecByQuat(Real vec[4], Real rotation_quat[4], Real result[4]);

// Should have copies between the host and device.
class Metric
{
    public:
        // Default metric is flat spacetime with Minkowski coordinates.
        // Calculates the metric tensor at r and stores it in g.
        virtual void calculateMetric(Real r[4], Real g[4][4]);
        // Returns whether to terminate a photon passing going through this metric.
        virtual bool terminateRay(Real r[4]);
        // Calculates the start velocity of a photon at pixel (x, y), where (0, 0) is the top-left corner of the camera.
        // Overwrites result into v. Assumes Minkowski coordinates.
        // Needed in some metrics (e.g. falling into a black hole).
        virtual bool setToBlack(Real r[4]);
        void calculateStartV(
            Real x,
            Real y,
            Real g[4][4],
            Real v[4],
            unsigned int cam_pixels[2],
            Real cam_quat[4],
            Real &cam_fov_conv_factor
        );
        // Makes a velocity vector null.
        void makeVNull(Real v[4], Real g[4][4]);

    private:
        const Real outer_limit_squared { 15. * 15. };
};

class Schwarzschild : public Metric
{
    public:
        // Schwarzschild spacetime.
        // Overwrite functions specific to this metric.
        void calculateMetric(Real r[4], Real g[4][4]) override;
        bool terminateRay(Real r[4]) override;
        bool setToBlack(Real r[4]) override;

    private:
        // Black hole radius (Schwarzschild radius).
        // Assumed fixed for now.
        const Real s_radius { 1. };
        const Real photon_sphere_radius { 1.5 * s_radius };
        const Real photon_sphere_squared { photon_sphere_radius * photon_sphere_radius };
        const Real outer_limit_squared { 25. * 25. };
};

// Calculates the scalar product of a velocity with in some metric.
Real scalarProduct(Real v[4], Real g[4][4]);

// Inverts a symmetric 4x4 metric; needed to get the inverse metric for the Christoffel symbols.
void invertSymmetric4Metric(Real m[4][4], Real m_inv[4][4]);

// Calculate metric derivatives at r.
void calculateMetricDerivs(Metric *metric, Real r[4], Real g_derivs[4][4][4]);

// Calculate the Christoffel symbols.
void calculateChristoffelSymbols(Metric *metric, Real r[4], Real g[4][4], Real c_symbols[4][4][4], Real g_derivs[4][4][4]);

// Advances with a step of RKF45.
void advanceRayRKF45(
    Metric *metric,
    Real x[4],
    Real v[4],
    Real &dl,
    const Real &tolerance
);

void traceImageRKF45(
    Metric *metric,
    unsigned int cam_pixels[2],
    unsigned char *cam_pixel_array,
    Real &cam_fov_conv_factor,
    Real cam_pos[4],
    Real cam_quat[4],
    Real &d_phi,
    Real &d_theta,
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
