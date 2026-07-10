#ifndef TRACE_KERNEL_UTILS
#define TRACE_KERNEL_UTILS

// I hate having so many things here, but CUDA doesn't like modularised code; it's easier
// to put everything the raytracing kernel uses in a single place.

// pi is needed by the host when importing sky maps and used by the device when getting pixels.
const float pi_host = 3.141592653589793;
__device__ __constant__ float pi_device = 3.141592653589793;

// Quaternionic arithmetic functions.

// Calculate the Hamilton (quaternionic) product of two quaternions.
__device__ void quatProduct(float u[4], float v[4], float result[4]);
// Rotates a 3D Cartesian vector, vec (a pure quaternion), by rotation_quat.
// result will be the rotated vector represented as a pure quaternion.
__device__ void rotateVecByQuat(float vec[4], float rotation_quat[4], float result[4]);

// Should have copies between the host and device.
class Metric
{
    public:
        // Default metric is flat spacetime with Minkowski coordinates.
        // Calculates the metric tensor at r and stores it in g.
        __device__ virtual void calculateMetric(float r[4], float g[4][4]);
        // Returns whether to terminate a photon passing going through this metric.
        __device__ virtual bool terminateRay(float r[4]);
        // Calculates the start velocity of a photon at pixel (x, y), where (0, 0) is the top-left corner of the camera.
        // Overwrites result into v. Assumes Minkowski coordinates.
        __device__ void calculateStartV(float x, float y, float g[4][4], float v[4],
            unsigned int d_cam_pixels[2], float d_cam_quat[4], float d_cam_fov_conv_factor);
        // Makes a velocity vector null.
        __device__ void makeVNull(float v[4], float g[4][4]);
};

class Schwarzschild : public Metric
{
    public:
        // Schwarzschild spacetime.
        // Overwrite functions specific to this metric.
        __device__ void calculateMetric(float r[4], float g[4][4]) override;
        __device__ bool terminateRay(float r[4]) override;

    private:
        // Black hole radius (Schwarzschild radius).
        // Assumed fixed for now.
        const float s_radius { 1. };
        const float s_radius_squared { s_radius * s_radius };
};

// Calculates the scalar product of a velocity with in some metric.
__host__ __device__ float scalarProduct(float v[4], float g[4][4]);

// Advances with a step of RKF45.
__device__ void advanceRayRKF45(Metric *metric, float x[4], float v[4], float g[4][4], bool stop_advance);

// CUDA kernels.
__global__ void traceImage(Metric *metric,
                           unsigned int d_cam_pixels[2],
                           unsigned char *d_cam_pixel_array,
                           float *d_cam_fov_conv_factor,
                           float d_cam_coords[8],
                           float *d_d_phi,
                           float *d_d_theta,
                           unsigned char *d_sky_pixels,
                           unsigned char *d_sky_map);

#endif // TRACE_KERNEL_UTILS
