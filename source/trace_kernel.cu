#include "trace_kernel_utils.h"

// Constants for RKF45 method. Have to be in global namespace, unfortunately.

__device__ __constant__ float A[6] { 0., 2./9., 1./3., 0.75, 1., 5./6. };
__device__ __constant__ float B_0[1] { 0. };    // B_0 should not be used! Exists for consistency.
__device__ __constant__ float B_1[1] { 2./9. };
__device__ __constant__ float B_2[2] { 1./12., 0.25 };
__device__ __constant__ float B_3[3] { 69./128., -243./128., 135./64. };
__device__ __constant__ float B_4[4] { -17./12., 27./4., -27./5., 16./15. };
__device__ __constant__ float B_5[5] { 65./432., -5./16., 13./16., 4./27., 5./144. };
__device__ __constant__ float *B[6] { &B_0[0], &B_1[0], &B_2[0], &B_3[0], &B_4[0], &B_5[0] };
__device__ __constant__ float c_k_4[6] { 1./9., 0., 9./20., 16./45., 1./12., 0. };
__device__ __constant__ float c_k_5[6] { 47./450., 0., 12./25., 32./225., 1./30., 6./25. };

// Quaternionic arithmetic functions.

// Calculate the Hamilton (quaternionic) product of two quaternions.
__device__ void
quatProduct(float u[4], float v[4], float result[4])
{
    result[0] = u[0]*v[0] - (u[1]*v[1] + u[2]*v[2] + u[3]*v[3]);
    // Cross product of the vector components of u and v is needed.
    float cross[3];
    cross[0] = u[2]*v[3] - u[3]*v[2];
    cross[1] = u[3]*v[1] - u[1]*v[3];
    cross[2] = u[1]*v[2] - u[2]*v[1];
    #pragma unroll
    for (int i { 1 }; i < 4; i++)
    {
        result[i] = u[0]*v[i] + v[0]*u[i] + cross[i-1];
    }
}

// Rotates a 3D Cartesian vector, vec (a pure quaternion), by rotation_quat.
// result will be the rotated vector represented as a pure quaternion.
__device__ void
rotateVecByQuat(float vec[4], float rotation_quat[4], float result[4])
{
    // Assume that rotation_quat is normalised; checking isn't worth the cost.
    float rotation_quat_inverse[4];
    rotation_quat_inverse[0] = rotation_quat[0];
    rotation_quat_inverse[1] = -rotation_quat[1];
    rotation_quat_inverse[2] = -rotation_quat[2];
    rotation_quat_inverse[3] = -rotation_quat[3];
    float intermediate_result[4];
    quatProduct(vec, rotation_quat_inverse, intermediate_result);
    quatProduct(rotation_quat, intermediate_result, result);
}

// Function definitions for Metric and its derived classes.

// Stores the calculated independent components in g.
// Flat/Minkowski spacetime; same everywhere.
__device__ void
Metric::calculateMetric(float r[4], float g[4][4])
{
    g[0][0] = -1.; g[0][1] = 0.; g[0][2] = 0.; g[0][3] = 0.;
    g[1][0] = 0.; g[1][1] = 1.; g[1][2] = 0.; g[0][3] = 0.;
    g[2][0] = 0.; g[2][1] = 0.; g[2][2] = 1.; g[2][3] = 0.;
    g[3][0] = 0.; g[3][1] = 0.; g[3][2] = 0.; g[3][3] = 1.;
}

// Returns whether to terminate a photon passing through a point in this metric.
__device__ bool
Metric::terminateRay(float r[4]){
    // Flat spacetime has no obvious termination condition.
    // Currently just measures whether the ray is beyond some radius.
    float radius_squared { r[1] * r[1] +
                           r[2] * r[2] +
                           r[3] * r[3] };
    return radius_squared < 100.;
}

// Calculates the start velocity of a photon at pixel (x, y), where (0, 0) is the top-left corner of the camera.
// Overwrites result into v. Assumes Minkowski coordinates.
__device__ void
Metric::calculateStartV(float x,
                        float y,
                        float g[4][4],
                        float v[4],
                        unsigned int d_cam_pixels[2],
                        float d_cam_quat[4],
                        float d_cam_fov_conv_factor)
{
    // Local phi and theta coordinates in the camera's reference frame.
    // Negative in phi because phi increases anticlockwise around the local z-axis.
    float phi { -((x - 0.5f * d_cam_pixels[0]) * (d_cam_fov_conv_factor)) };
    float theta { (y - 0.5f * d_cam_pixels[1]) * (d_cam_fov_conv_factor) + 0.5f * pi_device };
    // Minkowski/Cartesian coordinates.
    float unrotated_v[4];
    unrotated_v[0] = 0.;
    unrotated_v[1] = sin(theta)*cos(phi);
    unrotated_v[2] = sin(theta)*sin(phi);
    unrotated_v[3] = cos(theta);
    // Rotate to align with the camera's orientation in the global frame.
    rotateVecByQuat(unrotated_v, d_cam_quat, v);
    // Modify the t-component to make the velocity null.
    makeVNull(v, g);
}

// Make a velocity vector null (assuming Minkowski coordinates).
__device__ void
Metric::makeVNull(float v[4], float g[4][4])
{
    float a { g[0][0] };
    float b { 0. };
    float c { 0. };

    #pragma unroll
    for (int i { 1 }; i < 4; i++)
    {
        b += g[0][i]*v[i];
    }
    b *= 2.;

    // Calculate c.
    for (int i { 1 }; i < 4; i++)
    {
        float contraction { 0. };
        #pragma unroll
        for (int j { 1 }; j < 4; j++)
        {
            contraction += g[i][j]*v[j];
        }
        c += contraction*v[i];
    }

    // Take the positive root solution. a = g_00 is usually negative, so this normally makes v[0] negative
    // in order to evolve the photon backwards from the camera. Makes no difference for static metrics.
    v[0] = (-b + sqrt(b*b - 4.*a*c)) / (2.*a);
}

// Schwarzschild metric functions.
//--------------------------------
__device__ void
Schwarzschild::calculateMetric(float r[4], float g[4][4])
{
    float r_squared { r[1] * r[1] + r[2] * r[2] + r[3] * r[3] };
    float r_mag { sqrt(r_squared) };
    float mult_factor { s_radius / (r_squared * (r_mag - s_radius)) };
    for (int mu { 1 }; mu < 4; mu++)
    {
        g[0][mu] = 0.;
        g[mu][0] = 0.;
        for (int nu { mu }; nu < 4; nu++)
        {
            g[mu][nu] = mult_factor * r[mu] * r[nu];
            g[nu][mu] = g[mu][nu];
        }
    }
    g[0][0] = -1. + s_radius / r_mag;
    g[1][1] += 1.;
    g[2][2] += 1.;
    g[3][3] += 1.;
}

__device__ bool
Schwarzschild::terminateRay(float r[4])
{
    float r_squared { r[1] * r[1] + r[2] * r[2] + r[3] * r[3] };
    return r_squared > s_radius * s_radius;
}

// Calculates the scalar product of a velocity with a given metric tensor.
// Tries to use as little memory as possible; the goal
// is to minimize register occupancy, not computation.
__host__ __device__ float
scalarProduct(float v[4], float g[4][4])
{
    float result { 0. };
    for (int i { 0 }; i < 4; i++)
    {
        float intermediate { 0. };
        // One component of the matrix product of g with v.
        #pragma unroll
        for (int j { 0 }; j < 4; j++)
        {
            intermediate += g[i][j]*v[j];
        }
        result += v[i]*intermediate;
    }
    return result;
}

// Inverts a symmetric 4x4 metric; needed to get the inverse metric for the Christoffel symbols.
__device__ void
invertSymmetric4Metric(float m[4][4], float m_inv[4][4])
{
    // Computationally fastest way for such a small system is probably
    // a hard implementation of the 4x4 inverse.
    // For the love of God, don't make a typo here...
    // TODO: This needs checking!
    m_inv[0][0] = m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] +
        m[1][3]*m[2][1]*m[3][2] - m[1][1]*m[2][3]*m[3][2] -
        m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1];
    m_inv[0][1] = m[0][1]*m[2][3]*m[3][2] + m[0][2]*m[2][1]*m[3][3] +
        m[0][3]*m[2][2]*m[3][1] - m[0][1]*m[2][2]*m[3][3] -
        m[0][2]*m[2][3]*m[3][1] - m[0][3]*m[2][1]*m[3][2];
    m_inv[1][0] = m_inv[0][1];
    m_inv[0][2] = m[0][1]*m[1][2]*m[3][3] + m[0][2]*m[1][3]*m[3][1] +
        m[0][3]*m[1][1]*m[3][2] - m[0][1]*m[1][3]*m[3][2] -
        m[0][2]*m[1][1]*m[3][3] - m[0][3]*m[1][2]*m[3][1];
    m_inv[2][0] = m_inv[0][2];
    m_inv[0][3] = m[0][1]*m[1][3]*m[2][2] + m[0][2]*m[1][1]*m[2][3] +
        m[0][3]*m[1][2]*m[2][1] - m[0][1]*m[1][2]*m[2][3] -
        m[0][2]*m[1][3]*m[2][1] - m[0][3]*m[1][1]*m[2][2];
    m_inv[3][0] = m_inv[0][3];
    m_inv[1][1] = m[0][0]*m[2][2]*m[3][3] + m[0][2]*m[2][3]*m[3][0] +
        m[0][3]*m[2][0]*m[3][2] - m[0][0]*m[2][3]*m[3][2] -
        m[0][2]*m[2][0]*m[3][3] - m[0][3]*m[2][2]*m[3][0];
    m_inv[1][2] = m[0][0]*m[1][3]*m[3][2] + m[0][2]*m[1][0]*m[3][3] +
        m[0][3]*m[1][2]*m[3][0] - m[0][0]*m[1][2]*m[3][3] -
        m[0][2]*m[1][3]*m[3][0] - m[0][3]*m[1][0]*m[3][2];
    m_inv[2][1] = m_inv[1][2];
    m_inv[1][3] = m[0][0]*m[1][2]*m[2][3] + m[0][2]*m[1][3]*m[2][0] +
        m[0][3]*m[1][0]*m[2][2] - m[0][0]*m[1][3]*m[2][2] -
        m[0][2]*m[1][0]*m[2][3] - m[0][3]*m[1][2]*m[2][0];
    m_inv[3][1] = m_inv[1][3];
    m_inv[2][2] = m[0][0]*m[1][1]*m[3][3] + m[0][1]*m[1][3]*m[3][0] +
        m[0][3]*m[1][0]*m[3][1] - m[0][0]*m[1][3]*m[3][1] -
        m[0][1]*m[1][0]*m[3][3] - m[0][3]*m[1][1]*m[3][0];
    m_inv[2][3] = m[0][0]*m[1][3]*m[2][1] + m[0][1]*m[1][0]*m[2][3] +
        m[0][3]*m[1][1]*m[2][0] - m[0][0]*m[1][1]*m[2][3] -
        m[0][1]*m[1][3]*m[2][0] - m[0][3]*m[1][0]*m[2][1];
    m_inv[3][2] = m_inv[2][3];
    m_inv[3][3] = m[0][0]*m[1][1]*m[2][2] + m[0][1]*m[1][2]*m[2][0] +
        m[0][2]*m[1][0]*m[2][1] - m[0][0]*m[1][2]*m[2][1] -
        m[0][1]*m[1][0]*m[2][2] - m[0][2]*m[1][1]*m[2][0];

    // The scalar product of the metric with its inverse should give the number of dimensions, i.e. 4.
    // The metric must already be correctly normalised.
    float sum { 0. };
    for (int i { 0 }; i < 4; i++) {
        sum += m_inv[i][i] * m[i][i];
        float intermediate_sum { 0. };
        for (int j { i + 1 }; j < 4; j++) {
            intermediate_sum += m_inv[i][j] * m[i][j];
        }
        sum += 2. * intermediate_sum;
    }

    // Scale inverse metric appropriately.
    float scale_factor { 4.f / sum };
    #pragma unroll
    for (int i { 0 }; i < 4; i++) {
        #pragma unroll
        for (int j { 0 }; j < 4; j++) {
            m_inv[i][j] *= scale_factor;
        }
    }
}

// Calculate metric derivatives at r.
__device__ void
calculateMetricDerivs(Metric *metric, float r[4], float g_derivs[4][4][4])
{
    // FIXME: Fixed step for now; this needs to be adaptive!
    const float step { 5e-6 };
    const float half_step { 0.5 * step };
    const float inv_step { 1. / step };

    // Calculate metric derivatives.
    // Forward step.
    for (int alpha { 0 }; alpha < 4; alpha++) {
        float g_temp[4][4];

        // Use second-order central difference.
        // Forward step.
        r[alpha] += half_step;
        metric->calculateMetric(r, g_temp);
        for (int mu { 0 }; mu < 4; mu++) {
            for (int nu { mu }; nu < 4; nu++) {
                g_derivs[alpha][mu][nu] = g_temp[mu][nu];
                g_derivs[alpha][nu][mu] = g_temp[mu][nu];
            }
        }

        // Backward step.
        r[alpha] -= step;
        metric->calculateMetric(r, g_temp);
        for (int mu { 0 }; mu < 4; mu++) {
            for (int nu { mu }; nu < 4; nu++) {
                g_derivs[alpha][mu][nu] -= g_temp[mu][nu];
                g_derivs[alpha][mu][nu] *= inv_step;
                g_derivs[alpha][nu][mu] = g_derivs[alpha][mu][nu];
            }
        }

        // Reset r to actual camera position.
        r[alpha] += half_step;
    }
}

// Calculate the Christoffel symbols.
__device__ void
calculateChristoffelSymbols(
    Metric *metric,
    float r[4],
    float g[4][4],
    float c_symbols[4][4][4],
    float g_derivs[4][4][4])
{
    calculateMetricDerivs(metric, r, g_derivs);
    __syncthreads();

    float g_inv[4][4];
    invertSymmetric4Metric(g, g_inv);

    // Calculate the Christoffel symbols.
    for (int alpha { 0 }; alpha < 4; alpha++) {
        for (int mu { 0 }; mu < 4; mu++) {
            for (int nu { mu }; nu < 4; nu++) {
                float sum { 0. };
                for (int beta { 0 }; beta < 4; beta++) {
                    sum += g_inv[alpha][beta] * (g_derivs[nu][beta][mu] + g_derivs[mu][nu][beta] - g_derivs[beta][mu][nu]);
                }
                sum *= 0.5;
                c_symbols[alpha][mu][nu] = sum;
                c_symbols[alpha][nu][mu] = sum;
            }
        }
    }
}

// Advances with a step of RKF45.
__device__ void
advanceRayRKF45(
    Metric *metric,
    float x[4],
    float v[4],
    float g[4][4],
    float g_derivs[4][4][4],    // Shared.
    float c_symbols[4][4][4],   // Shared.
    float &dl,
    float k_all[6][8],          // Shared.
    bool stop_advance,
    const float tolerance
)
{
    const float max_dl { 5. };

    // Calculate the 6 k-vectors.
    for (int k_num { 0 }; k_num < 6; k_num++) {
        // _ marks variables with temporary offsets.
        float xv_[8] { x[0], x[1], x[2], x[3], v[0], v[1], v[2], v[3] };
        // No need to modify the affine parameter;
        // it has no effect on the derivative function (for now?).

        // Loop does nothing for k_0.
        float *B_set { B[k_num] };
        for (int i { 0 }; i < k_num; i++) {
            #pragma unroll
            for (int mu { 0 }; mu < 8; mu++) {
                xv_[mu] += B_set[i] * k_all[i][mu];
            }
        }

        // Current set of derivatives to modify.
        float *k { &k_all[k_num][0] };

        // Calculate velocity derivatives with the Christoffel symbols.
        // Need the metric tensor first.
        metric->calculateMetric(&xv_[0], g);
        calculateChristoffelSymbols(metric, &xv_[0], g, c_symbols, g_derivs);
        for (int mu { 0 }; mu < 4; mu++) {
            // 4-position derivative is just the 4-velocity.
            k[mu] = xv_[4 + mu] * dl;

            float component { 0. };
            #pragma unroll
            for (int nu { 0 }; nu < 4; nu++) {
                // Sum the diagonal first.
                component += c_symbols[mu][nu][nu] * v[nu] * v[nu];
                // Double sum the off-diagonals (Christoffel symbol symmetry).
                for (int sigma { nu + 1 }; sigma < 4; sigma++) {
                    component += 2. * c_symbols[mu][nu][sigma] * v[nu] * v[sigma];
                }
            }
            k[4 + mu] = -component * dl;
        }
    }

    // Calculate 4th and 5th-order estimate deltas.
    float xv_4[4] { 0., 0., 0., 0. };
    float xv_5[8] { 0., 0., 0., 0., 0., 0., 0., 0. };
    for (int i { 0 }; i < 6; i++) {
        // Only need positions to 4th-order for tolerance testing.
        #pragma unroll
        for (int mu { 0 }; mu < 4; mu++) {
            xv_4[mu] += c_k_4[i] * k_all[i][mu];
        }
        #pragma unroll
        for (int mu { 0 }; mu < 8; mu++) {
            xv_5[mu] += c_k_5[i] * k_all[i][mu];
        }
    }

    // Test truncation error tolerances in position.
    bool advance { true };
    float max_error { 0. };
    #pragma unroll
    for (int mu { 0 }; mu < 4; mu++) {
        float error { fabsf(xv_5[mu] - xv_4[mu]) };
        advance = advance && error < tolerance;
        bool replace_error { error > max_error };
        max_error = (replace_error * error) + (!replace_error * max_error);
    }

    // If stop_advance is true, don't advance no matter what.
    advance = advance && !stop_advance;

    // Advance positions and velocities.
    // Doesn't advance if tolerance checks failed or stop_advance is true.
    #pragma unroll
    for (int mu { 0 }; mu < 4; mu++) {
        x[mu] += xv_5[mu] * advance;
        v[mu] += xv_5[4 + mu] * advance;
    }

    // Calculate next step size (or the step size to try next if the current iteration failed tolerance checks).
    dl *= 0.9 * powf(tolerance / max_error, 0.2);
    // Limit max step size.
    bool limit_size { dl > max_dl };
    dl = (!limit_size * dl) + (limit_size * max_dl);
}

// CUDA kernels.

// Spacetime raytracing kernel. Should be called from a Tracer object.
// Uses RKF45 (Runge-Kutta-Fehlberg adaptive step).
// Modifies the array d_cam_pixel_array in place with the traced image.
__global__ void
traceImage(
    Metric *metric,
    unsigned int d_cam_pixels[2],
    unsigned char *d_cam_pixel_array,
    float *d_cam_fov_conv_factor,
    float d_cam_coords[8],
    float *d_d_phi,
    float *d_d_theta,
    int d_sky_pixels[2],
    unsigned char *d_sky_map)
{
    // Currently intended for 8x4 thread blocks.
    // Big thread blocks are more likely to need different numbers of steps (thread divergence)
    // and require more iteration over the shared array pixel_done.

    const float tolerance { 1e-4 };
    // Set initial step length to maximum; it will probably be cut down automatically.
    float d_l { 5. };

    // 32 bytes each.
    __shared__ bool pixel_valid[8][4];
    __shared__ bool pixel_done[8][4];
    // Metric tensor. Should be okay to keep this in registers (64 bytes).
    float g[4][4];
    // Intermediate derivatives for RKF45.
    __shared__ float k[8][4][6][8];
    // Keep the Christoffel symbols in shared memory for safety. These can probably be stored
    // safely in registers (256 bytes per core, 8 KB per block), but it might be bad on older GPUs.
    __shared__ float c_symbols[8][4][4][4][4];
    // Metric derivatives.
    __shared__ float g_derivs[8][4][4][4][4];

    unsigned int pixel_x { blockIdx.x * blockDim.x + threadIdx.x };
    unsigned int pixel_y { blockIdx.y * blockDim.y + threadIdx.y };

    // If false, then the pixel is outside the image; ignore it.
    pixel_valid[threadIdx.x][threadIdx.y] = (pixel_x < d_cam_pixels[0]) && (pixel_y < d_cam_pixels[1]);
    // Count any invalid pixels as complete (stops the raytracer from moving their rays).
    pixel_done[threadIdx.x][threadIdx.y] = !pixel_valid[threadIdx.x][threadIdx.y];

    int num_valid_pixels { 0 };
    for (int i { 0 }; i < 8; i++) {
        for (int j { 0 }; j < 4; j++) {
            num_valid_pixels += 1 * pixel_valid[i][j];
        }
    }

    // Store coordinates and velocity together.
    // First 4 numbers are the 4-position, last 4 are the 4-velocity.
    float xv[8];
    #pragma unroll
    for (int i { 0 }; i < 4; i++) {
        xv[i] = d_cam_coords[i];
    }

    // Initial metric tensor at the camera coordinates. Same for all rays.
    metric->calculateMetric(&xv[0], g);
    // Calculate ray starting velocity.
    metric->calculateStartV(static_cast<float>(pixel_x), static_cast<float>(pixel_y), g, &xv[4],
        d_cam_pixels, &d_cam_coords[4], *d_cam_fov_conv_factor);

    // TEST: This might be unnecessary.
    // Potential thread divergence due to Taylor expansions in calculateMetric and calculateStartV.
    __syncthreads();

    // Main raytracing loop. Iterates until all the pixels in the thread block are done.
    // Should avoid thread divergence.
    int num_pixels_done { 0 };
    while (num_pixels_done < num_valid_pixels) {
        pixel_done[threadIdx.x][threadIdx.y] = metric->terminateRay(&xv[0]) || pixel_done[threadIdx.x][threadIdx.y];

        advanceRayRKF45(metric, &xv[0], &xv[4], g, &g_derivs[threadIdx.x][threadIdx.y][0],
            &c_symbols[threadIdx.x][threadIdx.y][0], d_l, &k[threadIdx.x][threadIdx.y][0],
            pixel_done[threadIdx.x][threadIdx.y], tolerance);

        // Might be unnecessary; need to test.
        __syncthreads();

        // Small thread blocks are useful here to reduce summations.
        num_pixels_done = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                num_pixels_done += pixel_done[i][j];
            }
        }
    }

    // Use the velocity to take the photon to infinity and sample the sky box.
    float phi { atan2(xv[6], xv[5]) };
    // Move into the range 0 to 2*pi if phi < 0.
    phi += 2. * pi_device * (phi < 0.);
    float theta { acos(xv[7] * rnorm3df(xv[5], xv[6], xv[7])) };
    // WARNING: Potential thread divergence from Taylor expansions?

    // Convert to pixel locations on the sky map; floor the number.
    // Phi goes anticlockwise, so 2.*pi - phi transforms it to stop
    // the image using the wrong phi coordinates.
    unsigned int sky_x { (unsigned int)((2. * pi_device - phi) / *d_d_phi) };
    unsigned int sky_y { (unsigned int)(theta / *d_d_theta) };
    // Address of the pixel RGB colour.
    unsigned char *colour { &d_sky_map[3 * (sky_y * d_sky_pixels[0] + sky_x)] };

    // Write camera image.
    // Some thread divergence if the block goes off the camera view is inevitable
    // here. Should be a very minor effect and avoidable entirely
    // with good choices of resolutions and kernel sizes.
    if (pixel_valid[threadIdx.x][threadIdx.y]) {
        // TODO: Set pixels to black if they enter a black hole (when viewed from beyond the photon sphere..).
        unsigned int pixel_index { 3*(pixel_y * d_cam_pixels[0] + pixel_x) };
        for (unsigned int i = 0; i < 3; i++) {
            d_cam_pixel_array[pixel_index + i] = colour[i];
        }
    }
}
