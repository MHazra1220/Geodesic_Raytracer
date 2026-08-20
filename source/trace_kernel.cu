#include <iostream>
#include <cmath>
#include <omp.h>

#include "utilities/float_defn.h"
#include "trace_kernel_utils.h"

// Constants for RKF45 method. Have to be in global namespace, unfortunately.

// GPU constants for a future GPU-conversion.
__device__ __constant__ Real A[6] { 0., 2./9., 1./3., 0.75, 1., 5./6. };
__device__ __constant__ Real B_0[1] { 0. };    // B_0 should not be used! Exists for consistency.
__device__ __constant__ Real B_1[1] { 2./9. };
__device__ __constant__ Real B_2[2] { 1./12., 0.25 };
__device__ __constant__ Real B_3[3] { 69./128., -243./128., 135./64. };
__device__ __constant__ Real B_4[4] { -17./12., 27./4., -27./5., 16./15. };
__device__ __constant__ Real B_5[5] { 65./432., -5./16., 13./16., 4./27., 5./144. };
__device__ __constant__ Real *B[6] { &B_0[0], &B_1[0], &B_2[0], &B_3[0], &B_4[0], &B_5[0] };
__device__ __constant__ Real c_k_4[6] { 1./9., 0., 9./20., 16./45., 1./12., 0. };
__device__ __constant__ Real c_k_5[6] { 47./450., 0., 12./25., 32./225., 1./30., 6./25. };

namespace RKF45 {
    Real A[6] { 0., 2./9., 1./3., 0.75, 1., 5./6. };
    Real B_0[1] { 0. };    // B_0 should not be used! Exists for consistency.
    Real B_1[1] { 2./9. };
    Real B_2[2] { 1./12., 0.25 };
    Real B_3[3] { 69./128., -243./128., 135./64. };
    Real B_4[4] { -17./12., 27./4., -27./5., 16./15. };
    Real B_5[5] { 65./432., -5./16., 13./16., 4./27., 5./144. };
    Real *B[6] { &B_0[0], &B_1[0], &B_2[0], &B_3[0], &B_4[0], &B_5[0] };
    Real c_k_4[6] { 1./9., 0., 9./20., 16./45., 1./12., 0. };
    Real c_k_5[6] { 47./450., 0., 12./25., 32./225., 1./30., 6./25. };
};

// Quaternionic arithmetic functions.

// Calculate the Hamilton (quaternionic) product of two quaternions.
void
quatProduct(Real u[4], Real v[4], Real result[4])
{
    result[0] = u[0]*v[0] - (u[1]*v[1] + u[2]*v[2] + u[3]*v[3]);
    // Cross product of the vector components of u and v is needed.
    Real cross[3];
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
void
rotateVecByQuat(Real vec[4], Real rotation_quat[4], Real result[4])
{
    // Assume that rotation_quat is normalised; checking isn't worth the cost.
    Real rotation_quat_inverse[4];
    rotation_quat_inverse[0] = rotation_quat[0];
    rotation_quat_inverse[1] = -rotation_quat[1];
    rotation_quat_inverse[2] = -rotation_quat[2];
    rotation_quat_inverse[3] = -rotation_quat[3];
    Real intermediate_result[4];
    quatProduct(vec, rotation_quat_inverse, intermediate_result);
    quatProduct(rotation_quat, intermediate_result, result);
}

// Function definitions for Metric and its derived classes.

// Stores the calculated independent components in g.
// Flat/Minkowski spacetime; same everywhere.
void
Metric::calculateMetric(Real r[4], Real g[4][4])
{
    g[0][0] = -1.; g[0][1] = 0.; g[0][2] = 0.; g[0][3] = 0.;
    g[1][0] = 0.; g[1][1] = 1.; g[1][2] = 0.; g[0][3] = 0.;
    g[2][0] = 0.; g[2][1] = 0.; g[2][2] = 1.; g[2][3] = 0.;
    g[3][0] = 0.; g[3][1] = 0.; g[3][2] = 0.; g[3][3] = 1.;
}

// Returns whether to terminate a photon passing through a point in this metric.
bool
Metric::terminateRay(Real r[4])
{
    // Flat spacetime has no obvious termination condition.
    // Currently just measures whether the ray is beyond some radius.
    Real radius_squared { r[1] * r[1] +
                           r[2] * r[2] +
                           r[3] * r[3] };
    return radius_squared > outer_limit_squared;
}

bool
Metric::setToBlack(Real r[4])
{
    return false;
}

// Calculates the start velocity of a photon at pixel (x, y), where (0, 0) is the top-left corner of the camera.
// Overwrites result into v. Assumes Minkowski/Cartesian coordinates.
void
Metric::calculateStartV(
    Real x,
    Real y,
    Real g[4][4],
    Real v[4],
    unsigned int cam_pixels[2],
    Real cam_quat[4],
    Real &cam_fov_conv_factor
)
{
    // Local phi and theta coordinates in the camera's reference frame.
    // Negative in phi because phi increases anticlockwise around the local z-axis.
    Real phi { -((x - 0.5 * cam_pixels[0]) * (cam_fov_conv_factor)) };
    Real theta { (y - 0.5 * cam_pixels[1]) * (cam_fov_conv_factor) + 0.5 * pi_host };
    // Minkowski/Cartesian coordinates.
    Real unrotated_v[4];
    unrotated_v[0] = 0.;
    unrotated_v[1] = std::sin(theta) * std::cos(phi);
    unrotated_v[2] = std::sin(theta) * std::sin(phi);
    unrotated_v[3] = std::cos(theta);
    // Rotate to align with the camera's orientation in the global frame.
    rotateVecByQuat(unrotated_v, cam_quat, v);
    // Modify the t-component to make the velocity null.
    makeVNull(v, g);
}

// Make a velocity vector null (assuming Minkowski coordinates).
void
Metric::makeVNull(Real v[4], Real g[4][4])
{
    Real a { g[0][0] };
    Real b { 0. };
    Real c { 0. };

    #pragma unroll
    for (int i { 1 }; i < 4; i++)
    {
        b += g[0][i]*v[i];
    }
    b *= 2.;

    // Calculate c.
    for (int i { 1 }; i < 4; i++)
    {
        Real contraction { 0. };
        #pragma unroll
        for (int j { 1 }; j < 4; j++)
        {
            contraction += g[i][j]*v[j];
        }
        c += contraction*v[i];
    }

    // Take the positive root solution. a = g_00 is usually negative, so this normally makes v[0]
    // in order to evolve the photon backwards from the camera. Makes no difference for static metrics.
    v[0] = (-b + std::sqrt(b*b - 4.*a*c)) / (2.*a);
}

// Schwarzschild metric functions.
//--------------------------------
void
Schwarzschild::calculateMetric(Real r[4], Real g[4][4])
{
    Real r_squared { r[1] * r[1] + r[2] * r[2] + r[3] * r[3] };
    Real r_mag { std::sqrt(r_squared) };
    Real mult_factor { s_radius / (r_squared * (r_mag - s_radius)) };
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

bool
Schwarzschild::terminateRay(Real r[4])
{
    Real r_squared { r[1] * r[1] + r[2] * r[2] + r[3] * r[3] };
    return (r_squared < photon_sphere_squared) || (r_squared > outer_limit_squared);
}

bool
Schwarzschild::setToBlack(Real r[4])
{
    Real r_squared { r[1] * r[1] + r[2] * r[2] + r[3] * r[3] };
    // Fallen into the photon sphere/black hole if true.
    return r_squared < photon_sphere_squared;
}

// Calculates the scalar product of a velocity with a given metric tensor.
// Tries to use as little memory as possible; the goal
// is to minimize register occupancy, not computation.
Real
scalarProduct(Real v[4], Real g[4][4])
{
    Real result { 0. };
    for (int i { 0 }; i < 4; i++)
    {
        Real intermediate { 0. };
        // One component of the matrix product of g with v.
        #pragma unroll
        for (int j { 0 }; j < 4; j++)
        {
            intermediate += g[i][j] * v[j];
        }
        result += v[i] * intermediate;
    }
    return result;
}

// Inverts a symmetric 4x4 metric; needed to get the inverse metric for the Christoffel symbols.
void
invertSymmetric4Metric(Real m[4][4], Real m_inv[4][4])
{
    // Computationally fastest way for such a small system is probably
    // a hard implementation of the 4x4 inverse.
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
    Real sum { 0. };
    for (int i { 0 }; i < 4; i++) {
        sum += m_inv[i][i] * m[i][i];
        Real intermediate_sum { 0. };
        for (int j { i + 1 }; j < 4; j++) {
            intermediate_sum += m_inv[i][j] * m[i][j];
        }
        sum += 2. * intermediate_sum;
    }

    // Scale inverse metric appropriately.
    Real scale_factor { 4.f / sum };
    #pragma unroll
    for (int i { 0 }; i < 4; i++) {
        #pragma unroll
        for (int j { 0 }; j < 4; j++) {
            m_inv[i][j] *= scale_factor;
        }
    }
}

// Calculate metric derivatives at r.
void
calculateMetricDerivs(Metric *metric, Real r[4], Real g_derivs[4][4][4])
{
    // FIXME: Fixed step for now; this needs to be adaptive!
    const Real step { 1e-6 };
    const Real half_step { 0.5 * step };
    const Real inv_step { 1. / step };

    // Calculate metric derivatives.
    // Forward step.
    for (int alpha { 0 }; alpha < 4; alpha++) {
        Real g_temp[4][4];

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
void
calculateChristoffelSymbols(
    Metric *metric,
    Real r[4],
    Real g[4][4],
    Real c_symbols[4][4][4],
    Real g_derivs[4][4][4])
{
    calculateMetricDerivs(metric, r, g_derivs);

    Real g_inv[4][4];
    invertSymmetric4Metric(g, g_inv);

    // Calculate the Christoffel symbols.
    for (int alpha { 0 }; alpha < 4; alpha++) {
        for (int mu { 0 }; mu < 4; mu++) {
            for (int nu { mu }; nu < 4; nu++) {
                Real sum { 0. };
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
void
advanceRayRKF45(
    Metric *metric,
    Real x[4],
    Real v[4],
    Real &dl,
    const Real &tolerance
)
{
    const Real max_dl { 5. };

    Real xv_4[4];
    Real xv_5[8];
    bool success { false };

    while (!success) {
        success = true;

        // Metric tensor.
        Real g[4][4];
        // Intermediate derivatives for RKF45.
        Real k_all[6][8];
        // Christoffel symbols.
        Real c_symbols[4][4][4];
        // Metric derivatives.
        Real g_derivs[4][4][4];

        // Calculate the 6 k-vectors.
        for (int k_num { 0 }; k_num < 6; k_num++) {
            // _ marks variables with temporary offsets.
            Real xv_[8] { x[0], x[1], x[2], x[3], v[0], v[1], v[2], v[3] };
            // No need to modify the affine parameter;
            // it has no effect on the derivative function (for now?).

            // Loop does nothing for k_0.
            Real *B_set { RKF45::B[k_num] };
            for (int i { 0 }; i < k_num; i++) {
                #pragma unroll
                for (int mu { 0 }; mu < 8; mu++) {
                    xv_[mu] += B_set[i] * k_all[i][mu];
                }
            }

            // Current set of derivatives to modify.
            Real *k { &k_all[k_num][0] };

            // Calculate velocity derivatives with the Christoffel symbols.
            // Need the metric tensor first.
            metric->calculateMetric(&xv_[0], g);
            calculateChristoffelSymbols(metric, &xv_[0], g, c_symbols, g_derivs);
            for (int mu { 0 }; mu < 4; mu++) {
                // 4-position derivative is just the 4-velocity.
                k[mu] = xv_[4 + mu] * dl;

                Real component { 0. };
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

        // Calculate 4th and 5th-order estimate deltas. Set to zero first.
        #pragma unroll
        for (int i { 0 }; i < 4; i++) xv_4[i] = 0.;
        #pragma unroll
        for (int i { 0 }; i < 8; i++) xv_5[i] = 0.;

        for (int i { 0 }; i < 6; i++) {
            // Only need positions to 4th-order for tolerance testing.
            #pragma unroll
            for (int mu { 0 }; mu < 4; mu++) {
                xv_4[mu] += RKF45::c_k_4[i] * k_all[i][mu];
            }
            #pragma unroll
            for (int mu { 0 }; mu < 8; mu++) {
                xv_5[mu] += RKF45::c_k_5[i] * k_all[i][mu];
            }
        }

        // Test truncation error tolerances in position.
        // bool advance { true };
        Real max_error { 0. };

        // for (int mu { 0 }; mu < 4; mu++) {
        //     Real error { fabsf(xv_5[mu] - xv_4[mu]) };
        //     advance = advance && (error < tolerance);
        //     bool replace_error { error > max_error };
        //     max_error = (replace_error * error) + (!replace_error * max_error);
        // }
        // If stop_advance is true, don't advance no matter what.
        // advance = advance && (!stop_advance);

        for (int mu { 0 }; mu < 4; mu++) {
            Real error { std::abs(xv_5[mu] - xv_4[mu]) };
            if (error > max_error) max_error = error;
        }

        success = max_error < tolerance;

        // Calculate next step size to try if tolerance checks failed.
        dl *= 0.9 * std::pow(tolerance / max_error, 0.2);
        // Limit max step size.
        bool limit_size { dl > max_dl };
        dl = (!limit_size * dl) + (limit_size * max_dl);
    }

    // Advance positions and velocities.
    // Doesn't advance until tolerance checks pass.
    #pragma unroll
    for (int mu { 0 }; mu < 4; mu++) {
        x[mu] += xv_5[mu];
        v[mu] += xv_5[4 + mu];
    }
}

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
)
{
    const Real tolerance { 2.5e-5 };
    unsigned int num_pixels = cam_pixels[0] * cam_pixels[1];

    #pragma omp parallel for
    for (unsigned int i = 0; i < num_pixels; i++) {
        unsigned int pixel_x = i % cam_pixels[0];
        unsigned int pixel_y = i / cam_pixels[0];

        // Store coordinates and velocity together.
        // First 4 numbers are the 4-position, last 4 are the 4-velocity.
        Real xv[8];
        #pragma unroll
        for (int mu { 0 }; mu < 4; mu++) {
            xv[mu] = cam_pos[mu];
        }

        // Metric tensor.
        Real g[4][4];

        // Initial metric tensor and starting velocity.
        metric->calculateMetric(&xv[0], g);
        metric->calculateStartV(
            static_cast<Real>(pixel_x),
            static_cast<Real>(pixel_y),
            g,
            &xv[4],
            cam_pixels,
            cam_quat,
            cam_fov_conv_factor
        );

        // Set initial step length to maximum; it will probably be cut down automatically.
        Real const max_dl { 5. };
        Real dl { max_dl };

        // Main raytracing loop.
        while (!metric->terminateRay(&xv[0])) {
            advanceRayRKF45(metric, &xv[0], &xv[4], dl, tolerance);
        }

        // Use the velocity to take the photon to infinity and sample the sky box.
        Real phi { std::atan2(xv[6], xv[5]) };
        // Move into the range 0 to 2*pi if phi < 0.
        phi += 2. * pi_host * (phi < 0.);
        Real theta { std::acos(xv[7]) / (std::sqrt(xv[5] * xv[5] +
                                                     xv[6] * xv[6] +
                                                     xv[7] * xv[7]))};

        // Convert to pixel locations on the sky map; floor the number.
        // Phi goes anticlockwise, so 2.*pi - phi transforms it to stop
        // the image using the wrong phi coordinates.
        int sky_x { (int)((2. * pi_host - phi) / d_phi) };
        int sky_y { (int)(theta / d_theta) };
        // Address of the pixel RGB colour.
        unsigned char *colour { &sky_map[3 * (sky_y * sky_pixels[0] + sky_x)] };
        // Fallen into a photon sphere/black hole if true.
        bool set_to_black = metric->setToBlack(&xv[0]);

        // Write camera image.
        // Some thread divergence may occur here in a GPU rewrite.
        // TODO: Set pixels to black if they enter a black hole (when viewed from beyond the photon sphere).
        unsigned int pixel_index { 3 * (pixel_y * cam_pixels[0] + pixel_x) };
        #pragma unroll
        for (unsigned int j = 0; j < 3; j++) {
            if (!set_to_black) {
                cam_pixel_array[pixel_index + j] = colour[j];
            }
            else {
                cam_pixel_array[pixel_index + j] = 0;
            }
        }
    }
}

// CUDA kernels.

// Spacetime raytracing kernel. Should be called from a Tracer object.
// Uses RKF45 (Runge-Kutta-Fehlberg adaptive step).
// Modifies the array d_cam_pixel_array in place with the traced image.
/*__global__ void
traceImage(
    unsigned int d_cam_pixels[2],
    unsigned char *d_cam_pixel_array,
    Real *d_cam_fov_conv_factor,
    Real d_cam_coords[8],
    Real *d_d_phi,
    Real *d_d_theta,
    int d_sky_pixels[2],
    unsigned char *d_sky_map
)
{
    // Currently intended for 8x4 thread blocks.
    // Big thread blocks are more likely to need different numbers of steps (thread divergence)
    // and require more iteration over the shared array pixel_done.

    const Real tolerance { 1e-4 };
    // Set initial step length to maximum; it will probably be cut down automatically.
    Real d_l { 5. };

    // 32 bytes each.
    __shared__ bool pixel_valid[8][4];
    __shared__ bool pixel_done[8][4];
    // Metric tensor. Should be okay to keep this in registers (64 bytes).
    Real g[4][4];
    // Intermediate derivatives for RKF45.
    __shared__ Real k[8][4][6][8];
    // Keep the Christoffel symbols in shared memory for safety. These can probably be stored
    // safely in registers (256 bytes per core, 8 KB per block), but it might be bad on older GPUs.
    __shared__ Real c_symbols[8][4][4][4][4];
    // Metric derivatives.
    __shared__ Real g_derivs[8][4][4][4][4];

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
    Real xv[8];
    #pragma unroll
    for (int i { 0 }; i < 4; i++) {
        xv[i] = d_cam_coords[i];
    }

    // Initial metric tensor at the camera coordinates. Same for all rays.
    Dev::calculateMetric(&xv[0], g);
    // Calculate ray starting velocity.
    Dev::calculateStartV(static_cast<Real>(pixel_x), static_cast<Real>(pixel_y), g, &xv[4],
        d_cam_pixels, &d_cam_coords[4], d_cam_fov_conv_factor);

    // TEST: This might be unnecessary.
    // Potential thread divergence due to Taylor expansions in calculateMetric and calculateStartV.
    __syncthreads();

    // Main raytracing loop. Iterates until all the pixels in the thread block are done.
    // Should avoid thread divergence.
    int num_pixels_done { 0 };
    while (num_pixels_done != num_valid_pixels) {
        pixel_done[threadIdx.x][threadIdx.y] = Dev::terminateRay(&xv[0]) || pixel_done[threadIdx.x][threadIdx.y];

        advanceRayRKF45(&xv[0], &xv[4], g, &g_derivs[threadIdx.x][threadIdx.y][0],
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
    Real phi { atan2(xv[6], xv[5]) };
    // Move into the range 0 to 2*pi if phi < 0.
    phi += 2. * pi_device * (phi < 0.);
    Real theta { acos(xv[7] * rnorm3df(xv[5], xv[6], xv[7])) };

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
        unsigned int pixel_index { 3 * (pixel_y * d_cam_pixels[0] + pixel_x) };
        #pragma unroll
        for (unsigned int i = 0; i < 3; i++) {
            d_cam_pixel_array[pixel_index + i] = colour[i];
        }
    }
}*/
