#include <iostream>
#include <cmath>
#include <omp.h>

#include "utilities/float_defn.h"
#include "utilities/math_functions.h"
#include "trace_kernel_utils_CPU.h"

namespace RKF45 {
    Real A[6] { 0., 2./9., 1./3., 0.75, 1., 5./6. };
    Real B_0[1] { 0. };    // B_0 should not be used! Exists for consistency.
    Real B_1[1] { 2./9. };
    Real B_2[2] { 1./12., 1./4. };
    Real B_3[3] { 69./128., -243./128., 135./64. };
    Real B_4[4] { -17./12., 27./4., -27./5., 16./15. };
    Real B_5[5] { 65./432., -5./16., 13./16., 4./27., 5./144. };
    Real *B[6] { &B_0[0], &B_1[0], &B_2[0], &B_3[0], &B_4[0], &B_5[0] };
    Real c_k_4[6] { 1./9., 0., 9./20., 16./45., 1./12., 0. };
    Real c_k_5[6] { 47./450., 0., 12./25., 32./225., 1./30., 6./25. };
};

// ----------------
// Host functions for the Schwarzschild metric.

void
Schwarzschild::calculateMetric(Real const r[4], Real g[4][4])
{
    Real r_squared { rSquared(&r[1]) };
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

// Calculates the start velocity of a photon at pixel (x, y), where (0, 0) is the top-left corner of the camera.
// Overwrites result into v. Assumes Minkowski/Cartesian coordinates.
void
Schwarzschild::calculateStartV(
    Real const x,
    Real const y,
    Real const g[4][4],
    Real v[4],
    unsigned int const cam_pixels[2],
    Real cam_quat[4],
    Real const &cam_fov_conv_factor
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

// Pseudo-Newtonian central force that corresponds to null geodesics.
void
Schwarzschild::calculateCentralAccel(Real const r[3], Real const &h_squared, Real accel[3])
{
    Real const r_norm { rMagnitude(r) };
    Real const scale_factor = (-1.5 * s_radius * h_squared) / std::pow(r_norm, 5);
    #pragma unroll
    for (int i { 0 }; i < 3; i++) {
        accel[i] = scale_factor * r[i];
    }
}

bool
Schwarzschild::terminateRay(Real const r[4])
{
    Real const r_squared { rSquared(&r[1]) };
    return (r_squared < inner_limit_squared) || (r_squared > outer_limit_squared);
}

bool
Schwarzschild::setToBlack(Real const r[4])
{
    // Fallen into the photon sphere/black hole if true.
    return rSquared(&r[1]) < inner_limit_squared;
}

Real
Schwarzschild::schwarzschildRadius() const
{
    return s_radius;
}

// Make a velocity vector null (assuming Cartesian Schwarzschild coordinates).
void
Schwarzschild::makeVNull(Real v[4], Real const g[4][4])
{
    Real const a { g[0][0] };
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
    // in order to evolve the photon backwards from the camera.
    v[0] = (-b + std::sqrt(b*b - 4.*a*c)) / (2.*a);
}

// ----------------

// Advances with a step of RKF45.
void
advanceRayRKF45(
    Schwarzschild *metric,
    Real x[4],
    Real v[4],
    Real const &e,
    Real const &h_squared,
    Real &dl,
    Real const &tolerance
)
{
    // Real const max_dl { 4. };

    Real xv_4[8];
    Real xv_5[8];
    bool success { false };

    while (!success) {
        success = true;

        // Intermediate derivatives for RKF45.
        Real k_all[6][8];

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

            // Calculate spatial velocity derivatives with the central
            // pseudo-Newtonian potential/force field.
            Real accel[3];
            metric->calculateCentralAccel(&xv_[1], h_squared, accel);

            // Current set of derivatives to modify.
            Real *k { &k_all[k_num][0] };
            // Set k components.
            // 4-position derivatives are already known.
            // t is evolved using the conserved pseudo-energy, e.
            k[0] = e / (1. - metric->schwarzschildRadius() / rMagnitude(&xv_[1]));
            #pragma unroll
            for (int i = 1; i < 4; i++) {
                k[i] = xv_[4 + i] * dl;
            }
            // Set velocity derivatives.
            // There is no equation to evolve dt/d(lambda); k[4] just exists to maintain the array structure.
            k[4] = 0.;
            #pragma unroll
            for (int i = 1; i < 4; i++) {
                k[4 + i] = accel[i - 1] * dl;
            }
        }

        // Calculate 4th and 5th-order estimate deltas. Set to zero first.
        #pragma unroll
        for (int i { 0 }; i < 8; i++) {
            xv_4[i] = 0.;
            xv_5[i] = 0.;
        }

        for (int i { 0 }; i < 6; i++) {
            #pragma unroll
            for (int mu { 0 }; mu < 8; mu++) {
                xv_4[mu] += RKF45::c_k_4[i] * k_all[i][mu];
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

        #pragma unroll
        for (int mu { 0 }; mu < 8; mu++) {
            Real error { std::abs(xv_5[mu] - xv_4[mu]) };
            if (error > max_error) max_error = error;
        }

        success = max_error < tolerance;

        // Calculate next step size to try if tolerance checks failed.
        dl *= 0.9 * std::pow(tolerance / max_error, 0.2);
        // Limit max step size.
        // if (dl > max_dl) dl = max_dl;
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
)
{
    Real const tolerance { metric->schwarzschildRadius() * 1e-5 };
    unsigned int const num_pixels = cam_pixels[0] * cam_pixels[1];

    #pragma omp parallel for
    for (unsigned int i = 0; i < num_pixels; i++) {
        unsigned int const pixel_x = i % cam_pixels[0];
        unsigned int const pixel_y = i / cam_pixels[0];

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
        metric->calculateMetric(xv, g);
        metric->calculateStartV(
            static_cast<Real>(pixel_x),
                                static_cast<Real>(pixel_y),
                                g,
                                &xv[4],
                                cam_pixels,
                                cam_quat,
                                cam_fov_conv_factor
        );

        // Pseudo-energy of the photon; acts as a conserved quantity
        // used to evolve t.
        // FIXME: Doesn't work at the event horizon.
        Real const e = xv[4] * (1. - metric->schwarzschildRadius() / rMagnitude(&xv[1]));

        // Get the angular momentum per unit mass (i.e. treat it as
        // a classic, massive particle).
        // "Mass" is a bit of a misnomer here, it's just |r x v|.
        Real L[3];
        crossProduct(&xv[1], &xv[5], L);
        Real const h_squared = L[0] * L[0] + L[1] * L[1] + L[2] * L[2];

        // Set initial step length; doesn't really matter much
        // because it gets modified automatically.
        Real dl { 1. };

        // Main raytracing loop.
        while (!metric->terminateRay(xv)) {
            advanceRayRKF45(metric, &xv[0], &xv[4], e, h_squared, dl, tolerance);
            if (i == 0) {
                metric->calculateMetric(xv, g);
                xv[4] = e / (1. - metric->schwarzschildRadius() / rMagnitude(&xv[1]));
                std::cout << scalarProduct(&xv[4], g) << "\n";
            }
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
            // Fallen into the photon sphere/black hole if true.
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
