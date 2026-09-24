#include <cmath>

#include "float_defn.h"
#include "math_functions.h"

__host__ __device__ void crossProduct(Real const u[3], Real const v[3], Real cross[3])
{
    cross[0] = u[1]*v[2] - u[2]*v[1];
    cross[1] = u[2]*v[0] - u[0]*v[2];
    cross[2] = u[0]*v[1] - u[1]*v[0];
}

// Calculate the Hamilton (quaternionic) product of two quaternions.
__host__ __device__ void
quatProduct(Real const u[4], Real const v[4], Real result[4])
{
    result[0] = u[0]*v[0] - (u[1]*v[1] + u[2]*v[2] + u[3]*v[3]);
    Real cross[3];
    crossProduct(&u[1], &v[1], cross);
    #pragma unroll
    for (int i { 1 }; i < 4; i++)
    {
        result[i] = u[0]*v[i] + v[0]*u[i] + cross[i-1];
    }
}

// Rotates a 3D Cartesian vector, vec (a pure quaternion), by rotation_quat.
// result will be the rotated vector represented as a pure quaternion.
__host__ __device__ void
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

Real
rMagnitude(Real const r[3])
{
    return std::sqrt(rSquared(r));
}

__device__ Real
rMagnitudeDev(Real const r[3])
{
    return norm3d(r[0], r[1], r[2]);
}

__device__ Real
rInvMagnitudeDev(Real const r[3])
{
    return rnorm3d(r[0], r[1], r[2]);
}

Real
rSquared(Real const r[3])
{
    return r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
}

// Calculates the scalar product of a velocity with a given metric tensor.
// Tries to use as little memory as possible; the goal
// is to minimize register occupancy, not computation.
__host__ __device__ Real
scalarProduct(Real const v[4], Real const g[4][4])
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
__host__ __device__ void
invertSymmetric4Metric(Real const m[4][4], Real m_inv[4][4])
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
