#ifndef MATH_FUNCTIONS
#define MATH_FUNCTIONS

#include "float_defn.h"

// Quaternionic arithmetic functions.

__host__ __device__ void crossProduct(Real const u[3], Real const v[3], Real cross[3]);
// Calculate the Hamilton (quaternionic) product of two quaternions.
__host__ __device__ void quatProduct(Real const u[4], Real const v[4], Real result[4]);
// Rotates a 3D Cartesian vector, vec (a pure quaternion), by rotation_quat.
// result will be the rotated vector represented as a pure quaternion.
__host__ __device__ void rotateVecByQuat(Real vec[4], Real rotation_quat[4], Real result[4]);

Real rMagnitude(Real const r[3]);
__device__ Real rMagnitudeDev(Real const r[3]);
__device__ Real rInvMagnitudeDev(Real const r[3]);
Real rSquared(Real const r[3]);

// Calculates the scalar product of a velocity with in some metric.
__host__ __device__ Real scalarProduct(Real const v[4], Real const g[4][4]);
__host__ __device__ void invertSymmetric4Metric(Real const m[4][4], Real m_inv[4][4]);

#endif
