#ifndef QUATERNION
#define QUATERNION

// Quaternionic arithmetic functions.

// Calculate the Hamilton (quaternionic) product of two quaternions.
__device__ void quatProduct(float u[4], float v[4], float result[4]);
// Rotates a 3D cartesian vector, vec (a pure quaternion), by rotation_quat.
// result will be the rotated vector represented as a pure quaternion.
__device__ void rotateVecByQuat(float vec[4], float rotation_quat[4], float result[4]);

#endif
