#include "metric.h"

// Stores the calculated independent components in g.
// Flat/Minkowski spacetime; same everywhere.
__device__ void Metric::calculateMetric(float r[4], float g[4][4])
{
    g[0][0] = -1.; g[0][1] = 0.; g[0][2] = 0.; g[0][3] = 0.;
    g[1][0] = 0.; g[1][1] = 1.; g[1][2] = 0.; g[0][3] = 0.;
    g[2][0] = 0.; g[2][1] = 0.; g[2][2] = 1.; g[2][3] = 0.;
    g[3][0] = 0.; g[3][1] = 0.; g[3][2] = 0.; g[3][3] = 1.;
}

// Returns whether to terminate a photon passing through a point in this metric.
__device__ bool Metric::terminateRay(float r[4]){
    // Flat spacetime has no obvious termination condition.
    // Currently just measures whether the ray is beyond some radius.
    float radius { norm3df(r[1], r[2], r[3]) };
    return radius < 10.;
}

// Calculates the scalar product of a velocity in some metric.
// Tries to use as little memory as possible; the goal
// is to minimize register occupancy, not computation.
__host__ __device__ float scalarProduct(float v[4], float g[4][4])
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
