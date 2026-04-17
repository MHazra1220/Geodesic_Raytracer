#include "metric.h"

// Stores the calculated independent components in g.
// Flat/Minkowski spacetime; same everywhere.
__host__ __device__ void Metric::calculateMetric(float r[4], float g[4][4])
{
    g[0][0] = -1.;
    #pragma unroll
    for (int i { 1 }; i < 4; i++)
    {
        for (int j { i }; j < 4; j++)
        {
            if (i == j)
            {
                g[i][j] = -1.;
            }
            else
            {
                g[i][j] = 0.;
                g[j][i] = 0.;
            }
        }
    }
}

// Returns whether to terminate a photon passing going through this metric.
__host__ __device__ bool Metric::terminateRay(float r[4]){
    // Flat spacetime has no obvious termination condition; just return true for now.
    return true;
}

// Calculates the scalar product of a velocity in some metric.
// Tries to use as little memory as possible; the goal
// is to minimize register occupancy, not computation.
__host__ __device__ float scalarProduct(float v[4], float g[4][4])
{
    float result { 0. };
    #pragma unroll
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
