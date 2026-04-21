#ifndef METRIC
#define METRIC

class Metric
{
    public:
        // Default metric is flat spacetime with Minkowski coordinates.
        // Stores the calculated independent components in g.
        __device__ void calculateMetric(float r[4], float g[4][4]);
        // Returns whether to terminate a photon passing going through this metric.
        __device__ bool terminateRay(float r[4]);
};

class Schwarzschild: public Metric
{
    public:
        // Schwarzschild spacetime.
        // Overwrite functions specific to this metric.
        __device__ void calculateMetric(float r[4], float g[4][4]);
        __device__ bool terminateRay(float r[4]);
};

// Calculates the scalar product of a velocity with in some metric.
__host__ __device__ float scalarProduct(float v[4], float g[4][4]);

#endif
