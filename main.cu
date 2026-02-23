#include <iostream>
#include <string>
#include "native/tracer.h"

int main()
{
    // Path to the sky map. Should be a 2:1 aspect ratio image.
    std::string sky_map { "/media/mh2001/SSD2/Programming/General_Relativity/Geodesic_Raytracer/sky_box_samples/full_milky_way.jpg" };

    // Initial camera position and quaternion.
    // A quaternion of (1, 0, 0, 0) is the "null" quaternion and aligns the camera with the xyz axes (angle = 0).
    float pos[4] { 0., -20., 0., 0. };
    float quat[4] { 1., 0., 0., 0. };

    // Path to output the image (for now). Want to create a "real-time" view later on.
    std::string output_image_path { "/media/mh2001/SSD2/Programming/General_Relativity/Geodesic_Raytracer/output_images/GPU_test.jpg" }

    return 0;
}
