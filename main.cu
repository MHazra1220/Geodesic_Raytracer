#include "source/tracer.h"

#include <iostream>

int main()
{
    // Path to the sky map. Should be a 2:1 aspect ratio image.
    // TODO: Make this a command-line argument. Remove it entirely and put it in a GUI of some sort later.
    char sky_map[] { "/home/mh2001/Documents/Programming/General_Relativity/Geodesic_Raytracer/sky_box_samples/full_milky_way.jpg" };

    // Initial camera position and quaternion.
    float pos[4] { 0., -5., 0., 0. };
    // A quaternion of (1, 0, 0, 0) is the "null" quaternion and aligns the camera with the xyz axes (angle = 0).
    float quat[4] { 1., 0., 0., 0. };
    unsigned int cam_pixels[2] { 1920, 1080 };
    float cam_fov { 90. };

    // Path to output the image (for now). Want to create a "real-time" view later on.
    char output_image_path[] { "/home/mh2001/Documents/Programming/General_Relativity/Geodesic_Raytracer/output_images/flat_test.jpg" };

    Tracer tracer_test { pos, quat, cam_pixels, cam_fov, sky_map };
    tracer_test.callTraceKernel();
    tracer_test.transferImageToHost();
    tracer_test.saveTracedImage(output_image_path);

    return 0;
}
