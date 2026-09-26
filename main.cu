#include "source/utilities/float_defn.h"
#include "source/tracer.h"
// #include "source/tracer_CPU.h"

#include <iostream>

int main()
{
    // Path to the sky map. Should be a 2:1 aspect ratio image.
    // TODO: Make this a command-line argument. Remove it entirely and put it in a GUI of some sort later.
    char sky_map[] { "/home/mh2001/Documents/Programming/General_Relativity/Geodesic_Raytracer/sky_box_samples/full_milky_way.jpg" };

    // Initial camera position and quaternion.
    // First number of pos is the time coordinate.
    Real pos[4] { 0., 10., 0., 0. };
    // A quaternion of (1, 0, 0, 0) is the "null" quaternion and aligns the camera with the xyz axes,
    // looking along +x with +y to the left and +z straight up.
    Real quat[4] { 0., 0., 0., 1. };
    unsigned int cam_pixels[2] { 2560, 1440 };
    Real cam_fov { 120. };

    // Path to output the image (for now). Want to create a "real-time" view later on.
    char output_image_path[] { "/home/mh2001/Documents/Programming/General_Relativity/Geodesic_Raytracer/output_images/schwarzschild_test_f32_GPU.jpg" };

    // TracerCPU tracer_test_CPU { pos, quat, cam_pixels, cam_fov, sky_map };
    // tracer_test_CPU.traceImage();
    // tracer_test_CPU.saveTracedImage(output_image_path);

    Tracer tracer_test_GPU { pos, quat, cam_pixels, cam_fov, sky_map };
    tracer_test_GPU.traceImageSchwarzschild();
    tracer_test_GPU.saveTracedImage(output_image_path);

    return 0;
}
