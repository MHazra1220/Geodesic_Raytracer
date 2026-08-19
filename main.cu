#include "source/utilities/float_defn.h"
#include "source/tracer.h"

#include <iostream>

int main()
{
    // Path to the sky map. Should be a 2:1 aspect ratio image.
    // TODO: Make this a command-line argument. Remove it entirely and put it in a GUI of some sort later.
    char sky_map[] { "/home/mh2001/Documents/Programming/General_Relativity/Geodesic_Raytracer/sky_box_samples/full_milky_way.jpg" };

    // Initial camera position and quaternion.
    // First number of pos is the time coordinate.
    Real pos[4] { 0., 10., 0., 0. };
    // A quaternion of (1, 0, 0, 0) is the "null" quaternion and aligns the camera with the xyz axes (angle = 0).
    Real quat[4] { 0., 0., 0., 1. };
    unsigned int cam_pixels[2] { 2560, 1440 };
    Real cam_fov { 120. };

    // Path to output the image (for now). Want to create a "real-time" view later on.
    char output_image_path[] { "/home/mh2001/Documents/Programming/General_Relativity/Geodesic_Raytracer/output_images/test.jpg" };

    Tracer tracer_test { pos, quat, cam_pixels, cam_fov, sky_map };
    tracer_test.traceImage();
    tracer_test.saveTracedImage(output_image_path);

    return 0;
}
