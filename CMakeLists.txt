cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(
    Geodesic_Raytracer
    VERSION 1.0
    LANGUAGES CXX CUDA
)

set (CMAKE_CXX_STANDARD_REQUIRED 23)
set (CMAKE_CUDA_STANDARD_REQUIRED 12)

find_package(CUDAToolkit REQUIRED)

set (SOURCE_DIR "source")
set (UI_DIR "ui")

add_executable(main.sh main.cu)
target_compile_options(main.sh PRIVATE "-O1")

add_library(Tracer "${SOURCE_DIR}/tracer.h" "${SOURCE_DIR}/tracer.cu")
add_subdirectory(${SOURCE_DIR})
# target_compile_options(Tracer PRIVATE "-diag-suppress 550")
target_link_libraries(main.sh PRIVATE Tracer)
