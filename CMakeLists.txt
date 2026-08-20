cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(
    Geodesic_Raytracer
    VERSION 0.1
    LANGUAGES CUDA
)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CUDA_FLAGS}")
# message(${CMAKE_CUDA_FLAGS})

set(CMAKE_CXX_STANDARD_REQUIRED 23)
# set(CMAKE_CUDA_STANDARD_REQUIRED 13)

set(SOURCE_DIR "source")
set(UI_DIR "ui")

add_executable(main.sh main.cu)
# target_compile_options(main.sh PRIVATE "-O1")

add_subdirectory(${SOURCE_DIR})
# target_compile_options(Tracer PRIVATE "-diag-suppress 550")
target_link_libraries(main.sh PRIVATE Tracer)
