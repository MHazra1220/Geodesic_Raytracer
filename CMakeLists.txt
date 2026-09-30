cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(
    Geodesic_Raytracer
    VERSION 0.1
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# set(Qt6_DIR /home/mh2001/Qt/6.11.2/gcc_64/lib/cmake/Qt6/)
# find_package(Qt6 REQUIRED COMPONENTS Core)
# set(CMAKE_AUTOUIC ON)
# set(CMAKE_AUTOMOC ON)

set(SOURCE_DIR "source")
set(UI_DIR "ui")

add_executable(main.sh main.cu)

add_subdirectory(${SOURCE_DIR})
# target_compile_options(Tracer PRIVATE "-diag-suppress 550")
target_link_libraries(main.sh PRIVATE Tracer Tracer_CPU)
