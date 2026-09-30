set(UTILITIES_DIR "utilities")
add_subdirectory(${UTILITIES_DIR})

add_library(Trace_Kernels trace_kernel.cu)

add_library(Tracer tracer.cu)
target_link_libraries(Tracer PRIVATE Trace_Kernels)

add_library(Trace_Kernels_CPU trace_kernel_CPU.cpp)
target_compile_options(Trace_Kernels_CPU PRIVATE -fopenmp)
target_link_libraries(Trace_Kernels_CPU PRIVATE Math_Functions)
target_link_libraries(Trace_Kernels_CPU PUBLIC OpenMP::OpenMP_CXX)

add_library(Tracer_CPU tracer_CPU.cpp)
target_link_libraries(Tracer_CPU PRIVATE Trace_Kernels_CPU)
