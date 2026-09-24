set(UTILITIES_DIR "utilities")
add_subdirectory(${UTILITIES_DIR})

add_library(Trace_Kernels_CPU trace_kernel_CPU.cu)
target_compile_options(Trace_Kernels_CPU PRIVATE -Xcompiler=-fopenmp)
target_link_libraries(Trace_Kernels_CPU PRIVATE Math_Functions)
target_link_libraries(Trace_Kernels_CPU PUBLIC OpenMP::OpenMP_CUDA)

add_library(Tracer tracer.cu)
target_link_libraries(Tracer PRIVATE Trace_Kernels_CPU)
