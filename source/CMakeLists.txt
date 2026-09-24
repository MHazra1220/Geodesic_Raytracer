add_library(Math_Functions math_functions.cu)

add_library(Trace_Kernels trace_kernel.cu)
target_compile_options(Trace_Kernels PRIVATE -Xcompiler=-fopenmp)
target_link_libraries(Trace_Kernels PRIVATE Math_Functions)
target_link_libraries(Trace_Kernels PUBLIC OpenMP::OpenMP_CUDA)

add_library(Tracer tracer.cu)
target_link_libraries(Tracer PRIVATE Trace_Kernels)
