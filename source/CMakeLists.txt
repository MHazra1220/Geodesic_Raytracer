add_library(Tracer tracer.cu)
add_library(Trace_Kernels trace_kernel.cu)
target_link_libraries(Tracer PRIVATE Trace_Kernels)
