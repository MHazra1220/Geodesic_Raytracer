# set(UTILS_DIR "utilities")
# add_subdirectory(${UTILS_DIR})

add_library(Tracer tracer.cu)
add_library(Trace_Kernels trace_kernel.cu)
target_link_libraries(Tracer PRIVATE Trace_Kernels)
# target_link_libraries(Tracer PRIVATE Float_Defn)
