set(GEOMETRY_DIR "geometry")

add_library(Tracer tracer.cu)

add_subdirectory(${GEOMETRY_DIR})
target_link_libraries(Tracer PRIVATE Metric)
