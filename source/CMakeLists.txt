target_sources(Tracer
    PRIVATE
        quaternion.h
        quaternion.cu
)

add_library(Quaternion quaternion.h quaternion.cu)
target_link_libraries(Tracer PRIVATE Quaternion)

set (GEOMETRY_DIR "geometry")
add_subdirectory(${GEOMETRY_DIR})
target_link_libraries(Tracer PRIVATE Metric)
