target_sources(Tracer
    PRIVATE
        quaternion.h
        quaternion.cu
)

add_library(Quaternion quaternion.h quaternion.cu)
target_link_libraries(Tracer PRIVATE Quaternion)
