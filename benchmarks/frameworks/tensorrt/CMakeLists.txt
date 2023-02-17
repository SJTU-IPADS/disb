find_package(CUDA REQUIRED)
message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)

find_package(OpenCV REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}/include)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src BENCHMARK_SRC)
add_executable(trt_benchmark ${BENCHMARK_SRC})
target_link_libraries(trt_benchmark disb ${CUDA_LIBRARIES} nvinfer nvonnxparser jsoncpp ${OpenCV_LIBRARIES})

install(TARGETS trt_benchmark RUNTIME DESTINATION bin)
