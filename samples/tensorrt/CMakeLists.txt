
add_executable(tensorrt_benchmark main.cpp rtclient.cpp accanalyzer.cpp)

find_package(CUDA 11.4 REQUIRED)
message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)

find_package(OpenCV REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

target_link_libraries(tensorrt_benchmark disb ${CUDA_LIBRARIES} nvinfer nvonnxparser jsoncpp ${OpenCV_LIBRARIES})