find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_program(PROTO_COMPLIER protoc)
find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

set(PROTO_DIR ${CMAKE_CURRENT_LIST_DIR}/protos)
set(PROTO_GENERATE_DIR ${CMAKE_CURRENT_LIST_DIR}/rpc)

file(GLOB_RECURSE PROTO_FILE ${PROTO_DIR} *.proto)

foreach(proto ${PROTO_FILE})
    execute_process(
        COMMAND ${PROTO_COMPLIER}
            -I ${PROTO_DIR}
            --cpp_out ${PROTO_GENERATE_DIR}
            --grpc_out ${PROTO_GENERATE_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
            ${proto}
    )
endforeach()

include_directories(${PROTO_GENERATE_DIR})
file(GLOB_RECURSE PROTO_SRC ${PROTO_GENERATE_DIR} *.cc)

add_library(tfproto ${PROTO_SRC})
target_link_libraries(tfproto
    gRPC::grpc++
    ${PROTOBUF_LIBRARIES})

find_package(OpenCV REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR})
add_executable(tf_benchmark main.cpp tfclient.cpp accanalyzer.cpp)
target_link_libraries(tf_benchmark disb tfproto jsoncpp ${OpenCV_LIBRARIES})