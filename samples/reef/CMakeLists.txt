add_subdirectory(reef)

add_definitions(-DREEF_RESOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}/reef/resource/")


find_package(glog REQUIRED)

include_directories("${CMAKE_CURRENT_LIST_DIR}/reef/src")

add_executable(reefdisb 
    "reefclient.cpp"
    "main.cpp"
)

target_link_libraries(reefdisb 
    reef_client_lib
    reef_util 
    reef_grpc_proto
    glog::glog
    ${GTEST_BOTH_LIBRARIES}
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF}
    disb
    jsoncpp
)