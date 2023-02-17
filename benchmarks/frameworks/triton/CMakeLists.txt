set(TRITON_CLIENT_CPP_PATH "/workspace/triton-client")
set(TRITON_CLIENT_CPP_THIRD_PARTY_PATH "${TRITON_CLIENT_CPP_PATH}/third-party")

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(CURL CONFIG REQUIRED PATHS ${TRITON_CLIENT_CPP_THIRD_PARTY_PATH})
find_package(absl CONFIG REQUIRED PATHS ${TRITON_CLIENT_CPP_THIRD_PARTY_PATH})
find_package(gRPC CONFIG REQUIRED PATHS ${TRITON_CLIENT_CPP_THIRD_PARTY_PATH})
find_package(c-ares CONFIG REQUIRED PATHS ${TRITON_CLIENT_CPP_THIRD_PARTY_PATH})
find_package(protobuf CONFIG REQUIRED PATHS ${TRITON_CLIENT_CPP_THIRD_PARTY_PATH})

find_package(TritonCommon REQUIRED PATHS ${TRITON_CLIENT_CPP_PATH})
find_package(TritonClient CONFIG REQUIRED PATHS ${TRITON_CLIENT_CPP_PATH})


aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src BENCHMARK_SRC)
add_executable(triton_benchmark ${BENCHMARK_SRC})

target_include_directories(
	triton_benchmark
	PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include
	PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
	triton_benchmark
	PRIVATE
		disb
		jsoncpp
		TritonClient::grpcclient_static
		TritonClient::httpclient_static
		${OpenCV_LIBS}
)

install(TARGETS triton_benchmark RUNTIME DESTINATION bin)
