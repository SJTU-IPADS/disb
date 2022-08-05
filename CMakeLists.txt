cmake_minimum_required(VERSION 3.5)
project(DISB)

option(SAMPLE_TENSORRT "whether to build tensorrt sample" FALSE)
option(SAMPLE_TFSERVING "whether to build tfserving sample" FALSE)

include_directories("${PROJECT_SOURCE_DIR}/include")
aux_source_directory("${PROJECT_SOURCE_DIR}/src" disb_src)

add_library(disb STATIC ${disb_src})

target_link_libraries(disb jsoncpp pthread)

add_subdirectory(samples)