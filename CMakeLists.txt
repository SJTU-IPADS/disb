cmake_minimum_required(VERSION 3.5)
project(DISB)

option(BUILD_TENSORRT "whether to build tensorrt" OFF)
option(BUILD_TRITON "whether to build triton" OFF)
option(BUILD_TFSERVING "whether to build tfserving" OFF)
option(BUILD_REEF "whether to build reef" OFF)

include_directories("${PROJECT_SOURCE_DIR}/include")
aux_source_directory("${PROJECT_SOURCE_DIR}/src" disb_src)

add_library(disb STATIC ${disb_src})

target_link_libraries(disb jsoncpp pthread)

file(GLOB_RECURSE include_src
    ${CMAKE_CURRENT_LIST_DIR}/include/*.h
)

install(TARGETS disb ARCHIVE DESTINATION lib)
install(FILES ${include_src} DESTINATION include)

add_subdirectory(samples)
add_subdirectory(benchmarks/frameworks)
