add_executable(hellodisb hellodisb.cpp)

target_link_libraries(hellodisb disb jsoncpp)

install(TARGETS hellodisb RUNTIME DESTINATION bin)
