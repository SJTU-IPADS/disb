if (BUILD_TENSORRT)
    add_subdirectory(tensorrt)
endif()

if (BUILD_TRITON)
    add_subdirectory(triton)
endif()

if (BUILD_TFSERVING)
    add_subdirectory(tfserving)
endif()

