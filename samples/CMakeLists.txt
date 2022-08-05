add_subdirectory(hellodisb)

if (SAMPLE_TENSORRT)
    add_subdirectory(tensorrt)
endif()

if (SAMPLE_TFSERVING)
    add_subdirectory(tfserving)
endif()

if (SAMPLE_REEF)
    add_subdirectory(reef)
endif()