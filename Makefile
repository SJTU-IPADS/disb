# BUILD_TYPE				= Debug / Release
BUILD_TYPE					= Release

BUILD_TENSORRT				= OFF
BUILD_TRITON				= OFF
BUILD_TFSERVING				= OFF

DISB_PATH					= $(shell pwd)
BUILD_PATH					= ${DISB_PATH}/build
INSTALL_PATH				= ${DISB_PATH}/install

TESTCASES					= A B C D E REAL

.PHONY: build
build: ${BUILD_PATH}/CMakeCache.txt
	rm -rf ${INSTALL_PATH}; \
	cmake --build ${BUILD_PATH} --target install -- -j$(shell nproc)

.PHONY: install
install: build

${BUILD_PATH}/CMakeCache.txt:
	${MAKE} configure

.PHONY: configure
configure:
	cmake -B${BUILD_PATH} \
		  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
		  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
		  -DBUILD_TENSORRT=${BUILD_TENSORRT} \
		  -DBUILD_TRITON=${BUILD_TRITON} \
		  -DBUILD_TFSERVING=${BUILD_TFSERVING}

.PHONY: trt
trt:
	${MAKE} clean; \
	${MAKE} BUILD_TENSORRT=ON

.PHONY: triton
triton:
	${MAKE} clean; \
	${MAKE} BUILD_TRITON=ON

.PHONY: tfs
tfs:
	${MAKE} clean; \
	${MAKE} BUILD_TFSERVING=ON

.PHONY: trt-test
trt-test:
	@for testcase in ${TESTCASES}; do ${DISB_PATH}/run.sh trt $$testcase; done; \
	echo "Testcases completed, logs are under ${DISB_PATH}/logs, results are under ${DISB_PATH}/benchmarks/results"

.PHONY: triton-test
triton-test:
	@for testcase in ${TESTCASES}; do ${DISB_PATH}/run.sh triton $$testcase; done; \
	echo "Testcases completed, logs are under ${DISB_PATH}/logs, results are under ${DISB_PATH}/benchmarks/results"

.PHONY: tfs-test
tfs-test:
	@for testcase in ${TESTCASES}; do ${DISB_PATH}/run.sh tfs $$testcase; done; \
	echo "Testcases completed, logs are under ${DISB_PATH}/logs, results are under ${DISB_PATH}/benchmarks/results"

.PHONY: trt-container
trt-container:
	docker run -it \
			   --name disb-trt8.4 \
			   --gpus all \
			   -v ${PWD}:/workspace/disb \
			   shenwhang/disb-trt8.4:0.1 \
			   /bin/bash

.PHONY: triton-front
triton-front:
	docker run -it \
			   --name disb-triton-client \
			   --net=host \
			   -v ${PWD}:/workspace/disb \
			   shenwhang/disb-triton-client:0.2 \
			   /bin/bash

.PHONY: triton-back
triton-back:
	docker run --rm \
			   --name disb-triton-server \
			   --gpus all \
			   -p8000:8000 -p8001:8001 -p8002:8002 \
			   -v ${PWD}/benchmarks/models:/models \
			   nvcr.io/nvidia/tritonserver:22.08-py3 \
			   tritonserver \
			   --model-repository=/models

.PHONY: tfs-front
tfs-front:
	docker run -it \
			   --name disb-tfs-client \
			   --network host \
			   -v ${PWD}:/workspace/disb \
			   shenwhang/disb-tfs-client:1.0 \
			   /bin/bash

.PHONY: tfs-back
tfs-back:
	docker run -it --rm \
			   --name disb-tfs-server \
			   --gpus all \
			   -p8500:8500 -p8501:8501 \
			   -v $(DISB_PATH)/benchmarks/models:/models \
			   tensorflow/serving:2.5.4-gpu \
			   --model_config_file=/models/models.config

.PHONY: clean
clean:
	rm -rf ${BUILD_PATH} ${INSTALL_PATH}
