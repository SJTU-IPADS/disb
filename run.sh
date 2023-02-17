#!/bin/bash
workpath=$(cd $(dirname $0); pwd)
trtexec=${workpath}/install/bin/trt_benchmark
tfsexec=${workpath}/install/bin/tfs_benchmark
tritonexec=${workpath}/install/bin/triton_benchmark
framework="Unknown"

if [ $# != 2 ] ; then
    echo "Expect 2 arguments"
    echo "Usage: run.sh {framework name} {testcase name}"
    echo "Example: run.sh trt A"
    exit
fi

if [ $1 = "trt" ] ; then
    exec=${trtexec}
    framework="TensorRT"
elif [ $1 = "tfs" ] ; then
    exec=${tfsexec}
    framework="Tensorflow Serving"
elif [ $1 = "triton" ] ; then
    exec=${tritonexec}
    framework="Triton"
else
    echo "Unknown framework $1"
    exit
fi

mkdir -p ${workpath}/logs
mkdir -p ${workpath}/benchmarks/results
echo "Running ${framework} testcase $2 ..."
${exec} ${workpath}/benchmarks $2 1> ${workpath}/logs/$1.$2.out 2> ${workpath}/logs/$1.$2.err
echo "Testcase $2 completed"