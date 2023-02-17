#!/bin/bash

disbpath=$(cd $(dirname $0); cd ..; pwd)
exec=/workspace/tensorrt/bin/trtexec

${exec} --explicitBatch \
        --minShapes=input:1x224x224x3 \
        --optShapes=input:1x224x224x3 \
        --maxShapes=input:8x224x224x3 \
        --saveEngine=${disbpath}/benchmarks/models/densenet201-imagenet/1/model.plan \
        --onnx=${disbpath}/benchmarks/models/densenet201-imagenet/1/densenet201-imagenet.onnx

${exec} --explicitBatch \
        --minShapes=input_ids:1x1,attention_mask:1x1   \
        --optShapes=input_ids:1x32,attention_mask:1x32 \
        --maxShapes=input_ids:8x64,attention_mask:8x64 \
        --saveEngine=${disbpath}/benchmarks/models/distilbert/1/model.plan \
        --onnx=${disbpath}/benchmarks/models/distilbert/1/distilbert.onnx

${exec} --explicitBatch \
        --minShapes=input:1x224x224x3 \
        --optShapes=input:1x224x224x3 \
        --maxShapes=input:8x224x224x3 \
        --saveEngine=${disbpath}/benchmarks/models/inceptionv3-imagenet/1/model.plan \
        --onnx=${disbpath}/benchmarks/models/inceptionv3-imagenet/1/inceptionv3-imagenet.onnx

${exec} --explicitBatch \
        --minShapes='input:0':1x224x224x3 \
        --optShapes='input:0':1x224x224x3 \
        --maxShapes='input:0':8x224x224x3 \
        --saveEngine=${disbpath}/benchmarks/models/mobilenetv1-imagenet/1/model.plan \
        --onnx=${disbpath}/benchmarks/models/mobilenetv1-imagenet/1/mobilenetv1-imagenet.onnx

${exec} --explicitBatch \
        --minShapes='input:0':1x224x224x3 \
        --optShapes='input:0':1x224x224x3 \
        --maxShapes='input:0':8x224x224x3 \
        --saveEngine=${disbpath}/benchmarks/models/resnet50-imagenet/1/model.plan \
        --onnx=${disbpath}/benchmarks/models/resnet50-imagenet/1/resnet50-imagenet.onnx

${exec} --explicitBatch \
        --minShapes=input:1x224x224x3 \
        --optShapes=input:1x224x224x3 \
        --maxShapes=input:8x224x224x3 \
        --saveEngine=${disbpath}/benchmarks/models/resnet152-imagenet/1/model.plan \
        --onnx=${disbpath}/benchmarks/models/resnet152-imagenet/1/resnet152-imagenet.onnx

${exec} --explicitBatch \
        --minShapes=input:1x224x224x3 \
        --optShapes=input:1x224x224x3 \
        --maxShapes=input:8x224x224x3 \
        --saveEngine=${disbpath}/benchmarks/models/vgg19-imagenet/1/model.plan \
        --onnx=${disbpath}/benchmarks/models/vgg19-imagenet/1/vgg19-imagenet.onnx
