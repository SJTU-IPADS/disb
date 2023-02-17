# DISB Results

Test platform:

- CPU: Intel Core i7-9700 @ 3.00GHz
- GPU: NVIDIA GeForce RTX 3080 10G
- RAM: DDR4 8G * 2 @ 2400MHz
- OS: Ubuntu 18.04.2



DISB focus on **three key metrics**:

1. infer latency of real-time tasks (**RT-Latency**)

2. increased infer latency of real-time tasks compared to standalone execution (**RT-Latency-Increase**)

3. overall throughput of both real-time and best-effort tasks (**Overall-Throughput**)



If you want to see more detailed results, please refer to [the xlsx](results.xlsx) and the json files in this directory.



## Workload A

| framework          | RT-Latency (ms) | RT-Latency-Increase (ms) | Overall-Throughput (reqs/s) |
| ------------------ | --------------- | ------------------------ | --------------------------- |
| TensorRT           | 2.975           | +0.207 (+6.96%)          | 290.717                     |
| Triton             | 4.061           | -0.100 (-2.46%)          | 249.467                     |
| Tensorflow Serving | 10.098          | +1.553 (+15.38%)         | 131.950                     |



## Workload B

| framework          | RT-Latency (ms) | RT-Latency-Increase (ms) | Overall-Throughput (reqs/s) |
| ------------------ | --------------- | ------------------------ | --------------------------- |
| TensorRT           | 3.120           | +0.318 (+10.19%)         | 394.717                     |
| Triton             | 4.222           | -0.111 (-2.63%)          | 338.517                     |
| Tensorflow Serving | 10.007          | +1.226 (+12.25%)         | 133.600                     |



## Workload C

| framework          | RT-Latency (ms) | RT-Latency-Increase (ms) | Overall-Throughput (reqs/s) |
| ------------------ | --------------- | ------------------------ | --------------------------- |
| TensorRT           | 5.616           | +2.836 (+50.50%)         | 571.033                     |
| Triton             | 6.280           | +1.913 (+30.46%)         | 553.033                     |
| Tensorflow Serving | 23.930          | +15.011 (+62.73%)        | 152.467                     |



## Workload D

| framework          | RT-Latency (ms) | RT-Latency-Increase (ms) | Overall-Throughput (reqs/s) |
| ------------------ | --------------- | ------------------------ | --------------------------- |
| TensorRT           | 27.707          | +23.991 (+86.59%)        | 597.933                     |
| Triton             | 25.320          | +19.527 (+77.12%)        | 574.717                     |
| Tensorflow Serving | 182.115         | +164.024 (+90.07%)       | 96.933                      |



## Workload E

| framework          | RT-Latency (ms) | RT-Latency-Increase (ms) | Overall-Throughput (reqs/s) |
| ------------------ | --------------- | ------------------------ | --------------------------- |
| TensorRT           | 26.812          | +23.097 (+86.15%)        | 612.967                     |
| Triton             | 24.486          | +18.618 (+76.04%)        | 570.400                     |
| Tensorflow Serving | 174.612         | +156.448 (+89.60%)       | 100.700                     |



## Workload REAL

| framework          | RT-Latency (ms) | RT-Latency-Increase (ms) | Overall-Throughput (reqs/s) |
| ------------------ | --------------- | ------------------------ | --------------------------- |
| TensorRT           | 27.026          | +23.254 (+86.04%)        | 617.250                     |
| Triton             | 24.526          | +18.737 (+76.39%)        | 572.250                     |
| Tensorflow Serving | 171.870         | +153.824 (+89.50%)       | 102.683                     |

