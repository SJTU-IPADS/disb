# DISB: DNN Inference Serving Benchmark

DISB is a DNN inference serving benchmark with diverse workloads and models. It was originally designed to simulate real-time scenarios, e.g. autonomous driving systems, where both low latency and high throughput are demanded.

DISB uses the client-server architecture, where the clients send the DNN inference requests to the server via RPC, and the server returns the inference result. Clients can submit the inference requests periodically or randomly. An inference request should contain the model name (or id), the input data and other customized attributes (e.g., priority or deadline). 



## Table of Contents

- [DISB Toolkit](#disb-toolkit)
- [DISB Workloads](#disb-workloads)
- [Build & Install](#build--install)
- [Usage](#usage)
- [Samples](#samples)
- [Benchmark Result](#benchmark-result)
- [Paper](#paper)
- [The Team](#the-team)
- [Contact Us](#contact-us)
- [License](#license)



## DISB Toolkit

DISB provides a C++ library (`libdisb`) to perform benchmarking. To integrate your own DNN inference system with DISB, you only need to implement a `DISBClient` to wrap your inference interface. See [usage](#usage) for details.



## DISB Workloads

Currently, DISB provides 5 workloads with different DNN models and different number of clients. 

There are three pattern for submitting inference requests in DISB clients:
1. Uniform Distribution (U): the client sends inference requests periodically, with a fixed frequency (e.g., 20 reqs/s). This pattern is common in data-driven applications (e.g., obstacle detection with cameras).
2. Possion Distribution (P), the client sends inference requests with a Poisson arrival distribution. This pattern can simulate event-driven applications (e.g., speech recoginition).
3. Closed-loop (C), the client continuously issues inference requests, which simulates a contention load.

See [workloads](./workloads.md) for workload details.

[TBD] We're still working on providing more representative and general DNN inference serving workloads. We will support dependent load (inference DAG) in next release.



## Build & Install

Install dependencies:

```shell
sudo apt install build-essential cmake
sudo apt install libjsoncpp-dev
```

Build and install DISB tools:

```shell
# build libdisb.a
cmake -B build
cd build
make -j$(nproc)

# install
sudo make install
```



## Usage

- ### Client

  `DISB::Client` is an adaptor class between DISB and the serving backend. You can implement the following interfaces in its subclass. These interfaces will be called during the benchmark, and their execution time will be recorded by DISB.

  ```c++
  # init() will be called once the benchmark begin
  virtual void init();
  
  # The following interfaces will be called by DISB
  # within each inference request during benchmark.
  virtual void prepareInput();
  virtual void preprocess();
  virtual void copyInput();
  virtual void infer();
  virtual void copyOutput();
  virtual void postprocess();
  ```

  

- ### Strategy

  `DISB::Strategy` instructs when DISB should launch next inference request. There are two built-in strategies in `DISB::PeriodicStrategy` (launch inference request at a given frequency periodically) and `DISB::TraceStrategy` (launch inference request according to a given trace). They can be enabled by setting certain attribute in json configuration, see [HelloDISB](samples/hellodisb) for example.
  
  
  
- ### BenchmarkSuite

  `DISB::BenchmarkSuite` should be created and initialized before the benchmark is launched.

  ```c++
  void init(const std::string &configJsonStr,
            std::shared_ptr<Client> clientFactory(const Json::Value &config),
            std::shared_ptr<Strategy> strategyFactory(const Json::Value &config) = builtinStrategyFactory);
  
  void run(void strategyCoordinator(const std::vector<StrategyInfo> &strategyInfos) = builtinStrategyCoordinator);
  ```

  When initializing BenchmarkSuite, a json format string should be passed as config, and a factory method of your own subclass implementation of `DISB::Client` should be provided. The `Json::Value` passed to the factory method is the `"client"` attribute in each task in `configJsonStr`.

  

  If you want to custom strategies other than `DISB::PeriodicStrategy` or `DISB::TraceStrategy`, you should implement the virtual method `std::chrono::system_clock::time_point nextLaunchTime(const std::chrono::system_clock::time_point &now)` and provide your own strategy factory method. The `Json::Value` passed to the factory method is the `"strategy"` attribute in each task in `configJsonStr`.
  
  
  
  If your strategies have to coordinate with each other, you can pass `strategyCoordinator()` to `DISB::BenchmarkSuite::run()`, which makes sure that strategies will not conflict with each other. For example, the `builtinStrategyCoordinator()` will prevent the periodic strategies with the same frequency and the highest priority from launching at the same time by taking standalone latency of each client into account.
  
  
  
- ### Analyzer

  `DISB::Analyzer` is used to measure the performance of each inference task, each inference task can have multiple analyzers. `DISB::BasicAnalyzer`, which can measure latency and throughput is implemented by DISB and is enabled for every task by default.

  

  If you want to custom analyzers other than `DISB::BasicAnalyzer`, for example measures gpu usage and memory consumption, the following interfaces should be implemented.

  ```c++
  virtual void init();
  virtual void start(const std::chrono::system_clock::time_point &beginTime);
  virtual void stop(const std::chrono::system_clock::time_point &endTime);
  
  virtual std::shared_ptr<DISB::Record> produceRecord() = 0;
  virtual void consumeRecord(std::shared_ptr<DISB::Record> record);
  virtual Json::Value generateReport() = 0;
  
  // The following event callback will be invoked before 
  // the corresponding method of DISB::Client is invoked.
  virtual void onPrepareInput(std::shared_ptr<DISB::Record> record);
  virtual void onPreprocess(std::shared_ptr<DISB::Record> record);
  virtual void onCopyInput(std::shared_ptr<DISB::Record> record);
  virtual void onInfer(std::shared_ptr<DISB::Record> record);
  virtual void onCopyOutput(std::shared_ptr<DISB::Record> record);
  virtual void onPostprocess(std::shared_ptr<DISB::Record> record);
  ```

  `produceRecord()` will be called before each inference request, and should be implemented. You can return a subclass of `DISB::Record`, which can be customized to store other information. The attribute `timePoints` contains begin and end time of each inference phase, including `prepareInput`, `preprocess`, etc. `timePoints` will be set by DISB while running benchmark. Other information, for example, gpu usage and memory consumption can be stored in your subclass of `DISB::Record`.

  

  Lifecycle of each `DISB::Record`:

  1. Created by `DISB::Analyzer::produceRecord()` before each inference request.
  2. Passed to each event callback of `DISB::Analyzer`, here you can store specific infomation you needed into the record.
  3. Consumed by `DISB::Analyzer::consumeRecord()` after an inference request is over.

  
  
  After you have implemented `DISB::Analyzer`, you can add it to a `DISB::Client` by calling `DISB::Client::addAnalyzer()` in the factory method of client. You may refer to [TensorRT sample](samples/tensorrt/README.md) or [Tensorflow Serving sample](samples/tfserving/README.md) for more details. They both implement an `AccuarcyAnalyzer` to measure inference accuarcy.
  
  
  
- ### Run a benchmark

  ```c++
  #include "disb.h"
  
  class HelloClient: public DISB::Client
  {
      // your implementation
  }
  
  std::shared_ptr<DISB::Client> helloClientFactory(const Json::Value &config)
  {
      return std::make_shared<HelloClient>(config["name"].asString());
  }
  
  int main(int argc, char** argv)
  {
      if (argc != 2) {
          std::cout << "Usage: hellodisb config.json" << std::endl;
          return -1;
      }
  
      DISB::BenchmarkSuite benchmark;
      std::string jsonStr = readStringFromFile(argv[1]);
      benchmark.init(jsonStr, helloClientFactory);
      benchmark.run();
      Json::Value report = benchmark.generateReport();
      std::cout << report << std::endl;
  
      return 0;
  }
  ```



## Samples

- [HelloDISB](samples/hellodisb)

  A simple sample that shows how DISB works, needs no extra dependencies.

- [TensorRT](samples/tensorrt/README.md)

  A sample serves MNIST inference requests directly using TensorRT as serving backend, needs CUDA environment to compile.

  You can enable its compiling by adding a cmake parameter: `-DSAMPLE_TENSORRT=ON`.

- [Tensorflow Serving](samples/tfserving/README.md)

  A sample serves ResNet inference requests using Tensorflow Serving as serving backend, needs gRPC environment to compile.

  You can enable its compiling by adding a cmake parameter: `-DSAMPLE_TFSERVING=ON`.


## Benchmark Result

[TBD] We will provide the benchmark result of common DNN inference framework on DISB in next release.



## Paper

If you use DISB in your research, please cite our paper:
```bibtex
@inproceedings {osdi2022reef,
  author = {Mingcong Han and Hanze Zhang and Rong Chen and Haibo Chen},
  title = {Microsecond-scale Preemption for Concurrent {GPU-accelerated} {DNN} Inferences},
  booktitle = {16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)},
  year = {2022},
  isbn = {978-1-939133-28-1},
  address = {Carlsbad, CA},
  pages = {539--558},
  url = {https://www.usenix.org/conference/osdi22/presentation/han},
  publisher = {USENIX Association},
  month = jul,
}
```



## The Team

REEF is developed and maintained by members from [IPADS@SJTU](https://github.com/SJTU-IPADS) and Shanghai AI Laboratory. See [Contributors](CONTRIBUTORS.md).



## Contact Us

If you have any questions about DISB, feel free to contact us.

Weihang Shen: shenwhang@sjtu.edu.cn

Mingcong Han: mingconghan@sjtu.edu.cn

Rong Chen: rongchen@sjtu.edu.cn



## License

REEF is released under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).
