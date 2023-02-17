# DISB: DNN Inference Serving Benchmark

**DISB** is a **D**NN **I**nference **S**erving **B**enchmark with diverse workloads and models. It was originally designed to simulate real-time scenarios, e.g. autonomous driving systems, where both low latency and high throughput are demanded.

DISB uses the client-server architecture, where the clients send the DNN inference requests to the server via RPC, and the server returns the inference result. Clients can submit the inference requests periodically or randomly. An inference request may contain the model name (or id), the input data and other customized attributes (e.g., priority or deadline). 

**Note:** Please use git lfs to clone this repo in order to download model files.



## Table of Contents

- [DISB Toolkit](#disb-toolkit)
- [DISB Workloads](#disb-workloads)
- [Build & Install](#build--install)
- [Usage](#usage)
- [Samples](#samples)
- [Benchmark Results](#benchmark-results)
- [Paper](#paper)
- [The Team](#the-team)
- [Contact Us](#contact-us)
- [License](#license)



## DISB Toolkit

DISB provides a C++ library (`libdisb`) to perform benchmarking. To integrate your own DNN inference system with DISB, you only need to implement `DISB::Client` to wrap your inference interface. See [usage](#usage) for details.



## DISB Workloads

Currently, DISB provides 5 workloads with different DNN models and different number of clients. 

There are three pattern for submitting inference requests in DISB clients:
1. Uniform Distribution (U): The client sends inference requests periodically, with a fixed frequency (e.g., 20 reqs/s). This pattern is common in data-driven applications (e.g., obstacle detection with cameras).
2. Poisson Distribution (P): The client sends inference requests in a Poisson distribution pattern with a given average arrival speed (e.g., 25 reqs/s). This pattern can simulate event-driven applications (e.g., speech recoginition).
3. Closed-loop (C): The client continuously sends inference requests, which simulates a contention load.
4. Trace (T): The client sends inference requests according to a given trace file which contains a series of request time points. This pattern can reproduce real world workloads.
5. Dependent (D): The client sends inference requests when all prior tasks have completed, prior tasks can be other clients. This pattern can simulate inference graph (or inference DAG), where a model need the output of another model as its input.

We combined these patterns into 6 typical workloads for benchmarks, see [workloads](benchmarks/workloads/workloads.md) for workload details.

[TBD] We're still working on providing more representative and general DNN inference serving workloads.



## Build & Install

Install dependencies:

```shell
sudo apt install build-essential cmake
sudo apt install libjsoncpp-dev
```

Build and install DISB tools:

```shell
# will build and install into disb/install
make build
```



## Usage

- ### Client

  `DISB::Client` is an adaptor class between DISB and the serving backend. You can implement the following interfaces in its subclass. These interfaces will be called during the benchmark, and their execution time will be recorded by DISB.

  ```c++
  # init() will be called once when the benchmark begins
  virtual void init();
  
  # The following interfaces will be called by DISB
  # within each inference request during benchmark.
  # Average latency of each interface will be recorded.
  virtual void prepareInput();
  virtual void preprocess();
  virtual void copyInput();
  virtual void infer();
  virtual void copyOutput();
  virtual void postprocess();
  
  # If another task dependents on this client,
  # the InferResult will be passed to the next task.
  virtual std::shared_ptr<InferResult> produceResult();
  ```

  

- ### Load

  `DISB::Load` instructs when DISB should launch the next inference request. There are 5 built-in loads simulating the load patterns mentioned in [DISB Workloads](#disb-workloads). They can be enabled by setting certain attributes in json configuration, see [HelloDISB](samples/hellodisb) for example.
  
  
  
  If you want to use `DISB::DependentLoad`, your client class should inherit `DISB::DependentClient` and implement the virtual methods `consumePrevResults()` and `produceDummyPrevResults()`. `consumePrevResults()` will be called when one of the prior tasks finished one inference and produced one result. You can use the previous results as the input of the DependentClient. You can also inherit `DISB::InferResult` to pass custom data. `produceDummyPrevResults()` will be called when DISB
  
  is warming up and testing the standalone latency of each client. The results will be consumed by `consumePrevResults()`, making a dependent load  become independent in order to measure standalone latency.
  
  
  
- ### BenchmarkSuite

  `DISB::BenchmarkSuite` should be created and initialized before the benchmark is launched.

  ```c++
  void init(const std::string &configJsonStr,
            std::shared_ptr<Client> clientFactory(const Json::Value &config),
            std::shared_ptr<Load> loadFactory(const Json::Value &config) = builtinLoadFactory);
  
  void run(void loadCoordinator(const std::vector<LoadInfo> &loadInfos) = builtinLoadCoordinator);
  ```

  When initializing BenchmarkSuite, a json formatted string should be passed as config, and a factory method of your own subclass implementation of `DISB::Client` should be provided. The `Json::Value` passed to the factory method is the `"client"` attribute in each task in `configJsonStr`.

  

  If you need customized loads other than the built-in loads, you should implement the virtual method `waitUntilNextLaunch()` and provide your own load factory method. The `Json::Value` passed to the factory method is the `"load"` attribute in each task in `configJsonStr`.
  
  
  
  If your loads need to coordinate with each other, you can pass `loadCoordinator()` to `DISB::BenchmarkSuite::run()`, which makes sure that loads will not conflict with each other. For example, the `builtinLoadCoordinator()` will prevent the periodic loads with the same frequency and the highest priority from launching at the same time by setting different launch delay.
  
  
  
- ### Analyzer

  `DISB::Analyzer` is used to measure the performance of each inference task, each inference task can have multiple analyzers. `DISB::BasicAnalyzer`, which can measure latency and throughput, is implemented by DISB and is enabled for every task by default.

  

  If you want customized analyzers other than `DISB::BasicAnalyzer`, for example, an analyzer that measures gpu usage and memory consumption, the following interfaces should be implemented.

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

  
  
  After you have implemented `DISB::Analyzer`, you can add it to a `DISB::Client` by calling `DISB::Client::addAnalyzer()` in the factory method of client. You may refer to [TensorRT sample](benchmarks/frameworks/tensorrt/README.md) or [Tensorflow Serving sample](benchmarks/frameworks/tfserving/README.md) for more details. They both implement an `AccuarcyAnalyzer` to measure inference accuarcy.
  
  
  
- ### Run benchmarks

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



## Benchmark Results

We have supported DISB on some mainstream DNN inference serving frameworks, including:
- [TensorRT](benchmarks/frameworks/tensorrt/README.md)
- [Triton](benchmarks/frameworks/triton/README.md)
- [Tensorflow Serving](benchmarks/frameworks/tfserving/README.md)

We tested these DNN inference serving frameworks under 6 [DISB Workloads](#disb-workloads). Test results are shown in [results.md](benchmarks/results/results.md).

[TBD] We're still working on supporting more DNN inference serving frameworks.



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

DISB is developed and maintained by members from [IPADS@SJTU](https://github.com/SJTU-IPADS) and Shanghai AI Laboratory. See [Contributors](CONTRIBUTORS.md).



## Contact Us

If you have any questions about DISB, feel free to contact us.

Weihang Shen: shenwhang@sjtu.edu.cn

Mingcong Han: mingconghan@sjtu.edu.cn

Rong Chen: rongchen@sjtu.edu.cn



## License

DISB is released under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).