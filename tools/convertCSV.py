import os
import sys
import json

frameworks = ["trt", "triton", "tfs"]
workloads = ["A", "B", "C", "D", "E", "REAL"]
models = [
    "densenet201",
    "distilbert",
    "inceptionv3",
    "resnet152",
    "vgg19",
]

if __name__ == "__main__":
    resultsPath = sys.argv[1]
    csv = open(os.path.join(resultsPath, "results.csv"), "w+")
    
    csv.write(", , ")
    for model in models:
        csv.write("rt-latency, ")
    for model in models:
        csv.write("rt-latency-increase, ")
    for model in models:
        csv.write("rt-throughput, ")
    for model in models:
        csv.write("be-throughput, ")

    csv.write("\nframework, workload, ")
    for model in models:
        csv.write(model + ", ")
    for model in models:
        csv.write(model + ", ")
    for model in models:
        csv.write(model + ", ")
    for model in models:
        csv.write(model + ", ")
    csv.write("\n")

    for framework in frameworks:
        for workload in workloads:
            csv.write(framework + ", " + workload + ", ")
            
            jsonPath = os.path.join(resultsPath, framework + "." + workload + ".json")
            if not os.path.exists(jsonPath):
                csv.write("\n")
                continue
            jsonFile = json.load(open(jsonPath,'r'))
            
            clients = {}
            for client in jsonFile["results"]:
                clientName = client["clientName"]
                clients[clientName] = {
                    "throughput": client["analyzers"][0]["avgThroughput(req/s)"],
                    "latency": client["analyzers"][0]["avgTotalLatency(us)"],
                    "latencyIncrease": client["analyzers"][0]["avgTotalLatencyIncrease(us)"]
                }
            
            for model in models:
                if model + "_rt" in clients.keys():
                    csv.write(str(clients[model + "_rt"]["latency"]))
                csv.write(", ")
            for model in models:
                if model + "_rt" in clients.keys():
                    csv.write(str(clients[model + "_rt"]["latencyIncrease"]))
                csv.write(", ")
            for model in models:
                if model + "_rt" in clients.keys():
                    csv.write(str(clients[model + "_rt"]["throughput"]))
                csv.write(", ")
            for model in models:
                if model + "_be" in clients.keys():
                    csv.write(str(clients[model + "_be"]["throughput"]))
                csv.write(", ")
            csv.write("\n")
    csv.close()

