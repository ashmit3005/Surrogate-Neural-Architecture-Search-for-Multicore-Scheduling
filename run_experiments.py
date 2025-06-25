from nas_controller import SurrogateModel
from simulator import Simulator
from workload import generate_workload
from policies.neural_policy import NeuralSchedulerPolicy
import torch.nn as nn
import torch

def build_model(arch, num_cores):
    layers = []
    input_dim = 3  # arrival, duration, free_cores
    for _ in range(arch["num_layers"]):
        layers.append(nn.Linear(input_dim, arch["hidden_dim"]))
        layers.append(nn.ReLU() if arch["activation"] == "relu" else nn.Tanh())
        input_dim = arch["hidden_dim"]
    layers.append(nn.Linear(input_dim, num_cores))
    return nn.Sequential(*layers)

# Test loop
surrogate = SurrogateModel()
num_cores = 4

for iteration in range(3):  # try 3 NAS rounds
    candidates = surrogate.suggest()

    for arch, _ in candidates:
        model = build_model(arch, num_cores)
        policy = NeuralSchedulerPolicy(model)

        tasks = generate_workload("real_time", num_tasks=20, seed=iteration)
        sim = Simulator(num_cores=num_cores, scheduling_policy=policy)
        sim.load_tasks(tasks)
        result = sim.run(max_time=50)

        # Example: combine metrics into score (you can tweak this)
        score = -result["avg_latency"] - 2 * result["deadline_misses"]

        print(f"\nArch: {arch}")
        print("Result:", result)
        print("Score:", score)

        surrogate.update(arch, score)