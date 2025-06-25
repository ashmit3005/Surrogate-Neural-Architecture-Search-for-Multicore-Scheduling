import streamlit as st
from workload import generate_workload
from simulator import Simulator
from policies.baseline import RoundRobinPolicy
from policies.neural_policy import NeuralSchedulerPolicy
from nas_controller import SurrogateModel
import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt

# Define model building logic
def build_model(arch, num_cores):
    layers = []
    input_dim = 3
    for _ in range(arch["num_layers"]):
        layers.append(nn.Linear(input_dim, arch["hidden_dim"]))
        layers.append(nn.ReLU() if arch["activation"] == "relu" else nn.Tanh())
        input_dim = arch["hidden_dim"]
    layers.append(nn.Linear(input_dim, num_cores))
    return nn.Sequential(*layers)

# UI: Streamlit App
st.title("Neural Scheduling Architecture Advisor")

scenario = st.selectbox("Choose a workload type:", ["balanced", "bursty", "real_time", "memory_bound"])
num_tasks = st.slider("Number of tasks", 10, 50, 20)
num_cores = st.slider("Number of cores", 2, 8, 4)
seed = st.number_input("Random seed", value=42)

if st.button("Run Simulation"):
    # Generate workload
    workload_meta = {"scenario": scenario}
    tasks = generate_workload(scenario, num_tasks=num_tasks, seed=seed)

    # Baseline Simulation
    baseline_policy = RoundRobinPolicy()
    sim1 = Simulator(num_cores=num_cores, scheduling_policy=baseline_policy)
    sim1.load_tasks(tasks)
    baseline_result = sim1.run(max_time=100)

    # Surrogate NAS
    surrogate = SurrogateModel()
    arch = surrogate.suggest(workload_meta=workload_meta)[0][0]  # Take best candidate
    model = build_model(arch, num_cores)
    neural_policy = NeuralSchedulerPolicy(model)
    sim2 = Simulator(num_cores=num_cores, scheduling_policy=neural_policy)
    sim2.load_tasks(tasks)
    neural_result = sim2.run(max_time=100)

    # Output
    st.subheader("Best Neural Architecture:")
    st.json(arch)

    st.subheader("Performance Metrics")
    st.write("**Baseline (Round Robin):**", baseline_result)
    st.write("**Neural Policy:**", neural_result)

    # Plot Comparison
    metrics = ["avg_latency", "avg_utilization", "deadline_misses", "num_completed"]
    baseline_vals = [baseline_result[m] for m in metrics]
    neural_vals = [neural_result[m] for m in metrics]

    fig, ax = plt.subplots()
    bar_width = 0.35
    x = range(len(metrics))
    ax.bar(x, baseline_vals, bar_width, label='Baseline')
    ax.bar([p + bar_width for p in x], neural_vals, bar_width, label='Neural')
    ax.set_xticks([p + bar_width/2 for p in x])
    ax.set_xticklabels(metrics)
    ax.legend()
    st.pyplot(fig)
