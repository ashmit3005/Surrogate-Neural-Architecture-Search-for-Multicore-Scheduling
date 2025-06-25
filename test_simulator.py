from simulator import Task, Simulator, Core
from policies.baseline import RoundRobinPolicy

# Custom workload with deadlines and memory demands
tasks = [
    Task(0, arrival_time=0, duration=5, priority=1),
    Task(1, arrival_time=1, duration=3, priority=2),
    Task(2, arrival_time=2, duration=2, priority=1),
]

# Add optional fields manually
tasks[0].deadline = 4  # will miss
tasks[1].deadline = 6  # on time
tasks[2].deadline = 6  # on time

tasks[0].memory_demand = 3
tasks[1].memory_demand = 2
tasks[2].memory_demand = 5  # intentionally high

# Define heterogeneous cores
cores = [
    Core(0, memory_capacity=2, speed=1.0),  # fast but low memory
    Core(1, memory_capacity=6, speed=0.5),  # slow but high memory
]

# Plug into simulator
sim = Simulator(num_cores=2, scheduling_policy=RoundRobinPolicy())
sim.cores = cores  # override default cores
sim.load_tasks(tasks)
results = sim.run(max_time=20)

print("Simulation Result:", results)
